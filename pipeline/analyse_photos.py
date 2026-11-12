"""analyse_photos.py - run the whole single-photo pipeline over a folder.

    python analyse_photos.py PHOTO_FOLDER --out results/

For every photo:
  1. signal processing (single_frame.py): laser points, surface, groove apex
     or lug steps, depth in px
  2. ML gate (frame_classifier.py): P(usable) from the trained model
  3. measurement checks: the surface fit must agree with >= 50 % of the
     points (groove mode) or at least one clean lug step must be found
  4. px -> mm with the DESIGN triangulation geometry. NOT CALIBRATED: the
     mm values are estimates with a wide range (see README section "How sure
     are the mm values?"). The shim calibration replaces this.

Writes into --out:
  results.csv          one row per photo (every number below)
  summary.json         per-scene medians, spreads, classifier scores
  frames/<photo>.png   raw vs cleaned figure for every photo with a laser line
                       (photos of people are never drawn)
  overview.png         all accepted measurements, px and mm
  profiles.png         the cleaned groove profiles of the truck-tyre frames overlaid
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

import frame_classifier as fc
import report_figs as rf
import single_frame as sf

MIN_INLIERS_GROOVE = 0.5
MAX_GROOVE_PX = 80          # bigger than any groove at our scale: the line ran along an edge
Z_RANGE_MM = (70.0, 110.0)  # hand-held distance we can't rule out without a fixed stand-off


def measurement_check(res):
    """Second gate, after the classifier: does the measurement itself look sane?"""
    if res["mode"] == "groove":
        if res["surface"]["inlier_frac"] < MIN_INLIERS_GROOVE:
            return False, f"surface fit agrees with only {res['surface']['inlier_frac']:.0%} of points"
        if res["depth_px"] > MAX_GROOVE_PX:
            return False, f"apex {res['depth_px']:.0f} px is implausibly deep (line ran along an edge?)"
        # a groove only ever moves the line AWAY from the camera. A big jog the other
        # way means the line is crossing a block edge or a shadow, not a clean groove.
        near = [e["depth_px"] for e in res["excursions"] if e["direction"] == "near" and e["n"] >= 6]
        if near and max(near) > 0.5 * res["depth_px"]:
            return False, f"line also jogs {max(near):.0f} px towards the camera (block edge, not a groove)"
        return True, ""
    if res["mode"] == "steps":
        if res["row_coverage"] < 0.4:
            return False, "laser line covers < 40 % of rows"
        return True, ""
    return False, "no groove or step found"


def mm_estimates(depth_px, width):
    """design-geometry mm at the nominal 90 mm distance, and at the ends of Z_RANGE_MM"""
    nom = sf.fixed_scale_mm(depth_px, width, 90.0)
    lo = sf.fixed_scale_mm(depth_px, width, Z_RANGE_MM[0])
    hi = sf.fixed_scale_mm(depth_px, width, Z_RANGE_MM[1])
    return nom, lo, hi


def jitter(xs):
    """row-to-row wobble of a profile (px): median |x[i+1] - x[i]|"""
    xs = np.asarray(xs, float)
    return float(np.median(np.abs(np.diff(xs)))) if len(xs) > 2 else float("nan")


def raw_vs_clean_numbers(gray, res):
    """How much the processing cleans up: points that land on the laser, and wobble."""
    rx, ry, _ = sf.raw_profile(gray)
    if res["surface"] is None or not len(res["_x"]):
        return {}
    sx = sf.surface_x(res["surface"]["coef"], ry)
    on_line_raw = float(np.mean(np.abs(rx - sx) < 3))
    inl = res["_inl"]
    o = np.argsort(res["_y"][inl])
    # raw_on_line_frac: share of rows where the naive "brightest pixel" lands on the laser
    #                   (within 3 px); the rest is glare, background edges or the wrong branch
    # clean_rms:        scatter of the cleaned sub-pixel points about the fitted surface
    return {"raw_on_line_frac": on_line_raw,
            "clean_jitter_px": jitter(res["_x"][inl][o]),
            "clean_rms_to_surface_px": float(np.sqrt(np.mean(res["_r"][inl] ** 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    labels = {}
    if os.path.exists(fc.LABELS):
        labels = {r["file"]: r for r in csv.DictReader(open(fc.LABELS))}
    with open(fc.MODEL) as fh:
        model = json.load(fh)

    rows, keep_profiles = [], []
    for name in sorted(f for f in os.listdir(args.folder) if f.lower().endswith((".jpg", ".png"))):
        g = cv2.imread(os.path.join(args.folder, name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        res = sf.analyse(g)
        f = fc.features(g, res)
        p = fc.predict(f, model)
        ok_meas, why = measurement_check(res)
        accepted = p >= model["threshold"] and ok_meas
        lab = labels.get(name, {})
        scene = lab.get("scene", "unknown")
        feature = lab.get("feature", "")
        row = {"file": name, "scene": scene, "feature": feature, "label_usable": lab.get("usable", ""),
               "p_usable": round(p, 3), "mode": res["mode"],
               "depth_px": None if res["depth_px"] is None else round(res["depth_px"], 2),
               "accepted": int(accepted),
               "reason": "" if accepted else (why if p >= model["threshold"] else f"classifier: P(usable) {p:.2f}"),
               "surface_inliers": round(res["surface"]["inlier_frac"], 3) if res["surface"] else None,
               "line_x_at_centre": round(res["surface"]["x_at_centre"], 1) if res["surface"] else None,
               "n_steps": len(res["steps"]),
               "steps_px": " ".join(f"{s['step_px']:.1f}" for s in res["steps"])}
        if res["depth_px"] is not None:
            nom, lo, hi = mm_estimates(res["depth_px"], g.shape[1])
            row.update(depth_mm_nominal=round(nom, 1), depth_mm_low=round(lo, 1), depth_mm_high=round(hi, 1))
        row.update({k: round(v, 3) for k, v in raw_vs_clean_numbers(g, res).items()})
        row.update({f"f_{k}": round(v, 3) for k, v in f.items()})
        rows.append(row)
        if accepted and res["mode"] == "groove":
            keep_profiles.append((name, res))
        draw = scene != "people" and (lab.get("laser") == "1" or p >= 0.5)
        if draw and not args.no_figures:
            tag = "ACCEPTED" if accepted else f"rejected: {row['reason']}"
            rf.frame_figure(g, res, os.path.join(args.out, "frames", name.rsplit(".", 1)[0] + ".png"),
                            title=f"{name}   P(usable) {p:.2f}   {tag}")

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- per-scene summary of accepted measurements
    summary = {"photos": len(rows), "accepted": sum(r["accepted"] for r in rows), "scenes": {},
               "note": "mm values use the uncalibrated design geometry; see README"}
    for scene in sorted(set((r["scene"], r["feature"]) for r in rows)):
        acc = [r for r in rows if (r["scene"], r["feature"]) == scene and r["accepted"]]
        scene = scene[0] + (f" / {scene[1]}" if scene[1] else "")
        if not acc:
            continue
        px = np.array([r["depth_px"] for r in acc])
        mm = np.array([r["depth_mm_nominal"] for r in acc])
        summary["scenes"][scene] = {
            "frames": [r["file"] for r in acc], "mode": acc[0]["mode"],
            "depth_px_median": round(float(np.median(px)), 1),
            "depth_px_iqr": [round(float(np.percentile(px, 25)), 1), round(float(np.percentile(px, 75)), 1)],
            "depth_mm_nominal_median": round(float(np.median(mm)), 1),
            "depth_mm_range_at_median": [round(v, 1) for v in mm_estimates(float(np.median(px)), 320)[1:]],
        }
    lab_rows = [r for r in rows if r["label_usable"] != ""]
    if lab_rows:
        y = np.array([int(r["label_usable"]) for r in lab_rows])
        pr = np.array([int(r["p_usable"] >= 0.5) for r in lab_rows])
        summary["classifier_on_training_photos"] = fc.scores(y, pr)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if not args.no_figures:
        overview_figure(rows, os.path.join(args.out, "overview.png"))
        profiles_figure(keep_profiles, os.path.join(args.out, "profiles.png"))
    print(json.dumps(summary, indent=2))


def overview_figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    acc = [r for r in rows if r["accepted"]]
    if not acc:
        return
    acc.sort(key=lambda r: (r["scene"], r["file"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.32 * len(acc) + 1.6), sharey=True)
    ys = np.arange(len(acc))
    col = {"truck_tyre": "#2a6fbd", "rover_tyre": "#d9822b", "flat": "#6a6a6a"}
    for i, r in enumerate(acc):
        c = col.get(r["scene"], "k")
        ax[0].plot(r["depth_px"], i, "o", color=c)
        ax[1].plot([r["depth_mm_low"], r["depth_mm_high"]], [i, i], "-", color=c, alpha=0.35, lw=5)
        ax[1].plot(r["depth_mm_nominal"], i, "o", color=c)
    ax[0].set_yticks(ys)
    ax[0].set_yticklabels([f"{r['file'][6:21]}  ({r['scene']}, {r['feature'] or r['mode']})" for r in acc], fontsize=8)
    ax[0].set_xlabel("depth measured (px) - this is the real measurement")
    ax[1].set_xlabel("≈ depth (mm), design geometry: dot at 90 mm, bar = 70-110 mm stand-off")
    ax[1].axvspan(14, 15.5, color="green", alpha=0.12)
    ax[1].text(14.75, len(acc) - 0.5, "new truck/bus\ntyre 14-15.5 mm", ha="center", va="top", fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def profiles_figure(items, path):
    """cleaned groove profiles, lined up at their apex, to show frame-to-frame agreement"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not items:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    sgn = -1 if sf.GEOMETRY["far_is_left"] else 1
    for name, res in items:
        g = res["groove"]
        sel = g["idx"]
        yy = res["_y"][sel] - g["apex_y"]
        o = np.argsort(yy)
        ax.plot(yy[o], sgn * res["_r"][sel][o], ".", ms=3, label=f"{name[15:21]}: {res['depth_px']:.1f} px")
    ax.set_xlabel("rows from the groove apex")
    ax.set_ylabel("depth below the surface line (px)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("cleaned groove profiles (truck tyre), aligned at the apex")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
