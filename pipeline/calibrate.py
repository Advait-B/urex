"""calibrate.py - measure how line position maps to height (the triangulation).

WHY MEASURE INSTEAD OF USING THE FORMULA
The optics sheet predicts  shift_px = z*tan(theta)*f / ((L + z - f)*pitch).
Two problems with plugging numbers into it:
  * it is CURVED: about 4.0 px per mm near the tread surface, 3.2 px per mm at
    20 mm. A single "mm per pixel" number reads a 16 mm groove as 13.5 mm.
  * every input is uncertain: 1 degree of laser angle = 4 % error, the lens
    focal length is +-5 %, the bracket is hand-made, and the lens bends straight
    lines a little near the edges.
So we photograph surfaces at KNOWN heights and let a polynomial learn the
mapping for each column. That one fit soaks up the angle, lens, distortion and
laser tilt. Nothing needs to be measured with a protractor.

HOW (about 15 minutes on the bench)
  1. Flat, matt, non-shiny plate (painted aluminium, MDF, a ceramic tile).
     Put the device on it with its feet touching the plate. Press B1:
     you get shim_0.0_A (laser off) and shim_0.0_B (laser on).
  2. Put spacers of known thickness under all three feet, so the plate is
     that much further away. Measure the spacers with calipers, don't trust
     the label. Press B1 again. Rename the pair shim_<mm>_A / shim_<mm>_B.
  3. Repeat to cover 0 to about 20 mm (a new bus tyre has ~16-18 mm grooves),
     e.g. 0, 2, 4, 6, 8, 10, 12, 15, 18, 20. More heights = better; at least 6.
  4. python calibrate.py photos/calib/ --write calib.json

It finds the laser line in every pair (same code as laser_profile.py), then
fits ONE smooth surface  z = sum c_ij * u^i * v^j  (u = column, v = row, both
scaled to 0..1; cubic in the row direction, quadratic across). Every valid
column of every shot is a data point, so ~3000 points pin down 12 numbers and
the noise averages out. It prints how far each shim's measured height is from
its true height: that is your calibration accuracy. Aim for < 0.05 mm.
Don't move the lens or the laser after calibrating. If you do, recalibrate.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import laser_profile as lp


def load_shims(folder, calib):
    """Return [(shim_mm, y_per_column, contrast)] for every shim_<mm>_B.* in folder."""
    out = []
    for b in sorted(glob.glob(os.path.join(folder, "shim_*_B.*")),
                    key=lambda p: float(re.search(r"shim_([0-9.]+)_B", p).group(1))):
        mm = float(re.search(r"shim_([0-9.]+)_B", os.path.basename(b)).group(1))
        a = re.sub(r"_B(\.\w+)$", r"_A\1", b)
        on = lp.load_gray(b, calib["rotate180"])
        off = lp.load_gray(a, calib["rotate180"]) if os.path.exists(a) else None
        diff = lp.smooth(lp.subtract(on, off), sigma=max(0.7, calib["line_sigma_px"] * 0.6))
        y, contrast = lp.find_line(diff, min_contrast=calib["min_contrast"])
        out.append((mm, y, contrast, diff.shape))
    return out


def fit_height_poly(shots, deg_x=2, deg_y=3):
    """shots: [(z_mm, y_per_column, shape)]. Least-squares fit of z(u, v)."""
    h, w = shots[0][2]
    U, V, Z = [], [], []
    for z, y, shape in shots:
        if shape != (h, w):
            raise ValueError("all calibration shots must be the same image size")
        x = np.arange(w)
        ok = ~np.isnan(y)
        U.append(x[ok] / (w - 1))
        V.append(y[ok] / (h - 1))
        Z.append(np.full(ok.sum(), z))
    U, V, Z = map(np.concatenate, (U, V, Z))
    M = lp.poly_design(U, V, deg_x, deg_y)
    coef, *_ = np.linalg.lstsq(M, Z, rcond=None)
    return {"deg_x": deg_x, "deg_y": deg_y, "w": int(w), "h": int(h), "coef": coef.tolist(),
            "z_range": [float(min(s[0] for s in shots)), float(max(s[0] for s in shots))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with shim_<mm>_A / shim_<mm>_B photo pairs")
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json"),
                    help="existing calib.json to read settings (rotate180, min_contrast) from")
    ap.add_argument("--write", help="calib.json to update with the measured values")
    ap.add_argument("--deg-x", type=int, default=2)
    ap.add_argument("--deg-y", type=int, default=3)
    args = ap.parse_args()

    calib = dict(lp.DEFAULT_CALIB)
    if os.path.exists(args.calib):
        with open(args.calib) as fh:
            calib.update({k: v for k, v in json.load(fh).items() if not k.startswith("_")})

    shims = load_shims(args.folder, calib)
    if len(shims) < args.deg_y + 2:
        raise SystemExit(f"found {len(shims)} shim pairs; need at least {args.deg_y + 2} heights for a "
                         f"degree-{args.deg_y} fit (more is better)")
    for mm, y, con, _ in shims:
        print(f"shim {mm:5.2f} mm: line at row {np.nanmedian(y):7.2f}  valid {np.mean(~np.isnan(y))*100:3.0f} %"
              f"  contrast {np.nanmean(con):4.0f}")

    hp = fit_height_poly([(mm, y, shape) for mm, y, _, shape in shims], args.deg_x, args.deg_y)

    # ---- how good is it? predict each shim's height back from its own photo.
    #   bias  = median error over the columns: a systematic calibration error
    #   noise = RMS over single columns: random scatter, averaged away inside a groove
    print(f"\n{'shim mm':>8s} {'measured':>9s} {'bias':>7s} {'noise':>6s}   | straight-line guess (old method)")
    rows = np.array([np.nanmedian(y) for _, y, _, _ in shims])
    zs = np.array([mm for mm, *_ in shims])
    k, r0 = np.polyfit(zs, rows, 1)                  # old method: one px/mm for everything
    worst_bias, worst_noise = 0.0, 0.0
    for mm, y, _, shape in shims:
        x = np.arange(len(y), dtype=float)
        err = lp.height_mm(x, y, {**calib, "height_poly": hp}, shape) - mm
        bias, noise = np.nanmedian(err), np.sqrt(np.nanmean(err ** 2))
        worst_bias, worst_noise = max(worst_bias, abs(bias)), max(worst_noise, noise)
        lin = (np.nanmedian(y) - r0) / k
        print(f"{mm:8.2f} {mm + bias:9.3f} {bias:+7.3f} {noise:6.3f}   | {lin:7.2f} ({lin - mm:+.2f})")
    print(f"\nworst bias {worst_bias:.3f} mm" + ("  OK" if worst_bias < 0.05 else
          "  <- high: spacers unequal, plate not flat, or a shim value typed wrong"))
    print(f"single-column noise up to {worst_noise:.3f} mm" + ("  OK" if worst_noise < 0.15 else
          "  <- high: shiny plate, laser too dim/saturated, or out of focus"))
    print(f"average slope {abs(k):.3f} px per mm (about {1/abs(k):.3f} mm per px)")

    if args.write:
        out = {}
        if os.path.exists(args.write):
            with open(args.write) as fh:
                out = json.load(fh)
        out.update({"height_poly": hp, "mm_per_px_depth": round(1 / abs(k), 4), "depth_sign": 1 if k > 0 else -1,
                    "_calibrated_from": [f"shim {mm} mm -> row {r:.2f}" for mm, r in zip(zs, rows)]})
        with open(args.write, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote height_poly into {args.write}")


if __name__ == "__main__":
    main()
