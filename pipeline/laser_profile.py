"""laser_profile.py - turn a laser-line photo pair into tread depths.

    python laser_profile.py B.png                 (laser-on frame only)
    python laser_profile.py B.png --off A.png     (laser-off frame too: much cleaner)
    python laser_profile.py B.png --off A.png --calib calib.json --plot result.png --json result.json

The seven steps, in order (each is a small function below so you can test them
one at a time):

  1. load + rotate      read the frames as grey, undo the upside-down mounting
  2. subtract           B - A removes the room light and the rubber texture,
                        leaving (almost) only the laser line
  3. smooth             a small vertical Gaussian blur kills speckle without
                        moving the line centre (a Gaussian stays a Gaussian)
  4. find the line      in every COLUMN find the brightest row, then refine it
                        to a fraction of a pixel with an intensity-weighted
                        centroid in a +-3 px window. Reject columns whose peak
                        is too dim, too wide, or too far from their neighbours.
  5. triangulate        turn each column's line row into an absolute height
                        z (mm beyond the reference plane) with the calibration
                        polynomial z(column, row) that calibrate.py measured.
                        Without one, fall back to row * mm_per_px_depth (only
                        good to ~0.3 mm for shallow grooves; see calibrate.py).
  6. shoulder + depth   the tread surface is the straight line (in mm) most
                        columns agree on. RANSAC finds it even though the groove
                        columns are far away from it. depth = z - shoulder.
  7. grooves            runs of columns deeper than 0.5 mm and wider than 4 mm
                        (after a 5-column median to stop noise splitting a run).
                        The 0.5 mm threshold is deliberately far below the 1.6 mm
                        legal limit: a worn groove must be FOUND, then fail.
                        Each groove's depth = median of its central half.
                        PASS if every groove >= 1.6 mm. If expected_grooves is
                        set and fewer are found -> CHECK (a bald groove
                        looks like no groove at all).

Everything is plain numpy so it can be ported to the OpenMV board (ulab) later:
see openmv_profile.py for the port of steps 2-4.
"""
import argparse
import json
import os

import cv2
import numpy as np

DEFAULT_CALIB = {
    "mm_per_px_depth": 0.249, "mm_per_px_lateral": 0.35, "depth_sign": 1, "rotate180": True,
    "min_contrast": 40, "line_sigma_px": 1.5, "groove_min_depth_mm": 0.5,
    "groove_min_width_mm": 4.0, "legal_min_depth_mm": 1.6, "expected_grooves": None,
}


# ---------------------------------------------------------------- 1. load
def load_gray(path, rotate180=False):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f"cannot read {path}")
    if rotate180:
        g = cv2.rotate(g, cv2.ROTATE_180)
    return g.astype(np.float32)


# ------------------------------------------------------------ 2. subtract
def subtract(on, off=None):
    """Laser-on minus laser-off, clipped at 0. Without an off frame we fall
    back to removing each column's background level (its 20th percentile)."""
    if off is not None:
        d = on - off
    else:
        d = on - np.percentile(on, 20, axis=0, keepdims=True)
    return np.clip(d, 0, None)


# -------------------------------------------------------------- 3. smooth
def smooth(diff, sigma=1.0):
    # blur mostly along rows (vertical), only a touch sideways so a groove
    # edge stays sharp in the across-tread direction
    return cv2.GaussianBlur(diff, (0, 0), sigmaX=0.5, sigmaY=sigma)


# -------------------------------------------------------- 4. find the line
def find_line(diff, min_contrast=40, win=3, max_fwhm_px=14, jump_px=4.0):
    """Return y (row of the laser line, sub-pixel) per column, NaN where invalid,
    plus the peak contrast per column."""
    h, w = diff.shape
    peak_row = diff.argmax(axis=0)                  # brightest row in each column
    contrast = diff[peak_row, np.arange(w)]
    y = np.full(w, np.nan)
    fwhm = np.full(w, np.nan)
    rows = np.arange(h, dtype=np.float32)

    for c in range(w):
        p = peak_row[c]
        if contrast[c] < min_contrast:
            continue
        lo, hi = max(0, p - win), min(h, p + win + 1)
        seg = diff[lo:hi, c]
        # intensity-weighted centroid: sum(row * I) / sum(I). Subtracting half
        # the peak first stops the dim tails from pulling the centre around.
        wgt = np.clip(seg - 0.5 * contrast[c], 0, None)
        if wgt.sum() <= 0:
            continue
        y[c] = (rows[lo:hi] * wgt).sum() / wgt.sum()
        # width check: how many rows are above half the peak? a laser line is a
        # few px; a bright reflection or a blown-out blob is much wider
        fwhm[c] = (diff[:, c] > 0.5 * contrast[c]).sum()

    y[fwhm > max_fwhm_px] = np.nan

    # continuity: a real surface does not jump 4 px between neighbouring columns
    # (a groove edge does, but then the next columns agree with each other).
    med = median_nan(y, 5)
    jump = np.abs(y - med) > jump_px
    y[jump] = np.nan
    return y, contrast


def median_nan(v, k):
    """running median of width k that ignores NaNs"""
    out = np.full_like(v, np.nan)
    r = k // 2
    for i in range(len(v)):
        seg = v[max(0, i - r): i + r + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


# ------------------------------------------------------- 5. shoulder line
def ransac_line(x, y, thresh_px=1.0, iters=300, seed=0):
    """Fit y = a*x + b to the largest group of points that agree within
    thresh_px. Grooves are the outliers, the tread surface is the inliers."""
    ok = ~np.isnan(y)
    xs, ys = x[ok], y[ok]
    if len(xs) < 10:
        raise ValueError("too few valid columns to fit the shoulder line")
    rng = np.random.default_rng(seed)
    best_in, best = None, (0.0, float(np.nanmedian(ys)))
    for _ in range(iters):
        i, j = rng.choice(len(xs), 2, replace=False)
        if xs[i] == xs[j]:
            continue
        a = (ys[j] - ys[i]) / (xs[j] - xs[i])
        b = ys[i] - a * xs[i]
        inl = np.abs(ys - (a * xs + b)) < thresh_px
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (a, b)
    # refine with least squares on the inliers
    a, b = np.polyfit(xs[best_in], ys[best_in], 1)
    return a, b, best_in.sum() / len(xs)


# -------------------------------------------------------- 5. triangulate
def poly_design(u, v, deg_x, deg_y):
    """Columns u^i * v^j for i <= deg_x, j <= deg_y (u, v already scaled to 0..1)."""
    return np.stack([u ** i * v ** j for i in range(deg_x + 1) for j in range(deg_y + 1)], axis=-1)


def height_mm(x, y, calib, shape):
    """Line row y (px) in column x -> surface distance z (mm) beyond the reference plane.

    With calib["height_poly"] (written by calibrate.py) this is the measured
    curve for each column, which already includes the laser angle, the lens,
    its distortion and any tilt of the laser vs the sensor. Without it, it is
    the straight-line guess z = sign * y * mm_per_px_depth (offset unknown, which
    is fine because only differences from the shoulder are used)."""
    hp = calib.get("height_poly")
    if hp is None:
        return calib["depth_sign"] * y * calib["mm_per_px_depth"]
    h, w = shape
    if (hp["w"], hp["h"]) != (w, h):
        raise ValueError(f"calibration was made at {hp['w']}x{hp['h']} but this image is {w}x{h}: "
                         f"recalibrate or change FRAMESIZE back")
    u, v = x / (w - 1), y / (h - 1)
    z = poly_design(np.nan_to_num(u), np.nan_to_num(v), hp["deg_x"], hp["deg_y"]) @ np.asarray(hp["coef"])
    z[np.isnan(y)] = np.nan
    # outside the calibrated range the polynomial is a guess: refuse it
    lo, hi = hp.get("z_range", (-np.inf, np.inf))
    margin = 0.1 * (hi - lo)
    z[(z < lo - margin) | (z > hi + margin)] = np.nan
    return z


# ------------------------------------------------------------ 7. grooves
def find_grooves(depth_mm, calib):
    # decide WHERE the grooves are on a median-smoothed copy, so one noisy column
    # can't split a groove into pieces too narrow to count; measure on the raw data
    sm = median_nan(depth_mm, 5)
    deep = np.where(np.isnan(sm), False, sm > calib["groove_min_depth_mm"])
    grooves = []
    i, w = 0, len(depth_mm)
    while i < w:
        if not deep[i]:
            i += 1
            continue
        j = i
        while j < w and deep[j]:
            j += 1
        width_mm = (j - i) * calib["mm_per_px_lateral"]
        if width_mm >= calib["groove_min_width_mm"]:
            q = (j - i) // 4                                # ignore the sloping walls
            core = depth_mm[i + q: j - q] if j - i >= 8 else depth_mm[i:j]
            grooves.append({"col_start": int(i), "col_end": int(j), "width_mm": round(width_mm, 2),
                            "depth_mm": round(float(np.nanmedian(core)), 3),
                            "depth_max_mm": round(float(np.nanmax(core)), 3)})
        i = j
    return grooves


# ----------------------------------------------------------------- driver
def analyse(on_path, off_path=None, calib=None):
    calib = {**DEFAULT_CALIB, **(calib or {})}
    on = load_gray(on_path, calib["rotate180"])
    off = load_gray(off_path, calib["rotate180"]) if off_path else None
    diff = smooth(subtract(on, off), sigma=max(0.7, calib["line_sigma_px"] * 0.6))
    y, contrast = find_line(diff, min_contrast=calib["min_contrast"])
    x = np.arange(len(y), dtype=np.float64)
    z = height_mm(x, y, calib, diff.shape)                 # 5. triangulate
    a, b, inlier_frac = ransac_line(x, z, thresh_px=calib.get("ransac_thresh_mm", 0.3))
    depth = z - (a * x + b)                                # 6. + = below the tread surface
    grooves = find_grooves(depth, calib)
    warnings = []
    # Grooves are always LOWER than the tread surface. If the big deviations from
    # the shoulder line all come out negative, depth_sign in calib.json is wrong.
    big = depth[~np.isnan(depth)]
    big = big[np.abs(big) > calib["groove_min_depth_mm"]]
    if len(big) > 10 and np.median(big) < 0:
        warnings.append("most deviations are negative: flip depth_sign in calib.json (or rotate180)")

    calibrated = "height_poly" in calib
    if not calibrated:
        warnings.append("no height_poly in calib.json: using the straight-line mm_per_px_depth "
                        "(deep grooves read too shallow). Run calibrate.py.")
    valid_frac = float(np.mean(~np.isnan(z)))
    mean_contrast = float(np.nanmean(contrast[~np.isnan(z)])) if valid_frac else 0.0
    quality_ok = valid_frac >= 0.9 and mean_contrast > 60       # the CHECK state in the firmware sketch
    worst = min((g["depth_mm"] for g in grooves), default=None)
    result = {
        "on": on_path, "off": off_path,
        "valid_columns_frac": round(valid_frac, 3),
        "mean_contrast": round(mean_contrast, 1),
        "shoulder_line": {"slope_mm_per_col": float(a), "intercept_mm": float(b),
                          "inlier_frac": round(float(inlier_frac), 3),
                          # how far the tread surface sat beyond the reference plane, at the
                          # frame centre. Only meaningful when calibrated; expect 0-3 mm.
                          "standoff_mm": round(float(a * (len(x) - 1) / 2 + b), 3) if calibrated else None},
        "grooves": grooves,
        "min_groove_depth_mm": worst,
        "quality_ok": bool(quality_ok),
        "warnings": warnings,
        "verdict": verdict(grooves, worst, quality_ok, calib),
    }
    return result, dict(diff=diff, y=y, z=z, x=x, a=a, b=b, depth=depth, on=on)


def verdict(grooves, worst, quality_ok, calib):
    if not quality_ok:
        return "RETAKE"                       # can't judge this shot: take it again
    if grooves and worst < calib["legal_min_depth_mm"]:
        return "FAIL"                         # a groove under the legal limit
    exp = calib.get("expected_grooves")
    if not grooves or (exp and len(grooves) < exp):
        return "CHECK"                        # missing groove: worn flat, or line off the tread
    return "PASS"


def plot(result, inter, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                           gridspec_kw={"height_ratios": [2, 2, 1.4]})
    ax[0].imshow(inter["on"], cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax[0].set_title("laser-on frame")
    ax[1].imshow(inter["diff"], cmap="magma", aspect="auto")
    ax[1].plot(inter["x"], inter["y"], ".", ms=2, color="cyan", label="line centre (sub-pixel)")
    ax[1].set_title("on - off, with detected line")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(inter["x"], inter["depth"], color="tab:blue")
    ax[2].axhline(1.6, color="tab:red", ls="--", lw=1, label="legal min 1.6 mm")
    for g in result["grooves"]:
        ax[2].axvspan(g["col_start"], g["col_end"], color="tab:orange", alpha=0.25)
        ax[2].text((g["col_start"] + g["col_end"]) / 2, g["depth_mm"] + 0.3, f"{g['depth_mm']:.2f} mm",
                   ha="center", fontsize=8)
    ax[2].set_ylabel("depth (mm)")
    ax[2].set_xlabel("image column (across tread)")
    ax[2].invert_yaxis()
    ax[2].legend(loc="lower right", fontsize=8)
    ax[2].set_title(f"verdict: {result['verdict']}   valid {result['valid_columns_frac']*100:.0f} %   contrast {result['mean_contrast']:.0f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("on", help="laser-on frame")
    ap.add_argument("--off", help="laser-off frame (recommended)")
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(__file__), "calib.json"))
    ap.add_argument("--plot", help="write a diagnostic PNG")
    ap.add_argument("--json", help="write the result JSON")
    args = ap.parse_args()

    calib = {}
    if os.path.exists(args.calib):
        with open(args.calib) as fh:
            calib = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    result, inter = analyse(args.on, args.off, calib)

    print(json.dumps(result, indent=2))
    if args.plot:
        plot(result, inter, args.plot)
        print(f"plot -> {args.plot}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
