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
  5. shoulder line      the tread surface is the straight line most columns
                        agree on. A RANSAC fit finds it even though the groove
                        columns are far away from it.
  6. depth              residual (px) * mm_per_px_depth from calib.json
  7. grooves            runs of columns deeper than 1 mm and wider than 4 mm.
                        Each groove's depth = median of its central half.
                        PASS if every groove >= 1.6 mm.

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
    "min_contrast": 40, "line_sigma_px": 1.5, "groove_min_depth_mm": 1.0,
    "groove_min_width_mm": 4.0, "legal_min_depth_mm": 1.6,
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


# ------------------------------------------------------------- 6. depth
def depth_from_line(x, y, a, b, calib):
    resid = y - (a * x + b)                               # px, + = line moved "down"
    return calib["depth_sign"] * resid * calib["mm_per_px_depth"]


# ------------------------------------------------------------ 7. grooves
def find_grooves(depth_mm, calib):
    deep = depth_mm > calib["groove_min_depth_mm"]
    deep = np.where(np.isnan(depth_mm), False, deep)
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
    a, b, inlier_frac = ransac_line(x, y)
    depth = depth_from_line(x, y, a, b, calib)
    grooves = find_grooves(depth, calib)
    warnings = []
    # Grooves are always LOWER than the tread surface. If the big deviations from
    # the shoulder line all come out negative, depth_sign in calib.json is wrong.
    big = depth[~np.isnan(depth)]
    big = big[np.abs(big) > calib["groove_min_depth_mm"]]
    if len(big) > 10 and np.median(big) < 0:
        warnings.append("most deviations are negative: flip depth_sign in calib.json (or rotate180)")

    valid_frac = float(np.mean(~np.isnan(y)))
    mean_contrast = float(np.nanmean(contrast[~np.isnan(y)])) if valid_frac else 0.0
    quality_ok = valid_frac >= 0.9 and mean_contrast > 60       # the CHECK state in the firmware sketch
    worst = min((g["depth_mm"] for g in grooves), default=None)
    result = {
        "on": on_path, "off": off_path,
        "valid_columns_frac": round(valid_frac, 3),
        "mean_contrast": round(mean_contrast, 1),
        "shoulder_line": {"slope_px_per_col": float(a), "intercept_px": float(b), "inlier_frac": round(float(inlier_frac), 3)},
        "grooves": grooves,
        "min_groove_depth_mm": worst,
        "quality_ok": bool(quality_ok),
        "warnings": warnings,
        "verdict": ("NO_GROOVE" if not grooves else
                    "FAIL" if worst < calib["legal_min_depth_mm"] else "PASS") if quality_ok else "RETAKE",
    }
    return result, dict(diff=diff, y=y, x=x, a=a, b=b, depth=depth, on=on)


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
    ax[1].plot(inter["x"], inter["a"] * inter["x"] + inter["b"], "--", color="white", lw=1, label="shoulder (RANSAC)")
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
