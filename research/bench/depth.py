"""depth.py - the DEFAULT profile -> groove depth stage, shared by all line extractors.

Input: sub-pixel laser points (u, v) in the canonical orientation (line left-right,
deeper = down). With a Calib the work is done in mm, otherwise in px.

  1. q = depth coordinate of every point: -height_mm (calibrated) or v (px)
  2. surface: RANSAC fit of q = a*u^2 + b*u + c (the tread crown is gently curved)
  3. grooves: runs of points more than `min_depth` below the surface, at least
     `min_width_px` wide, split where consecutive points are > `gap_px` apart in u
  4. groove depth: anchor on the 3rd-deepest point and average everything within
     `band` of it (a floor estimate that 1-2 stray points cannot move)

An algorithm may replace this whole stage by defining its own measure().
"""
import numpy as np

MM = {"ransac": 0.30, "min_depth": 0.5, "band": 0.25}
PX = {"ransac": 1.2, "min_depth": 2.5, "band": 0.7}


def ransac_quadratic(u, q, thresh, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(u)
    best, best_n = np.array([0.0, 0.0, float(np.median(q)) if n else 0.0]), -1
    if n < 6:
        return best, np.ones(n, bool)
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        if np.ptp(u[i]) < 30:
            continue
        c = np.polyfit(u[i], q[i], 2)
        if abs(c[0]) > 0.002:                      # px per px^2 (or mm): flatter than any groove
            continue
        k = int((np.abs(q - np.polyval(c, u)) < thresh).sum())
        if k > best_n:
            best, best_n = c, k
    inl = np.abs(q - np.polyval(best, u)) < thresh
    if inl.sum() >= 8 and np.ptp(u[inl]) > 30:
        c = np.polyfit(u[inl], q[inl], 2)
        if abs(c[0]) <= 0.002:
            best = c
    return best, np.abs(q - np.polyval(best, u)) < thresh


def measure_from_line(u, v, calib=None, min_width_px=4.0, gap_px=5.0, min_points=5):
    """Returns {"grooves": [{"u0","u1","depth","unit","n"}], "surface": coef, "inlier_frac": float}."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    P = MM if calib is not None else PX
    unit = "mm" if calib is not None else "px"
    if len(u) < 12:
        return {"grooves": [], "surface": None, "inlier_frac": 0.0, "unit": unit}
    q = -calib.height(u, v) if calib is not None else v
    coef, inl = ransac_quadratic(u, q, P["ransac"])
    r = q - np.polyval(coef, u)                        # > 0 = below the surface
    deep = r > P["min_depth"]
    grooves = []
    if deep.sum() >= min_points:
        idx = np.nonzero(deep)[0]
        idx = idx[np.argsort(u[idx])]
        for run in np.split(idx, np.nonzero(np.diff(u[idx]) > gap_px)[0] + 1):
            if len(run) < min_points or np.ptp(u[run]) < min_width_px:
                continue
            rr = np.sort(r[run])
            ref = rr[-3] if len(rr) >= 3 else rr[-1]
            band = rr[(rr >= ref - P["band"]) & (rr <= ref + P["band"])]
            grooves.append({"u0": float(u[run].min()), "u1": float(u[run].max()), "depth": float(band.mean()),
                            "unit": unit, "n": int(len(run))})
    return {"grooves": grooves, "surface": [float(c) for c in coef], "inlier_frac": float(inl.mean()), "unit": unit}
