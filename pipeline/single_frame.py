"""single_frame.py - tread depth from ONE laser-on photo (no laser-off frame).

The hand-held photos from 28 and 30 Oct are single frames: there is no
laser-off partner to subtract. They also differ from what laser_profile.py
expects: the laser line runs TOP-TO-BOTTOM in the image, and inside a groove
the line does not make a flat step. It traces the groove's cross-section:
down the sloped wall, round the bottom, back up the far wall (partly hidden).
So this module finds the line in 2-D, and reads depth at the DEEPEST point of
each excursion, not as the height of a flat step.

    python single_frame.py photo.jpg --plot out.png

Steps (each a function below):
  1. top-hat        subtract the local background (anything wider than ~15 px),
                    which replaces the missing laser-off frame for thin lines
  2. ridge points   Steger's method: the Hessian (2nd derivatives after a
                    Gaussian blur) says "bright thin line, running this way".
                    Its centre is found to a fraction of a pixel along the
                    line's normal. Works for a line in any direction, so the
                    sideways run inside a groove is kept.
  3. keep the laser chain
                    link points into connected pieces; the laser is the
                    longest/strongest chain. Keep only pieces near it (drops
                    bright object edges in the background).
  4. surface line   RANSAC fit of a gently curved line x = a*y^2 + b*y + c
                    (the tyre crown and the lens bend it a few px): the tread
                    surface is the part most points agree on.
  5. excursions     groups of points more than 3 px off that line = grooves
                    (or lug steps). Depth in px = the excursion's apex: the
                    mean of the points within 0.7 px of the 3rd-furthest one
                    (so a stray point or two can't set the depth).
  5b. lug steps     for lugged tyres (28 Oct rover) the line is a staircase,
                    so there is no single surface. Instead measure each STEP:
                    fit short lines either side of every jump and take their
                    difference at the jump.
  6. triangulate    px -> mm with the DESIGN geometry (uncalibrated, see
                    GEOMETRY below): each point's column gives its distance
                    from the camera; depth = distance(apex) - distance(surface
                    directly beside it).
"""
import argparse
import json

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Design geometry (13 Sep optics sheet). The laser sits BASELINE_MM to the side of
# the lens and is aimed so its sheet crosses the image centre at L0_MM. Then for a
# point at image column x:   1/z = 1/L0 + (x - cx) / K,   K = f * baseline / pitch.
# NOT CALIBRATED: these are what the rig was designed to be. The shim test replaces it.
GEOMETRY = {
    "f_mm": 2.8,            # OpenMV H7 Plus stock lens
    "baseline_mm": 63.0,    # laser offset from the lens (tan 35 deg * 90 mm)
    "L0_mm": 90.0,          # distance where the line crosses the image centre
    "pitch_mm_at_640": 0.0056,   # OV5640 effective pixel at 640 px wide (4x binning)
    "far_is_left": True,    # seen in the 30 Oct photos: groove bottoms shift LEFT
}

DEFAULTS = {
    "sigma": 1.5,           # Gaussian scale for the Hessian, px (line half-width)
    "tophat": 15,           # background removal size, px (must be > line width)
    "min_contrast": 35,     # grey levels above local background to count as laser
    "rel_strength": 0.3,    # ridge strength vs the frame's strongest ridge
    "chain_gap": 3,         # px: points this close are linked into one piece
    "keep_within_px": 30,   # pieces further than this from the main chain are dropped
    "min_rel_contrast": 0.5,  # ...and pieces much dimmer than the main chain
    "min_piece": 6,         # points; smaller pieces are noise
    "ransac_px": 1.5,       # surface inlier tolerance
    "excursion_px": 3.0,    # a point this far off the surface belongs to a groove/step
    "min_excursion_pts": 6,
    "min_groove_pts": 15,   # an excursion this big (and going 'far') is reported as THE groove
    "max_curvature": 0.002, # px per px^2: anything more curved isn't a tread surface
    "step_min_px": 3.0, "step_max_px": 60.0, "step_fit_rows": 12, "step_min_rows": 6,
    "staircase_inliers": 0.45,  # surface inlier fraction below this -> measure lug steps instead
    "apex_band_px": 0.7,
}


# ------------------------------------------------------------- 1 + 2 ridges
def tophat(gray, size):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    g = gray.astype(np.float32)
    return g - cv2.morphologyEx(g, cv2.MORPH_OPEN, k)


def steger_points(gray, sigma=1.5, tophat_px=15, min_contrast=35, rel_strength=0.3):
    """Sub-pixel centre points of thin bright lines. Returns dict of arrays
    x, y (sub-pixel), strength (ridge), contrast (top-hat value), nx, ny (normal)."""
    th = tophat(gray, tophat_px)
    d = {o: ndi.gaussian_filter(th, sigma, order=o) for o in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 1)]}
    dx, dy, dxx, dyy, dxy = d[(0, 1)], d[(1, 0)], d[(0, 2)], d[(2, 0)], d[(1, 1)]
    # Hessian eigenvalues; the most negative one points ACROSS a bright line
    half_tr = (dxx + dyy) / 2
    disc = np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
    lmin = half_tr - disc
    nx, ny = dxy, lmin - dxx
    flat = (np.abs(nx) + np.abs(ny)) < 1e-9
    nx = np.where(flat, 1.0, nx)
    ny = np.where(flat, 0.0, ny)
    n = np.hypot(nx, ny)
    nx, ny = nx / n, ny / n
    hnn = dxx * nx * nx + 2 * dxy * nx * ny + dyy * ny * ny
    t = -(dx * nx + dy * ny) / np.where(hnn == 0, -1e-9, hnn)      # sub-pixel offset along the normal
    strength = -lmin * sigma ** 2
    top = np.percentile(strength, 99.7)
    ok = ((lmin < 0) & (np.abs(t * nx) <= 0.5) & (np.abs(t * ny) <= 0.5)
          & (th > min_contrast) & (strength > rel_strength * max(top, 1e-6)))
    yy, xx = np.nonzero(ok)
    return {"x": xx + t[ok] * nx[ok], "y": yy + t[ok] * ny[ok], "strength": strength[ok],
            "contrast": th[ok], "nx": nx[ok], "ny": ny[ok], "ix": xx, "iy": yy}


# ---------------------------------------------------- 3 keep the laser chain
def select_chain(pts, shape, gap=3, keep_within=30, min_piece=6, min_rel_contrast=0.5, touch_px=16):
    """Label connected pieces, pick the strongest long one as the laser, keep the
    pieces that lie near the straight line through it. Returns a boolean mask over pts."""
    if len(pts["x"]) == 0:
        return np.zeros(0, bool), 0
    m = np.zeros(shape, np.uint8)
    m[pts["iy"], pts["ix"]] = 1
    m = cv2.dilate(m, np.ones((gap, gap), np.uint8))
    n_lab, lab = cv2.connectedComponents(m, connectivity=8)
    piece = lab[pts["iy"], pts["ix"]]
    ids, counts = np.unique(piece, return_counts=True)
    score = {i: c * float(np.mean(pts["strength"][piece == i])) for i, c in zip(ids, counts)}
    best = max(score, key=score.get)
    sel = piece == best
    # straight-line guide through the best piece (x as a function of y: the line runs top-bottom)
    ys, xs = pts["y"][sel], pts["x"][sel]
    if np.ptp(ys) > 5:
        a, b = np.polyfit(ys, xs, 1)
    else:
        a, b = 0.0, float(np.median(xs))
    best_con = float(np.mean(pts["contrast"][sel]))
    cand = {}
    for i, c in zip(ids, counts):
        on = piece == i
        if i != best and (c < min_piece or np.mean(pts["contrast"][on]) < min_rel_contrast * best_con):
            continue                                   # too small or too dim to be laser
        cand[i] = on
    # grow from the best piece: add a piece if it lies along the guide line, OR if one
    # of its ends comes within touch_px of what is already kept (a groove's hook is
    # often a separate piece because the far wall hides part of the line)
    keep = cand[best].copy()
    added = True
    while added:
        added = False
        tree = cKDTree(np.c_[pts["x"][keep], pts["y"][keep]])
        for i, on in list(cand.items()):
            if i == best or (keep & on).any():
                continue
            dist_line = np.median(np.abs(pts["x"][on] - (a * pts["y"][on] + b)))
            d_near, _ = tree.query(np.c_[pts["x"][on], pts["y"][on]])
            if dist_line < keep_within or d_near.min() <= touch_px:
                keep |= on
                added = True
    return keep, int(sum(1 for i, c in zip(ids, counts) if c >= min_piece))


# -------------------------------------------------------- 4 surface line
def surface_x(coef, y):
    return np.polyval(coef, y)


def ransac_xy(y, x, thresh=1.5, iters=500, max_curv=0.002, seed=0):
    """x = c2*y^2 + c1*y + c0 fitted to the biggest group of agreeing points.
    Returns (coef, inlier mask). The laser line is roughly vertical (|slope| < 1)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    best, best_n = np.array([0.0, 0.0, float(np.median(x))]), -1
    if n < 3:
        return best, np.ones(n, bool)
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        yi = y[i]
        if np.ptp(yi) < 20 or min(abs(yi[0] - yi[1]), abs(yi[1] - yi[2]), abs(yi[0] - yi[2])) < 5:
            continue
        c = np.polyfit(yi, x[i], 2)
        ym = (y.min() + y.max()) / 2
        if abs(c[0]) > max_curv or abs(2 * c[0] * ym + c[1]) > 1.0:
            continue
        inl = np.abs(x - np.polyval(c, y)) < thresh
        if inl.sum() > best_n:
            best, best_n = c, inl.sum()
    inl = np.abs(x - np.polyval(best, y)) < thresh
    if inl.sum() >= 6 and np.ptp(y[inl]) > 20:
        c = np.polyfit(y[inl], x[inl], 2)
        if abs(c[0]) <= max_curv:
            best = c
        inl = np.abs(x - np.polyval(best, y)) < thresh
    return best, inl


# ------------------------------------------------------------ 5 excursions
def find_excursions(x, y, coef, shape, cfg):
    r = x - surface_x(coef, y)
    off = np.abs(r) > cfg["excursion_px"]
    out = []
    if off.sum() < cfg["min_excursion_pts"]:
        return out, r
    m = np.zeros(shape, np.uint8)
    iy, ix = np.clip(np.round(y[off]).astype(int), 0, shape[0] - 1), np.clip(np.round(x[off]).astype(int), 0, shape[1] - 1)
    m[iy, ix] = 1
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    _, lab = cv2.connectedComponents(m, connectivity=8)
    grp = lab[iy, ix]
    idx = np.nonzero(off)[0]
    for g in np.unique(grp):
        sel = idx[grp == g]
        if len(sel) < cfg["min_excursion_pts"]:
            continue
        rr = r[sel]
        far = np.argmax(np.abs(rr))
        sign = np.sign(rr[far])
        # anchor on the 3rd-furthest point, not the furthest: one or two stray points
        # (speckle, a glint) must not set the depth. Then average everything within
        # apex_band_px of that anchor (the bottom of the groove).
        same = np.abs(rr[np.sign(rr) == sign])
        ref = np.sort(same)[-3] if len(same) >= 3 else same.max()
        band = sel[(np.abs(rr) >= ref - cfg["apex_band_px"]) & (np.abs(rr) <= ref + cfg["apex_band_px"])
                   & (np.sign(rr) == sign)]
        out.append({"n": int(len(sel)), "apex_px": float(np.mean(r[band])), "depth_px": float(abs(np.mean(r[band]))),
                    "apex_x": float(np.mean(x[band])), "apex_y": float(np.mean(y[band])),
                    "rows": [float(y[sel].min()), float(y[sel].max())],
                    "direction": "far" if (sign < 0) == GEOMETRY["far_is_left"] else "near",
                    "idx": sel})
    out.sort(key=lambda e: -e["depth_px"])
    return out, r


# ------------------------------------------------------------ 5b lug steps
def find_steps(x, y, cfg):
    """Staircase measure: one x per row (median of points in that row), split where
    the line jumps sideways, fit a straight line to the rows just above and just
    below each jump, step = their difference at the jump row."""
    if len(x) < 2 * cfg["step_min_rows"]:
        return []
    rows = np.round(y).astype(int)
    ur = np.unique(rows)
    xr = np.array([np.median(x[rows == r]) for r in ur])
    spread = np.array([np.ptp(x[rows == r]) for r in ur])
    ok = spread < 4                     # rows where the line crosses once (not a sideways run)
    ur, xr = ur[ok], xr[ok]
    if len(ur) < 2 * cfg["step_min_rows"]:
        return []
    cut = np.nonzero((np.abs(np.diff(xr)) > cfg["step_min_px"]) | (np.diff(ur) > 6))[0]
    segs = np.split(np.arange(len(ur)), cut + 1)
    steps = []
    for s0, s1 in zip(segs[:-1], segs[1:]):
        if len(s0) < cfg["step_min_rows"] or len(s1) < cfg["step_min_rows"] or ur[s1[0]] - ur[s0[-1]] > 16:
            continue
        a = s0[-cfg["step_fit_rows"]:]
        b = s1[:cfg["step_fit_rows"]]
        ya, yb = ur[a], ur[b]
        fa = np.polyfit(ya, xr[a], 1) if np.ptp(ya) > 2 else np.array([0.0, np.median(xr[a])])
        fb = np.polyfit(yb, xr[b], 1) if np.ptp(yb) > 2 else np.array([0.0, np.median(xr[b])])
        if abs(fa[0]) > 1.5 or abs(fb[0]) > 1.5:
            continue
        yj = (ur[s0[-1]] + ur[s1[0]]) / 2
        d = np.polyval(fb, yj) - np.polyval(fa, yj)
        if cfg["step_min_px"] <= abs(d) <= cfg["step_max_px"]:
            steps.append({"row": float(yj), "step_px": float(d), "x_above": float(np.polyval(fa, yj)),
                          "x_below": float(np.polyval(fb, yj))})
    return steps


# ------------------------------------------------------------- 6 triangulate
def distance_mm(x, width, geom=GEOMETRY):
    """Distance from the camera (mm) of a laser point at image column x, by the design geometry."""
    pitch = geom["pitch_mm_at_640"] * 640.0 / width
    K = geom["f_mm"] * geom["baseline_mm"] / pitch
    u = (np.asarray(x, float) - (width - 1) / 2) * (1 if geom["far_is_left"] else -1)
    inv = 1.0 / geom["L0_mm"] + u / K
    return np.where(inv > 0, 1.0 / np.where(inv > 0, inv, 1), np.nan)


def depth_mm(apex_x, surface_x, width, geom=GEOMETRY):
    """Depth of an excursion = distance(apex) - distance(surface beside it)."""
    return float(distance_mm(apex_x, width, geom) - distance_mm(surface_x, width, geom))


def fixed_scale_mm(depth_px, width, z_mm=90.0, geom=GEOMETRY):
    """Alternative: assume the surface was exactly z_mm away (ignores where the line sits)."""
    pitch = geom["pitch_mm_at_640"] * 640.0 / width
    K = geom["f_mm"] * geom["baseline_mm"] / pitch
    inv = 1.0 / z_mm - depth_px / K
    return float(1.0 / inv - z_mm) if inv > 0 else float("nan")


# --------------------------------------------------- naive "raw" profile
def raw_profile(gray):
    """What you get WITHOUT the processing: brightest pixel in each row. Used to
    show raw vs cleaned. Picks up glare, background edges and the wrong branch."""
    g = gray.astype(np.float32)
    return g.argmax(axis=1).astype(float), np.arange(g.shape[0], dtype=float), g.max(axis=1)


# ----------------------------------------------------------------- driver
def analyse(gray, cfg=None, geom=GEOMETRY):
    cfg = {**DEFAULTS, **(cfg or {})}
    h, w = gray.shape
    pts = steger_points(gray, cfg["sigma"], cfg["tophat"], cfg["min_contrast"], cfg["rel_strength"])
    keep, n_pieces = select_chain(pts, gray.shape, cfg["chain_gap"], cfg["keep_within_px"], cfg["min_piece"],
                                  cfg["min_rel_contrast"])
    x, y = pts["x"][keep], pts["y"][keep]
    res = {"width": w, "height": h, "n_points_all": int(len(pts["x"])), "n_points": int(len(x)),
           "n_pieces": n_pieces, "excursions": [], "steps": [], "surface": None, "standoff_mm": None,
           "mode": "none", "depth_px": None, "row_coverage": 0.0, "mean_contrast": 0.0, "mean_strength": 0.0}
    res["_pts"], res["_keep"] = pts, keep
    if len(x) < 10:
        res["_x"], res["_y"], res["_r"] = x, y, np.zeros(len(x))
        return res
    coef, inl = ransac_xy(y, x, cfg["ransac_px"], max_curv=cfg["max_curvature"])
    exc, r = find_excursions(x, y, coef, gray.shape, cfg)
    yc = (h - 1) / 2
    xc = float(surface_x(coef, yc))
    res["surface"] = {"coef": [float(c) for c in coef], "inlier_frac": float(inl.mean()), "x_at_centre": xc}
    res["standoff_mm"] = float(distance_mm(xc, w, geom))
    for e in exc:
        sx = float(surface_x(coef, e["apex_y"]))
        e["depth_mm"] = depth_mm(e["apex_x"], sx, w, geom)
        e["depth_mm_fixed90"] = fixed_scale_mm(e["depth_px"] * (1 if e["direction"] == "far" else -1), w, 90.0, geom)
        e["standoff_mm"] = float(distance_mm(sx, w, geom))
    res["excursions"] = exc
    res["steps"] = find_steps(x, y, cfg)
    res["_x"], res["_y"], res["_r"], res["_inl"] = x, y, r, inl
    # the headline measurement for this frame
    grooves = [e for e in exc if e["direction"] == "far" and e["n"] >= cfg["min_groove_pts"]]
    if inl.mean() < cfg["staircase_inliers"] and res["steps"]:
        res["mode"] = "steps"
        res["depth_px"] = float(np.median([abs(s["step_px"]) for s in res["steps"]]))
    elif grooves:
        res["mode"] = "groove"
        res["depth_px"] = grooves[0]["depth_px"]
        res["groove"] = grooves[0]
    else:
        res["mode"] = "none"
        res["depth_px"] = None
    # quality numbers used by the frame classifier
    rows = np.unique(np.round(y).astype(int))
    res["row_coverage"] = float(len(rows) / h)
    res["mean_contrast"] = float(np.mean(pts["contrast"][keep]))
    res["mean_strength"] = float(np.mean(pts["strength"][keep]))
    return res


def public(res):
    """result without the big internal arrays (for JSON)"""
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    out["excursions"] = [{k: v for k, v in e.items() if k != "idx"} for e in res["excursions"]]
    if "groove" in out:
        out["groove"] = {k: v for k, v in res["groove"].items() if k != "idx"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--plot", help="write a raw-vs-cleaned figure")
    args = ap.parse_args()
    g = cv2.imread(args.photo, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f"cannot read {args.photo}")
    res = analyse(g)
    print(json.dumps(public(res), indent=2, default=float))
    if args.plot:
        import report_figs
        report_figs.frame_figure(g, res, args.plot, title=args.photo)


if __name__ == "__main__":
    main()
