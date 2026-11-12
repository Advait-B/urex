"""dp_dense.py - globally optimal laser-line tracking: Viterbi over EVERY row of every column.

The baseline finds ridge points and then greedily keeps "the strongest chain and nearby pieces".
Here the choice is global instead: one row v(c) per column c, picked so that the whole path
collects as much laser-like ridge as possible while moving smoothly.

  1. ridge map   top-hat, Gaussian blur, Hessian (Steger): strength = how much a pixel looks like the
                 centre of a thin bright line; reward = strength / (typical laser strength), capped at 1
                 so a glint that is brighter than the laser earns nothing extra.
  2. Viterbi     cost(c, v) = -reward(c, v) + min over d of [ cost(c-1, v-d) + pen(d) ]
                 pen(d) = A*|d| for |d| <= 3 (gentle slope: cheap), else J + B*|d| (a groove edge: a fixed
                 price plus a little per px, so a far-away glint costs more than a groove floor).
                 Rows without ridge have reward 0: that IS the "no line here" state, and unlike a single
                 gap state it remembers where the line was.
  3. points      keep the Steger sub-pixel points within NEAR px of the path, and between the two ends
                 of every jump (the groove wall, which one-row-per-column would lose).

Uses only the laser-on frame.
"""
import cv2
import numpy as np

NAME = "dp_dense"
USES_OFF = False
P = {
    "sigma": 1.5, "tophat": 15,     # same ridge scale and background removal as the baseline
    "min_contrast": 12.0,           # grey levels above background (baseline: 35; the path does the rejecting)
    "A": 0.15, "smooth_px": 3,      # cost per px of slope, for moves up to smooth_px rows per column
    "J": 1.0, "B": 0.06,            # cost of a jump (groove edge) = J + B*|d|
    "keep_rel": 0.2,                # a path column counts as "line" if reward >= this
    "near": 1.5,                    # px: sub-pixel points this close to the path are kept
    "wide_px": 5.0,                 # a line wider than this (saturated / defocused) is located by its two half-max edges
    "min_run": 4,                   # columns: shorter isolated bits of line are dropped
    "run_test": False, "short_run": 30,  # (dp_dense_rt) isolated runs shorter than this must run parallel to the line:
    "max_tilt": 0.35,                    # px/column against the main line
}


def ridge_maps(gray, sigma, tophat_px):
    """-> strength, contrast, du, dv  (du, dv = sub-pixel offset of the line centre from each pixel)"""
    g = gray.astype(np.float32)
    th = g - cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_px, tophat_px)))
    s = cv2.GaussianBlur(th, (0, 0), sigma)
    dy, dx = np.gradient(s)
    dyy, dyx = np.gradient(dy)
    dxx = np.gradient(dx, axis=1)
    disc = np.sqrt(((dxx - dyy) / 2) ** 2 + dyx ** 2)
    lmin = (dxx + dyy) / 2 - disc                        # most negative curvature = across a bright line
    nx, ny = dyx, lmin - dxx                             # its direction (the line's normal)
    n = np.hypot(nx, ny) + 1e-12
    nx, ny = nx / n, ny / n
    t = -(dx * nx + dy * ny) / np.minimum(lmin, -1e-6)   # step along the normal to the ridge top
    return np.maximum(-lmin, 0) * sigma ** 2, th, t * nx, t * ny


def reward_map(strength, contrast, p):
    col_max = strength.max(axis=0)
    scale = max(float(np.median(col_max)), 1e-3)         # the laser crosses most columns: typical laser strength
    r = np.minimum(strength / scale, 1.0)
    r[contrast < p["min_contrast"]] = 0
    return r, scale


def viterbi(reward, p):
    """Best row per column. reward: H x W in 0..1. Every row of every column is a state."""
    H, W = reward.shape
    s, A, J, B = p["smooth_px"], p["A"], p["J"], p["B"]
    rows = np.arange(H, dtype=np.float32)
    Br = B * rows
    R = np.ascontiguousarray(reward.T)
    C = np.full((W, H + 2 * s), 1e9, np.float32)      # C[c, s + v] = cost of the best path ending at (c, v)
    C[0, s:s + H] = -R[0]
    acc = np.minimum.accumulate
    for c in range(1, W):
        full = C[c - 1]
        prev = full[s:s + H]
        near = prev.copy()
        for k in range(1, s + 1):                     # smooth move of k rows up or down: A per px
            np.minimum(near, full[s - k:s - k + H] + A * k, out=near)
            np.minimum(near, full[s + k:s + k + H] + A * k, out=near)
        # jump from any row v': prev[v'] + J + B*|v - v'|, as two running minima (from above, from below)
        far = J + np.minimum(acc(prev - Br) + Br, acc((prev + Br)[::-1])[::-1] - Br)
        C[c, s:s + H] = np.minimum(near, far) - R[c]
    C = C[:, s:s + H]
    path = np.empty(W, np.int32)
    v = path[-1] = C[-1].argmin()
    for c in range(W - 1, 0, -1):                     # walk back: which row of column c-1 did we come from?
        d = np.abs(rows - v)
        v = path[c - 1] = np.argmin(C[c - 1] + np.where(d <= s, A * d, J + B * d))
    return path


def filter_runs(on, path, p):
    """Split the path into runs (cut at gaps and at jumps > 4 px) and drop the runs that cannot be laser:
    very short ones (speckle), and short ISOLATED ones that are tilted against the rest of the line
    (a glint picked up where the real line is hidden; a real groove floor runs parallel to the surface)."""
    W = len(on)
    c = np.arange(W)
    brk = np.r_[True, (np.abs(np.diff(path)) > 4) | ~on[1:] | ~on[:-1]]
    run = np.cumsum(brk) - 1
    n_run = run[-1] + 1
    size = np.bincount(run, weights=on, minlength=n_run)
    keep = on & (size[run] >= p["min_run"])
    if keep.sum() < 20 or not p["run_test"]:
        return keep
    w = keep.astype(float)                                    # robust straight line through the kept path
    for _ in range(3):
        k = np.polyfit(c, path, 1, w=w)
        res = path - np.polyval(k, c)
        w = keep * (np.abs(res) < 3)
        if w.sum() < 10:
            return keep
    for r in np.nonzero((size >= p["min_run"]) & (size < p["short_run"]))[0]:
        i = np.nonzero((run == r) & on)[0]
        halves = [i] if len(i) < 8 else [i[:len(i) // 2], i[len(i) // 2:]]     # a lit wall + floor: one half is flat
        tilt = min(abs(np.polyfit(h, path[h], 1)[0] - k[0]) for h in halves)
        if tilt > p["max_tilt"]:
            keep[i] = False
    return keep


def track(gray, p=P):
    strength, contrast, du, dv = ridge_maps(gray, p["sigma"], p["tophat"])
    reward, scale = reward_map(strength, contrast, p)
    path = viterbi(reward, p)
    cols = np.arange(gray.shape[1])
    on = reward[path, cols] >= p["keep_rel"]
    on = filter_runs(on, path, p)
    return {"path": path, "on": on, "reward": reward, "du": du, "dv": dv, "scale": scale, "strength": strength, "contrast": contrast}


def band_centre(th, r, cols, half=12):
    """Centre and width of the bright band through pixel (r, cols) in each column: the midpoint of its two
    half-maximum edges. A saturated or defocused line is a flat band whose EDGES are the ridges, so the
    ridge points sit ~3 px off the true centre; the midpoint of the edges does not care about the flat top."""
    k = np.arange(-half, half + 1)
    T = th[np.clip(r[:, None] + k, 0, th.shape[0] - 1), cols[:, None]]
    lvl = 0.5 * T[:, half - 2:half + 3].max(axis=1, keepdims=True)

    def edge(S):                                   # S[:, 0] = the path pixel, going outwards
        n = np.minimum.accumulate(S >= lvl, axis=1).sum(axis=1)          # pixels above half-max, contiguous
        n = np.clip(n, 1, S.shape[1] - 1)
        i = np.arange(len(S))
        a, b = S[i, n - 1], S[i, n]                # last pixel above, first below
        return n - 1 + np.clip((a - lvl[:, 0]) / np.maximum(a - b, 1e-6), 0, 1)
    down, up = edge(T[:, half:]), edge(T[:, half::-1])
    return r + (down - up) / 2, down + up


def points_near_path(t, p=P):
    """One sub-pixel point per path column, plus the wall between the two ends of each jump."""
    path, on, reward = t["path"], t["on"], t["reward"]
    H, W = reward.shape
    cols = np.nonzero(on)[0]
    # (a) the path itself: the strongest of the 3 rows around it, moved to the ridge top
    rows = np.clip(path[cols][:, None] + np.arange(-1, 2), 0, H - 1)
    r = rows[np.arange(len(cols)), t["strength"][rows, cols[:, None]].argmax(axis=1)]
    u = cols + np.clip(t["du"][r, cols], -0.5, 0.5)
    v = r + np.clip(t["dv"][r, cols], -1.0, 1.0)
    mid, width = band_centre(t["contrast"], r, cols)
    wide = width > p["wide_px"]
    u, v = np.where(wide, cols, u), np.where(wide, mid, v)
    # (b) walls: where the path jumps between neighbouring columns, the Steger points in between
    prev, nxt = np.r_[path[0], path[:-1]], np.r_[path[1:], path[-1]]
    lo, hi = np.minimum(prev, nxt) + 2, np.maximum(prev, nxt) - 2
    both = np.r_[False, on[:-1]] & np.r_[on[1:], False] & on         # the line continues on both sides
    vv = np.arange(H)[:, None]
    m = (vv >= lo) & (vv <= hi) & (np.abs(vv - path) >= 2) & both & (reward >= p["keep_rel"])
    m &= (np.abs(t["du"]) <= 0.5) & (np.abs(t["dv"]) <= 0.5)        # the pixel that contains the ridge top
    wv, wu = np.nonzero(m)
    return {"u": np.r_[u, wu + t["du"][m]], "v": np.r_[v, wv + t["dv"][m]],
            "strength": np.r_[t["strength"][r, cols], t["strength"][m]], "n_path": len(cols)}   # path points come first


def extract_line(gray, off=None, p=P):
    return points_near_path(track(gray, p), p)
