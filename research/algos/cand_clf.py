"""cand_clf.py - learned replacement for the baseline's hand-written chain selection.

1. candidates: Steger ridge points (same maths as pipeline/single_frame.py) with a PERMISSIVE
   threshold, so dim lines survive and so does a lot of clutter (glints, paint edges, texture).
2. features: 30 numbers per candidate point (own strength/contrast/width/orientation, its
   connected piece, what else is in its column, how much ridge there is along its row).
3. classify: a small HistGradientBoosting model (trained on the `train` set only, labels =
   within 1 px of the true line) gives P(laser) per point; keep points with P >= THRESH.
Depth: the default stage in bench/depth.py. Uses only the laser-on frame.

Variants (set by the wrapper modules cand_lr.py / cand_thresh.py):
  MODEL = "hgb"     models/cand_hgb.pkl    (default)
  MODEL = "lr"      models/cand_lr.json    logistic regression: 31 weights, runs anywhere
  MODEL = "thresh"  one threshold on the single best feature (the simplest alternative)
"""
import json
import os
import pickle

import cv2
import numpy as np

NAME = "cand_hgb"
USES_OFF = False
MODEL = "hgb"
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")
SIGMA, TOPHAT = 1.5, 15
MIN_CONTRAST, REL_STRENGTH = 6.0, 0.08          # baseline: 35 and 0.3
_cache = {}

FEATURES = [
    "strength", "contrast", "gray", "rel_strength", "rel_contrast", "strength_s3_ratio", "tophat7_ratio",
    "aniso", "normal_v", "loc_mean", "loc_std", "contrast_over_std",
    "row_support10", "row_support30", "row_support30_rel", "band_dark",
    "col_n", "col_rank", "col_dv_best", "col_ratio_best",
    "piece_n", "piece_du", "piece_dv", "piece_resid", "piece_strength_rel", "piece_score_rel", "piece_slope",
    "dv_mode", "abs_dv_mode", "n_cand",
]


def _ridge(img, sigma):
    """Hessian of the blurred image -> (lmin, lmax, dx, dy, dxx, dyy, dxy)."""
    k = int(4 * sigma + 0.5) * 2 + 1
    b = cv2.GaussianBlur(img, (k, k), sigma)
    dx, dy = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=1) / 2, cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=1) / 2   # [-1 0 1] / 2
    dxx, dyy = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=1), cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=1)
    dxy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3) / 4          # ksize 3 = [-1 0 1] x [-1 0 1]: 4x the central difference
    half, disc = (dxx + dyy) / 2, np.sqrt(((dxx - dyy) / 2) ** 2 + dxy ** 2)
    return half - disc, half + disc, dx, dy, dxx, dyy, dxy


def _tophat(g, size):
    return g - cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def candidates(gray):
    """Permissive Steger points + the per-pixel maps the features need."""
    g = gray.astype(np.float32)
    th = _tophat(g, TOPHAT)
    lmin, lmax, dx, dy, dxx, dyy, dxy = _ridge(th, SIGMA)
    nx, ny = dxy, lmin - dxx                                   # normal = eigenvector of lmin
    flat = (np.abs(nx) + np.abs(ny)) < 1e-9
    nx, ny = np.where(flat, 0.0, nx), np.where(flat, 1.0, ny)
    n = np.hypot(nx, ny)
    nx, ny = nx / n, ny / n
    hnn = dxx * nx * nx + 2 * dxy * nx * ny + dyy * ny * ny
    t = -(dx * nx + dy * ny) / np.where(hnn == 0, -1e-9, hnn)
    S = -lmin * SIGMA ** 2
    top = max(float(np.percentile(S, 99.7)), 1e-6)
    ok = (lmin < 0) & (np.abs(t * nx) <= 0.5) & (np.abs(t * ny) <= 0.5) & (th > MIN_CONTRAST) & (S > REL_STRENGTH * top)
    iv, iu = np.nonzero(ok)
    return {"u": iu + t[ok] * nx[ok], "v": iv + t[ok] * ny[ok], "iu": iu, "iv": iv, "ok": ok,
            "g": g, "th": th, "S": S, "lmax": lmax, "ny": ny, "top": top}


def features(c):
    """(n_candidates, len(FEATURES)) float32. Everything is local filters, column statistics
    or connected-piece statistics: no fitted line model, so nothing assumes a tyre shape."""
    iu, iv, g, th, S = c["iu"], c["iv"], c["g"], c["th"], c["S"]
    h, w = g.shape
    n = len(iu)
    if n == 0:
        return np.zeros((0, len(FEATURES)), np.float32)
    s, con = S[iv, iu], th[iv, iu]
    S3 = -_ridge(th, 3.0)[0] * 9.0
    th7 = _tophat(g, 7)
    mean = cv2.blur(g, (15, 15))
    std = np.sqrt(np.maximum(cv2.blur(g * g, (15, 15)) - mean ** 2, 0))
    # ridge strength of CANDIDATES only, spread +-2 rows, then averaged along the row:
    # the laser has neighbours to its left and right, a glint or a spike does not
    Sc = cv2.dilate(np.where(c["ok"], S, 0).astype(np.float32), np.ones((5, 1), np.uint8))
    sup10, sup30 = cv2.blur(Sc, (21, 1)), cv2.blur(Sc, (61, 1))
    band = cv2.blur(g, (9, 1))                                 # brightness left/right of the point
    # --- column statistics
    order = np.lexsort((-s, iu))                               # by column, strongest first
    first = np.r_[True, np.diff(iu[order]) != 0]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank = np.empty(n, np.float32); rank[order] = np.arange(n) - start
    best = order[start]                                        # index of the column's strongest candidate
    v_best = np.empty(n, np.float32); v_best[order] = c["v"][best]
    s_best = np.empty(n, np.float32); s_best[order] = s[best]
    col_n = np.bincount(iu, minlength=w)[iu]
    # --- connected pieces (candidates linked across gaps of <= 2 px)
    m = cv2.dilate(c["ok"].astype(np.uint8), np.ones((3, 3), np.uint8))
    n_lab, lab = cv2.connectedComponents(m, connectivity=8)
    p = lab[iv, iu]
    cnt = np.bincount(p, minlength=n_lab).astype(np.float32)
    def pm(x):                                                 # per-piece mean
        return np.bincount(p, weights=x, minlength=n_lab) / np.maximum(cnt, 1)
    u_, v_ = c["u"], c["v"]
    mu, mv = pm(u_), pm(v_)
    du, dv = u_ - mu[p], v_ - mv[p]
    suu, suv, svv = pm(du * du), pm(du * dv), pm(dv * dv)
    slope = suv / np.maximum(suu, 1e-3)                        # least-squares line v = slope * u per piece
    resid = np.sqrt(np.maximum(svv - slope * suv, 0))          # rms distance from it: straightness
    ps = pm(s)
    score = cnt * ps
    # --- the row most candidates' column-winners agree on (a 1-number stand-in for 'the main chain')
    vb = c["v"][rank == 0]
    hist = np.bincount(np.clip(vb.astype(int), 0, h - 1), minlength=h).astype(np.float32)
    mode = float(np.argmax(np.convolve(hist, np.ones(5), "same")))
    F = np.c_[
        s, con, g[iv, iu], s / c["top"], con / max(float(con.max()), 1e-6), S3[iv, iu] / np.maximum(s, 1e-6),
        th7[iv, iu] / np.maximum(con, 1e-6),
        np.maximum(c["lmax"][iv, iu], 0) * SIGMA ** 2 / np.maximum(s, 1e-6), np.abs(c["ny"][iv, iu]),
        mean[iv, iu], std[iv, iu], con / np.maximum(std[iv, iu], 1.0),
        sup10[iv, iu], sup30[iv, iu], sup30[iv, iu] / c["top"], band[iv, iu] / np.maximum(mean[iv, iu], 1.0),
        col_n, rank, v_ - v_best, s / np.maximum(s_best, 1e-6),
        cnt[p], np.sqrt(12 * suu[p]), np.sqrt(12 * svv[p]), resid[p], ps[p] / c["top"], score[p] / score.max(),
        slope[p], v_ - mode, np.abs(v_ - mode), np.full(n, n),
    ]
    return F.astype(np.float32)


def _load():
    if MODEL not in _cache:
        if MODEL == "hgb":
            with open(os.path.join(MODELS, "cand_hgb.pkl"), "rb") as fh:
                _cache[MODEL] = pickle.load(fh)
        else:
            with open(os.path.join(MODELS, f"cand_{MODEL}.json")) as fh:
                _cache[MODEL] = json.load(fh)
    return _cache[MODEL]


def predict(F):
    """P(laser) for every candidate."""
    m = _load()
    if MODEL == "hgb":
        return m["clf"].predict_proba(F)[:, 1]
    if MODEL == "lr":                                          # z-score, dot product, sigmoid
        z = (np.sign(F) * np.log1p(np.abs(F)) - np.array(m["mean"])) / np.array(m["scale"])
        return 1 / (1 + np.exp(-(z @ np.array(m["coef"]) + m["intercept"])))
    x = F[:, FEATURES.index(m["feature"])]                     # "thresh": one feature, one cut
    return ((x >= m["cut"]) if m["sign"] > 0 else (x <= m["cut"])).astype(float)


def extract_line(gray, off=None, return_all=False):
    c = candidates(gray)
    F = features(c)
    prob = predict(F) if len(F) else np.zeros(0)
    keep = prob >= _load()["thresh"]
    if return_all:
        return c, F, prob, keep
    return {"u": c["u"][keep], "v": c["v"][keep], "strength": F[keep, 0], "prob": prob[keep]}
