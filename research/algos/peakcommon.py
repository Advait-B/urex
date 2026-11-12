"""peakcommon.py - shared frame for the classical per-column peak detectors (NOT an algorithm itself).

Every pk_* module is this same four-step pipeline with a different step 3:

  1. background   'none' | 'tophat' (15 px ellipse, as the baseline) | 'tophat1d' (vertical only) |
                  'colpct' (subtract each column's percentile)
  2. candidates   response = background-removed image smoothed by a small Gaussian; in every COLUMN
                  keep up to `max_per_col` local maxima above a threshold. More than one per column
                  is needed: a hooked groove profile or a glint puts two bright spots in one column.
  3. sub-pixel    refine each candidate's v with the chosen detector (cog / gauss3 / blais / fir)
  4. laser chain  the baseline's select_chain (pipeline/single_frame.py) drops glints and edges,
                  so the comparison with baseline_steger is only about steps 1-3.

Convention: gray is HxW uint8, line runs left-right, v = row (sub-pixel), u = column (integer here).
"""
import os
import sys

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pipeline")))
import single_frame as sf  # noqa: E402

CFG = {
    "bg": "tophat", "bg_size": 15, "bg_pct": 50,
    "sigma_v": 1.0,          # candidate smoothing down the column (px)
    "sigma_u": 0.0,          # smoothing along the line (px); 0 = pure per-column
    "min_contrast": 12.0,    # grey levels above background (of the smoothed response)
    "rel": 0.25,             # ...and relative to the frame's typical line peak (median of column maxima)
    "max_per_col": 3,
    "detector": "cog", "half": 3,     # detector name and its half window / filter half length
    "refine_smooth_u": 0.0,  # smooth the image the detector reads along u first
    "rows": False,           # also detect along the rows where the line is steep (see extract)
    "rows_ratio": 1.0, "rows_flank": 0.5,
    "chain": True,
}


# ------------------------------------------------------------------ 1 background
def remove_background(gray, mode, size=15, pct=50):
    g = gray.astype(np.float32)
    if mode == "none":
        return g
    if mode == "tophat":
        return sf.tophat(gray, size)
    if mode == "tophat1d":                       # opening with a vertical bar only: O(1) row buffers on an MCU
        return g - cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((size, 1), np.uint8))
    if mode == "colpct":
        return np.clip(g - np.percentile(g, pct, axis=0, keepdims=True), 0, None)
    raise ValueError(mode)


# ------------------------------------------------------------------ 2 candidates
def smooth(img, sv, su):
    out = img
    if sv > 0:
        out = ndi.gaussian_filter1d(out, sv, axis=0, mode="nearest")
    if su > 0:
        out = ndi.gaussian_filter1d(out, su, axis=1, mode="nearest")
    return out


def threshold(resp, min_contrast, rel):
    """The line crosses most columns, so the median of the column maxima is 'a typical line peak'."""
    return max(min_contrast, rel * float(np.median(resp.max(axis=0))))


def column_peaks(resp, thr, max_per_col):
    """Integer (iv, iu) of the strongest local maxima above thr down each column."""
    c = resp[1:-1]
    m = (c > resp[:-2]) & (c >= resp[2:]) & (c > thr)
    m[:2] = False                                # leave room for the detector windows
    m[-2:] = False
    iv, iu = np.nonzero(m)
    iv = iv + 1
    if max_per_col and len(iv):
        order = np.lexsort((-resp[iv, iu], iu))  # by column, strongest first
        iv, iu = iv[order], iu[order]
        first = np.r_[0, np.nonzero(np.diff(iu))[0] + 1]
        rank = np.arange(len(iu)) - np.repeat(first, np.diff(np.r_[first, len(iu)]))
        iv, iu = iv[rank < max_per_col], iu[rank < max_per_col]
    return iv, iu


def windows(img, iv, iu, half):
    """(N, 2*half+1) pixel values down the column around each candidate (edge rows repeated)."""
    off = np.arange(-half, half + 1)
    rows = np.clip(iv[:, None] + off[None, :], 0, img.shape[0] - 1)
    return img[rows, iu[:, None]].astype(np.float64), off


# ------------------------------------------------------------------ 3 detectors
def cog(W, off):
    """Centre of gravity. The window minimum is taken off first, otherwise any leftover
    background pulls the answer to the window centre."""
    w = W - W.min(axis=1, keepdims=True)
    return (w * off).sum(1) / np.maximum(w.sum(1), 1e-9)


def gauss3(W, off):
    """Gaussian through the max and its two neighbours: a parabola through their logs."""
    h = len(off) // 2
    a, b, c = (np.log(np.maximum(W[:, h + k], 1.0)) for k in (-1, 0, 1))
    den = a - 2 * b + c
    d = np.where(den < -1e-6, 0.5 * (a - c) / np.where(den < -1e-6, den, -1.0), 0.0)
    return np.clip(d, -1.0, 1.0)


def parabola3(W, off):
    h = len(off) // 2
    a, b, c = W[:, h - 1], W[:, h], W[:, h + 1]
    den = a - 2 * b + c
    d = np.where(den < -1e-6, 0.5 * (a - c) / np.where(den < -1e-6, den, -1.0), 0.0)
    return np.clip(d, -1.0, 1.0)


def blais_rioux(W, off, order=4):
    """Derivative filter g(i) = sum_{k=1..order/2} f(i+k) - f(i-k); the peak is where g crosses
    zero going down. Linear interpolation between the two samples either side of the crossing."""
    n = order // 2
    h = len(off) // 2                            # needs half >= n + 2
    span = np.arange(-2, 3)                      # look for the crossing within +-2 px of the candidate
    g = np.stack([sum(W[:, h + s + k] - W[:, h + s - k] for k in range(1, n + 1)) for s in span], 1)
    cross = (g[:, :-1] >= 0) & (g[:, 1:] < 0)
    dist = np.where(cross, np.abs(span[:-1] + 0.5), 99.0)
    j = dist.argmin(1)
    r = np.arange(len(W))
    g0, g1 = g[r, j], g[r, j + 1]
    d = span[j] + g0 / np.where(g0 - g1 > 0, g0 - g1, 1.0)
    return np.where(cross.any(1), d, 0.0)


def refine(det, img, resp, iv, iu, half):
    if det == "cog":
        return cog(*windows(img, iv, iu, half))
    if det == "gauss3":
        return gauss3(*windows(img, iv, iu, 1))
    if det == "blais4":
        return blais_rioux(*windows(img, iv, iu, 4), order=4)
    if det == "blais8":
        return blais_rioux(*windows(img, iv, iu, 6), order=8)
    if det == "fir":                             # matched filter: the smoothed response itself + parabola
        return parabola3(*windows(resp, iv, iu, 1))
    if det == "fir_log":                         # same, Gaussian (log-parabola) interpolation
        return gauss3(*windows(resp, iv, iu, 1))
    if det == "none":
        return np.zeros(len(iv))
    raise ValueError(det)


# ------------------------------------------------------------------ driver
def one_pass(img, resp, thr, c):
    """Candidates + sub-pixel refinement down the columns of img. -> iv, iu, v (sub-pixel), strength."""
    iv, iu = column_peaks(resp, thr, c["max_per_col"])
    if len(iv) == 0:
        return iv, iu, np.zeros(0), np.zeros(0)
    src = smooth(img, 0, c["refine_smooth_u"]) if c["refine_smooth_u"] > 0 else img
    return iv, iu, iv + refine(c["detector"], src, resp, iv, iu, c["half"]), resp[iv, iu]


def extract(gray, cfg):
    c = {**CFG, **cfg}
    img = remove_background(gray, c["bg"], c["bg_size"], c["bg_pct"])
    resp = smooth(img, c["sigma_v"], c["sigma_u"])
    thr = threshold(resp, c["min_contrast"], c["rel"])
    iv, iu, v, strength = one_pass(img, resp, thr, c)
    u = iu.astype(float)
    if c["rows"]:
        # Second pass ALONG THE ROWS for the near-vertical stretches (groove walls, lug steps), where a
        # column holds a long smear instead of a peak. A row peak is kept only where the response is
        # more sharply curved across u than across v (the line is steeper than 45 degrees) and has
        # little slope down the column (otherwise it is speckle on the shoulder of a level line).
        respT = smooth(img.T, c["sigma_v"], c["sigma_u"]) if c["sigma_v"] != c["sigma_u"] else resp.T
        ju, jv, u2, s2 = one_pass(img.T, respT, thr, c)            # roles swapped: 'rows' of img.T are columns
        ok = (ju > 0) & (ju < gray.shape[1] - 1) & (jv > 0) & (jv < gray.shape[0] - 1)
        ju, jv, u2, s2 = ju[ok], jv[ok], u2[ok], s2[ok]
        cuu = 2 * resp[jv, ju] - resp[jv, ju - 1] - resp[jv, ju + 1]
        cvv = 2 * resp[jv, ju] - resp[jv - 1, ju] - resp[jv + 1, ju]
        gv = 0.5 * np.abs(resp[jv + 1, ju] - resp[jv - 1, ju])    # big on the FLANK of a level line
        steep = (cuu > c["rows_ratio"] * cvv) & (gv < c["rows_flank"] * cuu)
        iv, iu = np.r_[iv, jv[steep]], np.r_[iu, ju[steep]]
        u, v, strength = np.r_[u, u2[steep]], np.r_[v, jv[steep].astype(float)], np.r_[strength, s2[steep]]
    if len(u) == 0:
        return {"u": np.zeros(0), "v": np.zeros(0), "strength": np.zeros(0)}
    if c["chain"]:                               # single_frame works on the transpose: x = v, y = u
        pts = {"x": v, "y": u, "ix": iv, "iy": iu, "strength": strength, "contrast": strength}
        keep, _ = sf.select_chain(pts, gray.T.shape, sf.DEFAULTS["chain_gap"], sf.DEFAULTS["keep_within_px"],
                                  sf.DEFAULTS["min_piece"], sf.DEFAULTS["min_rel_contrast"])
        u, v, strength = u[keep], v[keep], strength[keep]
    return {"u": u, "v": v, "strength": strength}
