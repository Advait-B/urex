"""onoff_common.py - shared pieces of the laser-on / laser-off algorithms (not an algorithm itself).

    difference      on - off (float), optionally gain-matched / registered / conservative
    steger_line     the baseline detector (top-hat + Steger + chain selection) on any image
    cog_line        the cheap detector: per-column peak + centre of gravity
"""
import cv2
import numpy as np

from . import baseline_steger as base

sf = base.sf


def steger_line(img, min_contrast=None):
    """Baseline detector on a float/uint8 image in the canonical orientation."""
    g = np.ascontiguousarray(np.asarray(img, np.float32).T)
    c = sf.DEFAULTS
    mc = c["min_contrast"] if min_contrast is None else min_contrast
    pts = sf.steger_points(g, c["sigma"], c["tophat"], mc, c["rel_strength"])
    keep, _ = sf.select_chain(pts, g.shape, c["chain_gap"], c["keep_within_px"], c["min_piece"], c["min_rel_contrast"])
    return {"u": pts["y"][keep], "v": pts["x"][keep], "strength": pts["strength"][keep]}


def cog_line(d, sigma=1.2, k_noise=6.0, min_dn=12.0, rel=0.25, win=4, jump_px=3.0):
    """Per-column centre of gravity on a difference image d (float, background ~ 0).
    One point per column: brightest row after a small blur, refined by the intensity-weighted
    centroid of everything above half the peak within +-win rows. Columns are dropped when
    the peak is not clearly above the noise or when they disagree with both neighbours."""
    h, w = d.shape
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) + 0.5       # MAD of the difference
    s = cv2.GaussianBlur(d.astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=sigma)
    p = s.argmax(axis=0)
    cols = np.arange(w)
    peak = s[p, cols]
    thr = max(min_dn, k_noise * noise / 2.0, rel * np.percentile(peak, 60))
    v = np.full(w, np.nan)
    rows = np.arange(-win, win + 1)
    for c in np.nonzero(peak > thr)[0]:
        r = np.clip(p[c] + rows, 0, h - 1)
        wgt = np.clip(s[r, c] - 0.5 * peak[c], 0, None)
        if wgt.sum() > 0:
            v[c] = (r * wgt).sum() / wgt.sum()
    # continuity: keep a column only if a neighbour within 2 columns agrees to jump_px
    ok = np.isfinite(v)
    near = np.zeros(w, bool)
    for dc in (-2, -1, 1, 2):
        vs = np.roll(v, dc)
        if dc > 0:
            vs[:dc] = np.nan
        else:
            vs[dc:] = np.nan
        near |= np.abs(v - vs) < jump_px * abs(dc)
    ok &= near
    return {"u": cols[ok].astype(float), "v": v[ok], "strength": peak[ok]}


# ------------------------------------------------------------------ differences
def gain_match(on, off):
    """Exposure ratio on/off from the darker 60 % of the on frame (the laser is in the bright part)."""
    m = (on < np.percentile(on, 60)) & (off > 4)
    if m.sum() < 500:
        return 1.0
    return float(np.clip(np.median(on[m]) / max(np.median(off[m]), 1e-3), 0.6, 1.6))


def register(on, off, max_shift=6.0, min_response=0.08):
    """Translate `off` onto `on` by phase correlation (ambient texture is common to both).
    The laser line is only in `on`, so the correlation is run on images clipped at the off
    frame's bright level. Returns (off_registered, (dx, dy), response)."""
    hi = np.percentile(off, 99.5) + 10
    a, b = np.minimum(on, hi), np.minimum(off, hi)
    a, b = a - cv2.GaussianBlur(a, (0, 0), 12), b - cv2.GaussianBlur(b, (0, 0), 12)   # drop the slow gradient
    win = cv2.createHanningWindow(on.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(b, a, win)
    if resp < min_response or np.hypot(dx, dy) > max_shift:
        return off, (0.0, 0.0), float(resp)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(off, M, on.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), (dx, dy), float(resp)


def difference(on, off, mode="plain", guard_px=2, blur=1.0):
    """mode: plain | gain | reg (gain + phase correlation) | cons (gain + conservative)
             | robust (gain + registration + conservative with a 1 px guard)."""
    on, off = on.astype(np.float32), off.astype(np.float32)
    if mode == "plain":
        return np.clip(on - off, 0, None)
    off = off * gain_match(on, off)
    if mode in ("reg", "robust"):
        off, _, _ = register(on, off)
    if mode in ("cons", "robust"):
        # conservative: subtract the brightest off value within guard_px, so a background
        # feature that moved by up to guard_px is still fully removed (never leaves a residue)
        r = guard_px if mode == "cons" else 1
        off = cv2.dilate(cv2.GaussianBlur(off, (0, 0), blur), np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    return np.clip(on - off, 0, None)
