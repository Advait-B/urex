"""cnn_line.py - tiny U-Net laser-line segmenter (trained on simulated frames only) + sub-pixel readout.

The network maps the grey image to a heat-map: ~1 on the laser line centre, falling off as a
Gaussian (sigma 1 px) of the distance to the true sub-pixel line. Four ways to turn that
into sub-pixel points are implemented; READOUT picks one (the wrappers cnn_line_*.py
only change this):
  "gated"    centroid of heat-map-weighted image intensity across the line, at each heat-map peak  <- default
  "heat"     soft-argmax: centroid of the heat-map itself across the line at each heat-map peak
  "column"   classic one-point-per-column soft-argmax of the heat-map
  "steger"   the baseline's Steger ridge points on the IMAGE (relaxed thresholds), kept where the mask is on
Weights (float32 state dicts, loaded lazily): research/models/cnn_line.pt (trained with the sim-to-real
augmentation) and cnn_line_noaug.pt (crop + flip only; better on simulation, used by cnn_line_noaug.py).
Training: research/experiments/cnn-segmentation/train.py
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pipeline")))
import single_frame as sf  # noqa: E402

NAME = "cnn_line"
USES_OFF = False
READOUT = "gated"
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "cnn_line.pt")
THR = 0.35            # heat-map level that counts as "laser"
MIN_BLOB = 12         # px: smaller heat-map blobs are dropped
GUIDE_PX = 10        # px: a separate piece this close to the main line's straight guide is kept
TOUCH_PX = 12        # px: ... or one that comes this close to a kept piece
GAP_PX = 60          # px: ... or a piece in a column gap of the line, at most this far from the guide
MIN_POINTS = 40       # fewer points than this in a frame -> report no line at all

torch.set_num_threads(1)


def block(cin, cout, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    """3 down / 3 up, 8-16-32-48 channels, ~85k parameters, fully convolutional."""

    def __init__(self, c=(8, 16, 32, 48)):
        super().__init__()
        self.e1 = nn.Sequential(block(1, c[0]), block(c[0], c[0]))
        self.e2 = nn.Sequential(block(c[0], c[1]), block(c[1], c[1]))
        self.e3 = nn.Sequential(block(c[1], c[2]), block(c[2], c[2]))
        self.mid = nn.Sequential(block(c[2], c[3]), block(c[3], c[3], dilation=2))
        self.d3 = block(c[3] + c[2], c[2])
        self.d2 = block(c[2] + c[1], c[1])
        self.d1 = block(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):                               # x: (N, 1, H, W) in 0..1, H and W multiples of 8
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        up = lambda t: F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=False)  # noqa: E731
        d3 = self.d3(torch.cat([up(m), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.out(d1)                             # logits of the heat-map


_model = {}


def get_model(path=None):
    path = path or WEIGHTS
    if path not in _model:
        m = TinyUNet()
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd["model"] if "model" in sd else sd)
        _model[path] = m.eval()
    return _model[path]


def heatmap(gray, path=None):
    """uint8 HxW -> float32 HxW heat-map in 0..1."""
    h, w = gray.shape
    ph, pw = (-h) % 8, (-w) % 8
    x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    with torch.no_grad():
        p = torch.sigmoid(get_model(path)(x))[0, 0, :h, :w]
    return p.numpy()


def clean_mask(p, thr=None, min_blob=None, select=True):
    """Heat-map -> boolean laser mask. Small blobs are dropped; then (select=True) only ONE line is
    kept: the piece with the largest heat x width is the laser, and other pieces stay only if they
    lie along the straight line through it (within GUIDE_PX), come within TOUCH_PX of a kept piece
    (the hook inside a groove), or sit in columns where the kept line has a gap (a groove floor: the
    wall hides the part that would join it). The network alone marks every thin bright streak (glints)."""
    thr = THR if thr is None else thr
    min_blob = MIN_BLOB if min_blob is None else min_blob
    m = (p > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_blob]
    if not select or len(ids) <= 1:
        ok = np.zeros(n, bool)
        ok[ids] = True
        return ok[lab]
    heat = np.bincount(lab.ravel(), weights=p.ravel(), minlength=n)
    best = max(ids, key=lambda i: heat[i] * stats[i, cv2.CC_STAT_WIDTH])
    vv, uu = np.nonzero(lab == best)
    a, b = np.polyfit(uu, vv, 1) if np.ptp(uu) > 5 else (0.0, float(np.median(vv)))
    keep = lab == best
    todo = [i for i in ids if i != best]
    added = True
    while added and todo:
        added = False
        dist = cv2.distanceTransform((~keep).astype(np.uint8), cv2.DIST_L2, 3)
        for i in list(todo):
            sel = lab == i
            vi, ui = np.nonzero(sel)
            off = np.median(np.abs(vi - (a * ui + b)))
            in_gap = off < GAP_PX and np.mean(keep.any(axis=0)[ui]) < 0.3
            if off < GUIDE_PX or dist[sel].min() <= TOUCH_PX or in_gap:
                keep |= sel
                todo.remove(i)
                added = True
    return keep


# ------------------------------------------------------------ heat-map -> sub-pixel points
def _peaks_1d(p, weight, mask, axis, half=3):
    """Local maxima of p along `axis` inside mask; centre = centroid of `weight` in +-half px.
    Returns (coordinate along axis (sub-pixel), index on the other axis, peak row, peak col)."""
    q = np.moveaxis(p, axis, 0)
    wq = np.moveaxis(weight, axis, 0)
    mq = np.moveaxis(mask, axis, 0)
    pk = np.zeros_like(mq)
    pk[1:-1] = mq[1:-1] & (q[1:-1] >= q[:-2]) & (q[1:-1] > q[2:])
    a, b = np.nonzero(pk)
    n = q.shape[0]
    off = np.arange(-half, half + 1)
    idx = np.clip(a[:, None] + off[None, :], 0, n - 1)
    ww = wq[idx, b[:, None]]
    ww = ww * (np.abs(idx - a[:, None]) <= half)
    pos = (ww * idx).sum(1) / np.maximum(ww.sum(1), 1e-9)
    return pos, b.astype(float), a, b


def points_from_heat(p, gray=None, mask=None, columns_only=False):
    """Sub-pixel points from the heat-map. The line is mostly left-right, so take peaks down
    every COLUMN; on steep parts (groove walls) take peaks along every ROW instead. Which one
    applies at a pixel is decided by the heat-map curvature (larger |2nd derivative| = across the line).
    gray given -> the centroid weights are heat * (image - local background) instead of heat."""
    mask = clean_mask(p) if mask is None else mask
    w = p
    if gray is not None:
        g = gray.astype(np.float32)
        w = p * np.maximum(g - cv2.erode(g, np.ones((9, 9), np.uint8)), 0)
    ps = cv2.GaussianBlur(p, (0, 0), 1.0)
    pvv = np.abs(cv2.Sobel(ps, cv2.CV_32F, 0, 2, ksize=3))
    puu = np.abs(cv2.Sobel(ps, cv2.CV_32F, 2, 0, ksize=3))
    horiz = pvv >= puu                                   # line locally closer to horizontal
    v, u, a, b = _peaks_1d(p, w, mask if columns_only else mask & horiz, axis=0)
    if columns_only:
        return u, v
    u2, v2, a2, b2 = _peaks_1d(p, w, mask & ~horiz, axis=1)
    return np.r_[u, u2], np.r_[v, v2]


def points_steger_gated(gray, mask, grow=0):
    """The baseline's Steger ridge points, relaxed thresholds, no chain selection: the CNN mask
    decides which of them are laser."""
    g = np.ascontiguousarray(gray.T)
    pts = sf.steger_points(g, sf.DEFAULTS["sigma"], sf.DEFAULTS["tophat"], min_contrast=8, rel_strength=0.03)
    m = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) if grow else mask
    keep = m[pts["ix"], pts["iy"]] > 0                  # transposed: ix = row, iy = column of the original
    return pts["y"][keep], pts["x"][keep]


def extract_line(gray, off=None, readout=None, path=None):
    readout = readout or READOUT
    p = heatmap(gray, path)
    mask = clean_mask(p)
    if readout == "steger":
        u, v = points_steger_gated(gray, mask)
    elif readout == "heat":
        u, v = points_from_heat(p, None, mask)
    elif readout == "gated":
        u, v = points_from_heat(p, gray, mask)
    elif readout == "column":
        u, v = points_from_heat(p, None, mask, columns_only=True)
    else:
        raise ValueError(readout)
    if len(u) < MIN_POINTS:
        u, v = np.zeros(0), np.zeros(0)
    return {"u": np.asarray(u, float), "v": np.asarray(v, float), "heat": p, "mask": mask}
