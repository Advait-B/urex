"""e2e_gbm.py - end-to-end learner 2: classic ML on a compact hand-made description of the frame.

Step 1 (no learning): in every image column take the brightest row after a small blur and a
background subtraction, refined by a centroid -> the line position v(u). No chain selection,
no outlier rejection, no calibration.
Step 2 (learned): v(u) minus its median, resampled to 64 values, plus its quantiles, its deepest
points and the line brightness, go into a gradient-boosted regressor that outputs ONE number: the depth in
mm of the shallowest groove in view. Trained by experiments/end-to-end/train_gbm.py.

Because the model gives one number for the frame, measure() returns one "groove" spanning the
whole width. The harness then scores it against the first true groove only: miss_rate is
large by construction. The fair numbers for this model are verdict_acc and false_pass_rate.
"""
import os

import cv2
import joblib
import numpy as np

NAME = "e2e_gbm"
USES_OFF = False
W, H, NB, R0, R1 = 320, 240, 64, 80, 192
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "e2e_gbm.joblib")
QS = np.linspace(2, 98, 17)


def cheap_line(gray):
    """-> v (sub-pixel row of the brightest point) and its brightness for each of the 320 columns."""
    g = gray if gray.shape == (H, W) else cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
    g = (g - cv2.blur(g, (1, 31)))[R0:R1]                     # remove the slow background; keep the line's band
    r = g.argmax(0)
    rows = np.clip(r[None, :] + np.arange(-3, 4)[:, None], 0, g.shape[0] - 1)
    w = np.clip(np.take_along_axis(g, rows, 0), 0, None)
    v = (w * rows).sum(0) / np.maximum(w.sum(0), 1e-6) + R0
    return v, g.max(0)


def describe(v, a, kind="full"):
    """Line position per column -> fixed-length feature vector."""
    rel = v - np.median(v)                                    # most columns are tread surface
    bins = rel.reshape(NB, -1)                                # 5 columns per bin
    f = [np.percentile(rel, QS), [np.median(a), np.percentile(a, 10), np.median(v)]]
    if kind in ("profile", "full"):
        f.append(np.median(bins, axis=1))                     # the profile across the tread
    if kind == "full":
        f += [bins.max(axis=1), np.sort(rel)[::-1][:40:2]]    # deepest point per bin; the 40 deepest columns
    return np.concatenate(f).astype(np.float32)


def features(gray, kind="full"):
    return describe(*cheap_line(gray), kind)


_m = None


def predict_min_depth(gray):
    global _m
    if _m is None:
        _m = joblib.load(MODEL)
    return float(_m.predict(features(gray)[None])[0])


def measure(gray, calib=None, off=None):
    d = predict_min_depth(gray)
    return {"grooves": [{"u0": 0.0, "u1": float(gray.shape[1] - 1), "depth": d, "unit": "mm"}], "usable": 1.0}
