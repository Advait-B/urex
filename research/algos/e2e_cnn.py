"""e2e_cnn.py - end-to-end learner 1: a tiny CNN that reads groove depth straight from the frame.

No line extraction, no calibration. The frame is resized to 320x240, a 112-row band around
the laser line is cut out, and the network predicts for each of the 320 image columns
  (a) the probability that the column lies in a groove, and
  (b) the depth of that groove in mm.
Grooves are read off as runs of columns with p > 0.5; the depth is the median of (b) there.
Trained only on the simulated `train` set (experiments/end-to-end/train_cnn.py).
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

NAME = "e2e_cnn"
USES_OFF = False
W, H, R0, R1 = 320, 240, 80, 192          # canonical size and the band of rows the net sees
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "e2e_cnn.pt")
torch.set_num_threads(1)


def block(i, o, stride):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())


class Net(nn.Module):
    """2-D convs shrink the band 8x vertically and 4x horizontally; the 14 remaining rows are
    stacked into channels (so vertical position = line shift is kept), then 1-D convs along the
    tread give every column the context of its neighbours (the tread surface next to a groove)."""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(block(1, 12, 2), block(12, 24, 2), block(24, 32, (2, 1)), block(32, 32, 1))
        self.g = nn.Sequential(
            nn.Conv1d(32 * 14, 96, 1), nn.ReLU(),
            nn.Conv1d(96, 96, 5, padding=2), nn.ReLU(),
            nn.Conv1d(96, 96, 5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(96, 2, 1))

    def forward(self, x):                       # x: (B, 1, 112, 320) in 0..1
        f = self.f(x)                           # (B, 32, 14, 80)
        y = self.g(f.flatten(1, 2))             # (B, 2, 80)
        y = nn.functional.interpolate(y, scale_factor=4, mode="linear", align_corners=False)
        return y[:, 0], y[:, 1]                 # groove logit, depth in mm (per column, 320 wide)


_net = None


def load():
    global _net
    if _net is None:
        _net = Net()
        _net.load_state_dict(torch.load(MODEL, map_location="cpu"))
        _net.eval()
    return _net


def prep(gray):
    g = gray if gray.shape == (H, W) else cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(g[R0:R1].astype(np.float32) / 255.0)[None, None]


def profile(gray):
    """-> per-column groove probability and depth (mm), both length 320."""
    with torch.no_grad():
        logit, depth = load()(prep(gray))
    return torch.sigmoid(logit)[0].numpy(), depth[0].numpy()


def read_grooves(p, d, scale=1.0, min_w=6):
    on = np.r_[0, (p > 0.5).astype(int), 0]
    starts, ends = np.nonzero(np.diff(on) == 1)[0], np.nonzero(np.diff(on) == -1)[0]
    out = []
    for a, b in zip(starts, ends):              # columns a .. b-1
        if b - a >= min_w:
            out.append({"u0": float(a * scale), "u1": float((b - 1) * scale), "depth": float(np.median(d[a:b])),
                        "unit": "mm", "conf": float(p[a:b].mean())})
    return out


def measure(gray, calib=None, off=None):
    p, d = profile(gray)
    grooves = read_grooves(p, d, gray.shape[1] / W)
    return {"grooves": grooves, "usable": 1.0 if grooves else 0.0}
