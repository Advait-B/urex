"""calib.py - px -> mm. The same shim calibration as pipeline/calibrate.py, for the holder.

A flat plate is photographed at known heights (datasets.load_sim("calib"): +6 .. -22 mm
relative to the feet plane). A polynomial height_mm = poly(u, v) is fitted through the
laser-line points of all plates. It absorbs the laser angle, the lens and its distortion.
Algorithms get a Calib object and call calib.height(u, v).

    calib = default_calib()          # fitted once with the baseline line extractor, cached
"""
import json
import os

import numpy as np

from . import datasets


class Calib:
    def __init__(self, coef, deg_u, deg_v, w, h, z_range):
        self.coef, self.deg_u, self.deg_v, self.w, self.h, self.z_range = np.asarray(coef), deg_u, deg_v, w, h, z_range

    @staticmethod
    def design(u, v, deg_u, deg_v, w, h):
        a, b = np.asarray(u, float) / (w - 1) - 0.5, np.asarray(v, float) / (h - 1) - 0.5
        return np.stack([a ** i * b ** j for i in range(deg_u + 1) for j in range(deg_v + 1)], axis=-1)

    def height(self, u, v):
        """Surface height in mm above the feet plane (grooves are NEGATIVE-going)."""
        return self.design(u, v, self.deg_u, self.deg_v, self.w, self.h) @ self.coef

    def to_json(self):
        return {"coef": self.coef.tolist(), "deg_u": self.deg_u, "deg_v": self.deg_v, "w": self.w, "h": self.h,
                "z_range": self.z_range}


def fit(items, extract_line, deg_u=4, deg_v=3):
    """items: calib plates. extract_line(gray, off) -> {"u", "v"}."""
    U, V, Z = [], [], []
    h, w = items[0]["on"].shape
    for it in items:
        p = extract_line(it["on"], it["off"])
        u, v = np.asarray(p["u"], float), np.asarray(p["v"], float)
        if len(u) < 20:
            continue
        keep = np.abs(v - np.median(v)) < 6            # a plate gives one clean line; drop strays
        U.append(u[keep]); V.append(v[keep]); Z.append(np.full(keep.sum(), it["height_mm"]))
    U, V, Z = map(np.concatenate, (U, V, Z))
    M = Calib.design(U, V, deg_u, deg_v, w, h)
    coef, *_ = np.linalg.lstsq(M, Z, rcond=None)
    c = Calib(coef, deg_u, deg_v, w, h, [float(Z.min()), float(Z.max())])
    resid = c.height(U, V) - Z
    return c, {"rms_mm": float(np.sqrt(np.mean(resid ** 2))), "max_abs_mm": float(np.abs(resid).max()), "n": int(len(Z))}


def default_calib():
    path = os.path.join(datasets.DATA, "calib_default.json")
    if os.path.exists(path):
        with open(path) as fh:
            return Calib(**json.load(fh))
    from algos import baseline_steger
    c, info = fit(datasets.load_sim("calib"), baseline_steger.extract_line)
    with open(path, "w") as fh:
        json.dump(c.to_json(), fh)
    print("default calibration fitted:", info)
    return c
