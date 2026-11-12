"""datasets2.py - two extra simulated test sets that close gaps found by the first round of
research branches (the first sets always put the line at row ~115, thin and level, with
flat-floored grooves, which learned methods exploited).

  shifted   150 frames  three OTHER holder units (laser angle 33 / 37 deg, roll, height), so
                        the line sits at a different row and tilt. Each unit has its own
                        shim calibration (item["calib"]), fitted with the baseline extractor.
                        Scored in mm like ideal/stress.
  handheld  150 frames  looks like the real 28/30 Oct photos: device tilted and held 15 mm
                        closer..25 mm further, laser knocked +-4 deg and rolled, thick and
                        often saturated line, bright textured rubber, rounded groove floors,
                        zig-zag grooves crossed obliquely. No calibration applies, so it is
                        scored in PIXELS: item["truth"]["grooves"][i]["shift_px"] is the
                        true line shift at the deepest visible point of each groove.
Never train on these. They are generated once and cached next to the other sets.
"""
import copy
import os
import pickle
import sys

import numpy as np

from . import calib as calib_mod
from . import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = [  # (theta_deg, laser_roll_deg, Zc)
    {"theta_deg": 33.0, "laser_roll_deg": 1.5, "Zc": 82.0},
    {"theta_deg": 37.0, "laser_roll_deg": -2.0, "Zc": 88.0},
    {"theta_deg": 36.0, "laser_roll_deg": 2.5, "Zc": 85.0},
]


def _hs():
    sys.path.insert(0, os.path.join(HERE, "..", "sim"))
    import holder_sim as hs
    return hs


def add_shift_px(truth):
    """True line shift (px) of each groove: deepest visible line point below the surface line.
    Surface line = quadratic through the truth points that lie on the tread (groove depth 0)."""
    u, v, gd = truth["u"], truth["v"], truth["groove_depth_mm"]
    on = gd < 0.02
    if on.sum() < 10:
        for g in truth["grooves"]:
            g["shift_px"] = None
        return
    c = np.polyfit(u[on], v[on], 2)
    r = v - np.polyval(c, u)
    for g in truth["grooves"]:
        sel = (u >= g["u0"] - 1) & (u <= g["u1"] + 1) & (gd > 0.5 * g["max_visible_depth_mm"])
        g["shift_px"] = float(np.max(r[sel])) if sel.any() else None


def _handheld_scene(rng, hs):
    s = hs.sample_scene(rng, "stress")
    n = int(rng.choice([1, 1, 2]))
    cx = np.sort(rng.uniform(-30, 30, n))
    if n == 2 and cx[1] - cx[0] < 26:
        cx = np.array([cx[0]])
    base = rng.uniform(3.0, 16.0)
    s["grooves"] = [{"x": float(c), "w": float(rng.uniform(9, 15)), "d": float(base + rng.normal(0, 0.6)),
                     "wall_deg": float(rng.uniform(10, 22)), "shape": str(rng.choice(["round", "trap"]))} for c in cx]
    s.update({
        "tilt_x_deg": float(rng.normal(0, 3.0)), "tilt_y_deg": float(rng.normal(0, 3.0)),
        "dz": float(rng.uniform(-15, 25)),            # held closer (-) or further (+) than the feet would allow
        "yaw_deg": float(rng.normal(0, 12.0)), "zigzag_mm": float(rng.choice([0, 3, 5])),
        "ambient": float(rng.uniform(0.5, 1.6)), "rubber_dn": float(rng.uniform(60, 110)),
        "line_w_mm": float(rng.uniform(0.8, 2.6)), "blur_px": float(rng.uniform(0.8, 2.2)),
        "line_peak_dn": float(np.exp(rng.uniform(np.log(180), np.log(900)))),
        "stones": [], "gaps": [], "paint": None, "spikes": 0,
    })
    s["glints"] = s["glints"][:3]
    dev = copy.deepcopy(hs.DEVICE)
    dev["theta_deg"] = float(35 + rng.normal(0, 2.5))
    dev["laser_roll_deg"] = float(rng.normal(0, 4.0))
    return s, dev


def load_sim2(name):
    path = os.path.join(datasets.DATA, f"sim_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    hs = _hs()
    items = []
    if name == "shifted":
        from algos import baseline_steger
        rng = np.random.default_rng(505)
        for k, unit in enumerate(UNITS):
            dev = {**copy.deepcopy(hs.DEVICE), **unit}
            plates = []
            for z in datasets.CALIB_HEIGHTS:
                on, off, t = hs.render(hs.flat_scene(z, rng), dev=dev)
                plates.append({"on": on, "off": off, "truth": t, "height_mm": float(z)})
            cal, info = calib_mod.fit(plates, baseline_steger.extract_line)
            for _ in range(50):
                s = hs.sample_scene(rng, "ideal" if rng.random() < 0.5 else "stress")
                on, off, t = hs.render(s, dev=dev)
                items.append({"on": on, "off": off, "truth": t, "calib": cal.to_json(), "unit": k, "calib_rms": info["rms_mm"]})
    elif name == "handheld":
        rng = np.random.default_rng(606)
        while len(items) < 150:
            s, dev = _handheld_scene(rng, hs)
            on, off, t = hs.render(s, dev=dev)
            add_shift_px(t)
            ok = [g for g in t["grooves"] if g["in_view"] and g["shift_px"] is not None and g["shift_px"] > 3]
            if len(t["u"]) < 400 or not ok:          # the line left the frame or no groove is visible: re-draw
                continue
            items.append({"on": on, "off": off, "truth": t})
    else:
        raise ValueError(name)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(items, fh, protocol=4)
    os.replace(tmp, path)
    return items
