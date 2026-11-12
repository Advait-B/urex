"""datasets.py - the fixed test sets every algorithm is scored on.

Simulated (generated once, cached in $UREX_DATA, default /home/claude/urex_data):
  ideal   150 frames  holder locked, clean tyre, shade            (the product as designed)
  stress  200 frames  rocking, sunlight, glints, stones, blur...   (a bad day in the depot)
  legal   150 frames  shallow grooves 0.6-3.5 mm, mild conditions  (the pass/fail boundary at 1.6 mm)
  train  1200 frames  mixed, DIFFERENT seed                        (for ML training only: never test on it)
  calib    15 plates  flat plate at +6 .. -22 mm                   (shim calibration)
Each item: {"on": uint8 HxW, "off": uint8 HxW, "truth": dict from holder_sim.render}.

Real: the 86 hand-held photos (pipeline/data/photos), rotated so the laser line runs
LEFT-RIGHT and deeper = DOWN, the same convention as the simulator. 12 of them carry a
reference line and depth (research/bench/real_refs.json), made by pipeline/single_frame.py
and checked by eye on 12 Nov. They are 'silver' labels: good, but derived from the baseline.
"""
import csv
import json
import os
import pickle

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.environ.get("UREX_DATA", "/home/claude/urex_data")
SPECS = {"ideal": ("ideal", 150, 101), "stress": ("stress", 200, 202), "legal": ("legal", 150, 303),
         "train": ("train", 1200, 404)}
CALIB_HEIGHTS = [6, 4, 2, 0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22]


def _legal_scene(rng, hs):
    s = hs.sample_scene(rng, "ideal")
    base = rng.uniform(0.6, 3.5)
    for g in s["grooves"]:
        g["d"] = float(np.clip(base + rng.normal(0, 0.25), 0.4, 4.0))
    s["ambient"] = float(rng.uniform(0.05, 0.5))
    s["blur_px"] = float(rng.uniform(0.6, 1.5))
    s["tilt_x_deg"], s["tilt_y_deg"] = float(rng.normal(0, 0.5)), float(rng.normal(0, 0.5))
    return s


def load_sim(name):
    """name in ideal / stress / legal / train / calib. Returns a list of items."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"sim_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "sim"))
    import holder_sim as hs
    items = []
    if name == "calib":
        rng = np.random.default_rng(909)
        for z in CALIB_HEIGHTS:
            on, off, t = hs.render(hs.flat_scene(z, rng))
            items.append({"on": on, "off": off, "truth": t, "height_mm": float(z)})
    else:
        mode, n, seed = SPECS[name]
        rng = np.random.default_rng(seed)
        for _ in range(n):
            s = _legal_scene(rng, hs) if mode == "legal" else hs.sample_scene(rng, mode)
            on, off, t = hs.render(s)
            items.append({"on": on, "off": off, "truth": t})
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(items, fh, protocol=4)
    os.replace(tmp, path)
    return items


def to_canonical(gray_native):
    """Hand-held photos have the line top-to-bottom with deeper = LEFT. Rotate 90 deg
    anticlockwise: the line then runs left-right and deeper = DOWN."""
    return np.ascontiguousarray(np.rot90(gray_native, 1))


def load_real():
    """All 86 photos: [{"name", "gray" (canonical), "usable", "laser", "scene", "feature", "ref" or None}]."""
    folder = os.path.join(REPO, "pipeline", "data", "photos")
    with open(os.path.join(HERE, "real_refs.json")) as fh:
        refs = json.load(fh)
    out = []
    for r in csv.DictReader(open(os.path.join(REPO, "pipeline", "data", "photo_labels.csv"))):
        g = cv2.imread(os.path.join(folder, r["file"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        ref = refs.get(r["file"])
        if ref:
            ref = {**ref, "u": np.array(ref["u"], np.float32), "v": np.array(ref["v"], np.float32)}
        out.append({"name": r["file"], "gray": to_canonical(g), "usable": int(r["usable"]), "laser": int(r["laser"]),
                    "scene": r["scene"], "feature": r["feature"], "ref": ref})
    return out
