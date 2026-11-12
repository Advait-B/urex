"""make_real_refs.py - write real_refs.json: reference line + depth for the 12 photos that
pipeline/single_frame.py measured and that were checked by eye (results/2026-11-12).
Run once; the JSON is committed."""
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
import single_frame as sf  # noqa: E402

ACCEPTED = {r["file"]: r for r in csv.DictReader(open(os.path.join(REPO, "pipeline", "results", "2026-11-12", "results.csv")))
            if r["accepted"] == "1"}
out = {}
for name, row in ACCEPTED.items():
    g = cv2.imread(os.path.join(REPO, "pipeline", "data", "photos", name), 0)
    res = sf.analyse(g)
    W = g.shape[1]
    x, y = res["_x"], res["_y"]                      # native: line top-to-bottom
    ref = {"u": np.round(y, 2).tolist(), "v": np.round((W - 1) - x, 2).tolist(),   # canonical (rot90 anticlockwise)
           "depth_px": round(float(res["depth_px"]), 2), "mode": res["mode"], "feature": row["feature"]}
    if res["mode"] == "groove":
        gr = res["groove"]
        ref["groove_u"] = [round(float(gr["rows"][0]), 1), round(float(gr["rows"][1]), 1)]
    out[name] = ref
with open(os.path.join(HERE, "real_refs.json"), "w") as fh:
    json.dump(out, fh)
print(len(out), "references written")
