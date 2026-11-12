"""tune.py - choose each model's probability threshold with the harness scorer on the HELD-OUT
train frames 900..1199 (never on ideal/stress/legal). Prints a table; --write stores the choice."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from bench import datasets, run, calib, depth
from algos import cand_clf as cc

items = datasets.load_sim("train")[900:]
cal = calib.default_calib()
pre = {}
for it in items:
    c = cc.candidates(it["on"]); pre[id(it["on"])] = (c["u"], c["v"], cc.features(c))
choice = {}
for model in ("hgb", "lr"):
    cc.MODEL = model
    probs = {k: cc.predict(F) if len(F) else np.zeros(0) for k, (u, v, F) in pre.items()}
    best = None
    for th in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95):
        def measure(gray, c=None, off=None):
            u, v, _ = pre[id(gray)]; k = probs[id(gray)] >= th
            out = depth.measure_from_line(u[k], v[k], c); out["line"] = {"u": u[k], "v": v[k]}
            return out
        r = run.eval_sim(measure, items, cal)
        print(model, th, " ".join(f"{k}={r[k]:.3f}" for k in ("line_f1", "depth_mae_mm", "depth_p95_mm", "miss_rate", "false_per_frame", "verdict_acc")))
        cost = r["miss_rate"] + r["false_per_frame"] / 2.0 - r["line_f1"]      # ~2 grooves/frame: false and miss on one scale
        if best is None or cost < best[0]:
            best = (cost, th)
    choice[model] = best[1]
print("chosen", choice)
if "--write" in sys.argv:
    m = pickle.load(open("models/cand_hgb.pkl", "rb")); m["thresh"] = choice["hgb"]; pickle.dump(m, open("models/cand_hgb.pkl", "wb"))
    m = json.load(open("models/cand_lr.json")); m["thresh"] = choice["lr"]; json.dump(m, open("models/cand_lr.json", "w"))
