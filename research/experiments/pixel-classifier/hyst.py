"""hyst.py - does hysteresis (keep P >= lo points whose connected piece has >= 5 points with P >= hi) help?
Scored with the harness functions on held-out train frames 900..1199 and, as a DIAGNOSTIC only, on real."""
import os, sys
import cv2
import numpy as np
sys.path.insert(0, os.getcwd())
from bench import datasets, run, calib, depth
from algos import cand_clf as cc

def prep(img):
    c = cc.candidates(img); F = cc.features(c)
    lab = cv2.connectedComponents(cv2.dilate(c["ok"].astype(np.uint8), np.ones((3, 3), np.uint8)), connectivity=8)[1]
    return c["u"], c["v"], F, lab[c["iv"], c["iu"]]

def keep(p, piece, hi, lo):
    k = p >= hi
    if lo is None or not k.any():
        return k
    seeds = np.bincount(piece[k], minlength=piece.max() + 1)
    return k | ((p >= lo) & (seeds[piece] >= 5))

sim = datasets.load_sim("train")[900:]; real = datasets.load_real(); cal = calib.default_calib()
pre = {id(it["on"]): prep(it["on"]) for it in sim}
pre.update({id(it["gray"]): prep(it["gray"]) for it in real})
for model in ("hgb", "hgb_local"):
    cc.MODEL = model
    P = {k: (cc.predict(F) if len(F) else np.zeros(0)) for k, (u, v, F, pc) in pre.items()}
    for hi, lo in ((0.95, None), (0.9, None), (0.5, None), (0.95, 0.5), (0.95, 0.2), (0.9, 0.2), (0.95, 0.05)):
        def measure(gray, c=None, off=None):
            u, v, F, pc = pre[id(gray)]; k = keep(P[id(gray)], pc, hi, lo)
            out = depth.measure_from_line(u[k], v[k], c); out["line"] = {"u": u[k], "v": v[k]}
            return out
        r = run.eval_sim(measure, sim, cal); q = run.eval_real(measure, real)
        print(model, hi, lo, "VAL f1=%.3f mae=%.3f p95=%.3f miss=%.3f false=%.3f verd=%.3f" % tuple(r[k] for k in ("line_f1", "depth_mae_mm", "depth_p95_mm", "miss_rate", "false_per_frame", "verdict_acc")),
              "| REAL f1ref=%.3f maepx=%s found=%s usable=%.3f falsemeas=%.3f" % (q["line_f1_ref"], q["depth_mae_px"] and round(q["depth_mae_px"], 2), q["found_ref"], q["usable_recall"], q["false_measure"]), flush=True)
