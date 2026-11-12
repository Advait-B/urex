"""hybrid.py - classifier as a clutter filter (drop P < lo), THEN the baseline's hand-written chain selection.
Held-out train frames 900..1199 with the harness scorer; real as a diagnostic only."""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from bench import datasets, run, calib, depth
from algos import cand_clf as cc, baseline_steger as bs
sf = bs.sf

def prep(img):
    c = cc.candidates(img)
    return c, cc.features(c)

def chain(c, k, shape):
    if k.sum() == 0:
        return k
    pts = {"x": c["v"][k], "y": c["u"][k], "ix": c["iv"][k], "iy": c["iu"][k], "strength": c["S"][c["iv"][k], c["iu"][k]],
           "contrast": c["th"][c["iv"][k], c["iu"][k]]}
    d = sf.DEFAULTS
    kk, _ = sf.select_chain(pts, (shape[1], shape[0]), d["chain_gap"], d["keep_within_px"], d["min_piece"], d["min_rel_contrast"])
    out = np.zeros(len(k), bool); out[np.nonzero(k)[0][kk]] = True
    return out

sim = datasets.load_sim("train")[900:]; real = datasets.load_real(); cal = calib.default_calib()
pre = {id(it["on"]): prep(it["on"]) for it in sim}
pre.update({id(it["gray"]): prep(it["gray"]) for it in real})
for model in ("hgb", "hgb_local"):
    cc.MODEL = model
    P = {k: (cc.predict(F) if len(F) else np.zeros(0)) for k, (c, F) in pre.items()}
    for lo in (0.2, 0.5):
        def measure(gray, cal_=None, off=None):
            c, F = pre[id(gray)]; k = chain(c, P[id(gray)] >= lo, gray.shape)
            out = depth.measure_from_line(c["u"][k], c["v"][k], cal_); out["line"] = {"u": c["u"][k], "v": c["v"][k]}
            return out
        r = run.eval_sim(measure, sim, cal); q = run.eval_real(measure, real)
        print(model, lo, "VAL f1=%.3f mae=%.3f p95=%.3f miss=%.3f false=%.3f verd=%.3f" % tuple(r[k] for k in ("line_f1", "depth_mae_mm", "depth_p95_mm", "miss_rate", "false_per_frame", "verdict_acc")),
              "| REAL f1ref=%.3f maepx=%s found=%s usable=%.3f falsemeas=%.3f" % (q["line_f1_ref"], q["depth_mae_px"] and round(q["depth_mae_px"], 2), q["found_ref"], q["usable_recall"], q["false_measure"]), flush=True)
