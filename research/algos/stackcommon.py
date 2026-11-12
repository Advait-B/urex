"""stackcommon.py - glue for the stack_*.py modules: line extractor -> depth stage -> cap -> verdict rule -> gate.

Nothing new is computed here; every piece is imported from a round-one branch:
  line    any extract_line(gray, off)          (dp_dense, pk_fir2d_rows, baseline_steger, onoff_*)
  depth   robust_depth.stage, rd_final config  (or bench.depth for the ablation rows)
  cap     a tyre groove is at most ~25 mm deep: a deeper "groove" is a glint or a second line, drop it
  lb      use the lower_bound flag in the verdict (see verdict_rule)
  gate    "usable" < 0.5 = retake: frame gate of gated_baseline (unchanged) + risk gate re-fitted for this stack
"""
import json
import os

import numpy as np

from algos import robust_depth
from bench import depth as bench_depth

RD = {"floor": "dbscan", "detect": "shape", "lo_k": 3.5}          # = algos/rd_final.CONFIG
LEGAL_MM = 1.6
CAP_MM = 25.0                    # new bus tyre: 14-20 mm; nothing on a tyre is deeper than this
CAP_PX = CAP_MM * 2.1 * 1.5      # no calibration: 2.1 px/mm at the design distance, x1.5 for a camera held closer
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


def apply_cap(out):
    cap = CAP_MM if out["unit"] == "mm" else CAP_PX
    out["capped"] = [g for g in out["grooves"] if g["depth"] > cap]
    out["grooves"] = [g for g in out["grooves"] if g["depth"] <= cap]
    return out


def verdict_rule(grooves):
    """-> "PASS" | "FAIL" | "RETAKE" | "NONE".  A lower_bound groove means ">= depth" (no flat floor seen):
    it can prove PASS (>= 1.6 mm) but it cannot prove FAIL."""
    if not grooves:
        return "NONE"
    if any(g["depth"] < LEGAL_MM and not g.get("lower_bound") for g in grooves):
        return "FAIL"                                             # a groove with a seen floor is illegal
    if any(g["depth"] < LEGAL_MM for g in grooves):
        return "RETAKE"                                           # only "at least x mm" with x < 1.6: cannot decide
    return "PASS"


def stack_features(gray, line, out):
    """Numbers the re-fitted risk gate looks at: all come from the stack's own line and grooves."""
    h, w = gray.shape
    u, v = np.asarray(line["u"], float), np.asarray(line["v"], float)
    g = out["grooves"] + out.get("capped", [])
    k = 2.1 if out["unit"] == "mm" else 1.0                       # depths in px-equivalents: same meaning either way
    f = {"n_pts": len(u) / w, "u_coverage": 0.0, "max_gap_frac": 1.0, "multi_frac": 0.0, "v_std": 0.0,
         "inlier_frac": out.get("inlier_frac", 0.0), "n_grooves": float(len(g)), "n_capped": float(len(out.get("capped", []))),
         "max_depth": k * max([x["depth"] for x in g], default=0.0), "depth_spread": k * float(np.ptp([x["depth"] for x in g])) if g else 0.0,
         "max_sigma": k * max([x.get("sigma", 0.0) for x in g], default=0.0), "min_n": float(min([x["n"] for x in g], default=0)),
         "min_floor_px": min([x.get("floor_px", 0.0) for x in g], default=0.0), "n_lower_bound": float(sum(bool(x.get("lower_bound")) for x in g)),
         "min_width": min([x["u1"] - x["u0"] for x in g], default=0.0),
         "edge_groove": float(any(x["u0"] < 6 or x["u1"] > w - 7 for x in g)),
         "sat_frac": float((gray >= 250).mean()), "img_mean": float(gray.mean()) / 255.0,
         "bg_grad": float(np.hypot(np.ptp(gray[::8, ::8].mean(0)), np.ptp(gray[::8, ::8].mean(1)))) / 255.0}
    if len(u) >= 12:
        cols = np.round(u).astype(int)
        uc = np.unique(cols)
        order = np.argsort(cols, kind="stable")
        cs, vs = cols[order], v[order]
        start = np.r_[0, np.nonzero(np.diff(cs))[0] + 1]
        vr = np.maximum.reduceat(vs, start) - np.minimum.reduceat(vs, start)
        f.update({"u_coverage": len(uc) / w, "max_gap_frac": float(np.max(np.diff(np.r_[0, uc, w - 1])) / w),
                  "multi_frac": float((vr > 3).mean()), "v_std": float(np.std(v)) / h})
    return f


FEATURES = list(stack_features(np.zeros((8, 8), np.uint8), {"u": [], "v": []}, {"grooves": [], "unit": "mm"}))
_models = {}


def _prob(name, f):
    if name not in _models:
        _models[name] = json.load(open(os.path.join(MODELS, name + ".json")))
    m = _models[name]
    x = (np.array([f[k] for k in m["features"]]) - m["mean"]) / np.array(m["scale"])
    return float(1.0 / (1.0 + np.exp(-(x @ np.array(m["coef"]) + m["intercept"])))), m.get("threshold", 0.5)


def measure(gray, calib=None, off=None, line_fn=None, depth="rd", cap=False, lb=False, frame_gate=False, risk_model=None):
    line = line_fn(gray, off)
    if depth == "rd":
        out = robust_depth.stage(line["u"], line["v"], calib, **RD)
    else:
        out = bench_depth.measure_from_line(line["u"], line["v"], calib)
    out["line"] = line
    if cap:
        apply_cap(out)
    out["verdict"] = verdict_rule(out["grooves"]) if calib is not None else None
    if not (lb or frame_gate or risk_model):
        return out                                                # no abstaining: every frame is answered (as the baseline)
    usable = 1.0 if out["grooves"] else 0.0                       # with a gate, "no groove found" is also a retake
    if lb and out["verdict"] == "RETAKE":
        usable = 0.4
    if frame_gate and usable >= 0.5:                              # "is there a laser line on a tyre at all?"
        from algos import gate_features as gf
        f, _ = gf.features(gray)
        out["p_frame"], _ = _prob("frame_lr", f)
        usable = min(usable, out["p_frame"])
    if risk_model and calib is not None and usable >= 0.5:        # "is one of these depths probably wrong?"
        out["p_bad"], t = _prob(risk_model, stack_features(gray, line, out))
        if out["p_bad"] > t:
            usable = 0.4
    out["usable"] = usable
    return out
