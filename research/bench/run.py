"""run.py - score one algorithm on every test set. Run from the research/ folder:

    python -m bench.run algos.baseline_steger                     # all sets, prints a table
    python -m bench.run algos.my_algo --sets ideal stress --limit 40   # quick look while developing
    python -m bench.run algos.my_algo --out results/my_algo.json  # the file the paper reads

AN ALGORITHM is a module in research/algos/ with:
    NAME = "short_name"
    USES_OFF = False            # True if it needs the laser-off frame (then it is skipped on real photos' depth)
    def extract_line(gray, off=None) -> {"u": array, "v": array}     # sub-pixel laser points
  and/or
    def measure(gray, calib=None, off=None) -> {
        "grooves": [{"u0", "u1", "depth", "unit"}],   # unit "mm" when calib is given, else "px"
        "usable": float in 0..1,                      # optional: is this frame worth measuring?
        "line": {"u", "v"},                           # optional
    }
If only extract_line exists, depth comes from bench.depth.measure_from_line (the shared
default). If only measure exists, the line metrics are taken from its "line" (or skipped).
Images are uint8 greyscale in the canonical orientation: line left-right, deeper = down.

METRICS
  simulated sets (exact ground truth):
    line_f1        F1 of predicted points vs the true line at 1 px tolerance
    line_rmse_px   sub-pixel error of the points that are on the line
    depth_mae_mm   mean |error| over measurable grooves that were found
    depth_p95_mm   95th percentile of |error|
    depth_bias_mm  mean signed error (+ = reads too deep)
    miss_rate      measurable grooves not found
    false_per_frame  reported grooves that are not there
    verdict_acc    PASS / not-PASS at the 1.6 mm legal limit;  false_pass = unsafe mistakes
  real photos (no ground truth; 12 'silver' references derived from the baseline + eye check):
    line_f1_ref    agreement with the reference line at 1.5 px (12 frames)
    depth_mae_px   |depth - reference| on the 6 truck-tyre frames, in px
    found_ref      of those 6, how many got a depth at all
    usable_recall  of the 19 frames a person would measure, share with usable >= 0.5
    false_measure  of the 67 frames a person would NOT measure, share reported usable WITH a depth
  ms_per_frame     wall-clock, one thread
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from . import calib as calib_mod
from . import datasets, depth

LEGAL_MM = 1.6


def resolve(mod):
    """-> (extract_line or None, measure)"""
    extract = getattr(mod, "extract_line", None)
    if hasattr(mod, "measure"):
        return extract, mod.measure

    def measure(gray, calib=None, off=None):
        p = extract(gray, off)
        out = depth.measure_from_line(p["u"], p["v"], calib)
        out["line"] = p
        return out
    return extract, measure


def line_scores(pu, pv, tu, tv, tol):
    if len(tu) == 0:
        return None
    if len(pu) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "rmse": float("nan")}
    T = np.c_[tu, tv]
    P = np.c_[pu, pv]
    d_p, _ = cKDTree(T).query(P)
    Ts = np.unique(np.round(T * 2) / 2, axis=0)            # truth resampled on a 0.5 px grid
    d_t, _ = cKDTree(P).query(Ts)
    prec, rec = float(np.mean(d_p < tol)), float(np.mean(d_t < tol))
    on = d_p[d_p < tol]
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "rmse": float(np.sqrt(np.mean(on ** 2))) if len(on) else float("nan")}


def match_grooves(pred, truth):
    """truth groove -> best overlapping predicted groove (or None); also the unmatched predictions."""
    used, pairs = set(), []
    for t in truth:
        best, best_ov = None, 0.0
        for j, p in enumerate(pred):
            if j in used:
                continue
            ov = min(p["u1"], t["u1"] + 4) - max(p["u0"], t["u0"] - 4)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None:
            used.add(best)
        pairs.append((t, pred[best] if best is not None else None))
    return pairs, [p for j, p in enumerate(pred) if j not in used]


def eval_sim(measure, items, cal, limit=None):
    items = items[:limit] if limit else items
    f1, rmse, errs, true_d, pred_d = [], [], [], [], []
    miss = n_meas = false = 0
    v_ok = false_pass = 0
    t0 = time.perf_counter()
    outs = [measure(it["on"], cal, it["off"]) for it in items]
    ms = (time.perf_counter() - t0) / len(items) * 1000
    for it, out in zip(items, outs):
        tr = it["truth"]
        if out.get("line") is not None:
            s = line_scores(np.asarray(out["line"]["u"], float), np.asarray(out["line"]["v"], float), tr["u"], tr["v"], 1.0)
            if s:
                f1.append(s["f1"])
                if np.isfinite(s["rmse"]):
                    rmse.append(s["rmse"])
        pred = [g for g in out["grooves"] if g.get("unit", "mm") == "mm"]
        in_view = [g for g in tr["grooves"] if g["in_view"]]
        pairs, extra = match_grooves(pred, in_view)
        for t, p in pairs:
            if t["floor_visible_px"] < 3:                 # the floor cannot be seen: not scored
                continue
            n_meas += 1
            if p is None:
                miss += 1
            else:
                errs.append(p["depth"] - t["depth_mm"])
                true_d.append(t["depth_mm"])
                pred_d.append(p["depth"])
        false += len(extra)
        if in_view:
            true_pass = all(g["depth_mm"] >= LEGAL_MM for g in in_view)
            pred_pass = bool(pred) and all(g["depth"] >= LEGAL_MM for g in pred)
            v_ok += true_pass == pred_pass
            false_pass += (pred_pass and not true_pass)
    e = np.array(errs)
    n_v = sum(1 for it in items if any(g["in_view"] for g in it["truth"]["grooves"]))
    return {
        "n_frames": len(items), "n_grooves": n_meas,
        "line_f1": float(np.mean(f1)) if f1 else None, "line_rmse_px": float(np.mean(rmse)) if rmse else None,
        "depth_mae_mm": float(np.mean(np.abs(e))) if len(e) else None,
        "depth_p95_mm": float(np.percentile(np.abs(e), 95)) if len(e) else None,
        "depth_bias_mm": float(np.mean(e)) if len(e) else None,
        "miss_rate": miss / max(1, n_meas), "false_per_frame": false / len(items),
        "verdict_acc": v_ok / max(1, n_v), "false_pass_rate": false_pass / max(1, n_v),
        "ms_per_frame": ms, "_true_mm": [round(x, 3) for x in true_d], "_pred_mm": [round(x, 3) for x in pred_d],
    }


def eval_real(measure, items, uses_off=False):
    f1, derr, found = [], [], 0
    n_ref_depth = 0
    usable_hit = n_usable = false_meas = n_unusable = 0
    per = {}
    t0 = time.perf_counter()
    for it in items:
        out = measure(it["gray"], None, None)
        pred = out["grooves"]
        usable = out.get("usable", 1.0 if pred else 0.0)
        if it["usable"]:
            n_usable += 1
            usable_hit += usable >= 0.5
        else:
            n_unusable += 1
            false_meas += (usable >= 0.5 and bool(pred))
        ref = it["ref"]
        if ref is not None:
            if out.get("line") is not None:
                s = line_scores(np.asarray(out["line"]["u"], float), np.asarray(out["line"]["v"], float), ref["u"], ref["v"], 1.5)
                f1.append(s["f1"])
            if ref["mode"] == "groove":
                n_ref_depth += 1
                best = max(pred, key=lambda g: g["depth"]) if pred else None
                per[it["name"]] = {"ref_px": ref["depth_px"], "pred_px": None if best is None else round(best["depth"], 2)}
                if best is not None:
                    found += 1
                    derr.append(abs(best["depth"] - ref["depth_px"]))
    ms = (time.perf_counter() - t0) / len(items) * 1000
    pair = [per.get(f"photo_20261030_{k}.jpg", {}).get("pred_px") for k in ("154820", "154854")]
    return {
        "n_frames": len(items), "line_f1_ref": float(np.mean(f1)) if f1 else None,
        "depth_mae_px": float(np.mean(derr)) if derr else None, "found_ref": f"{found}/{n_ref_depth}",
        "repeat_pair_diff_px": abs(pair[0] - pair[1]) if None not in pair else None,
        "usable_recall": usable_hit / max(1, n_usable), "false_measure": false_meas / max(1, n_unusable),
        "ms_per_frame": ms, "_per_frame": per,
    }


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", help="e.g. algos.baseline_steger")
    ap.add_argument("--sets", nargs="*", default=["ideal", "stress", "legal", "real"])
    ap.add_argument("--limit", type=int, help="only the first N frames of each simulated set")
    ap.add_argument("--out")
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(args.module)
    _, measure = resolve(mod)
    cal = calib_mod.default_calib()
    res = {"name": getattr(mod, "NAME", args.module), "module": args.module, "uses_off": getattr(mod, "USES_OFF", False),
           "limit": args.limit, "sets": {}}
    for s in args.sets:
        if s == "real":
            res["sets"][s] = eval_real(measure, datasets.load_real(), res["uses_off"])
        else:
            res["sets"][s] = eval_sim(measure, datasets.load_sim(s), cal, args.limit)
    print(f"\n== {res['name']} ==")
    keys = ["line_f1", "line_rmse_px", "depth_mae_mm", "depth_p95_mm", "depth_bias_mm", "miss_rate", "false_per_frame",
            "verdict_acc", "false_pass_rate", "ms_per_frame"]
    print(f"{'set':8s} " + " ".join(f"{k[:13]:>13s}" for k in keys))
    for s, r in res["sets"].items():
        if s != "real":
            print(f"{s:8s} " + " ".join(f"{fmt(r[k]):>13s}" for k in keys))
    if "real" in res["sets"]:
        r = res["sets"]["real"]
        print("real     " + "  ".join(f"{k}={fmt(r[k])}" for k in
                                       ["line_f1_ref", "depth_mae_px", "found_ref", "repeat_pair_diff_px", "usable_recall",
                                        "false_measure", "ms_per_frame"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
