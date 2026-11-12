"""run2.py - scorer v2. Same algorithm API as bench/run.py; fixes what the first round of
research branches found wrong or missing in v1, and adds two test sets (bench/datasets2.py).

    python -m bench.run2 algos.baseline_steger --out results2/baseline_steger.json

What changed against run.py (v1 results stay valid for what they measured):
  * ABSTAINING is allowed. If measure() returns "usable" < 0.5 the frame is unanswered:
    it is not counted as missed grooves or a wrong verdict. `coverage` = share of frames
    answered; depth and verdict metrics are over answered frames; `false_pass_rate` is
    still over ALL frames (an unsafe PASS is unsafe however often you abstain).
  * SPLIT grooves: a second predicted piece on the same true groove is counted in
    `split_per_frame`, not as a false groove. `false_per_frame` = predictions on no groove.
  * `depth_med_mm` (median |error|) and `gross_rate` (share of grooves wrong by > 1 mm, or
    > 3 px on handheld) next to MAE / P95, because means were set by a few huge outliers.
  * `cpu_ms` is process CPU time, not wall-clock (the machine is shared).
  * New sets: `shifted` (other holder units, each with its own calibration, mm) and
    `handheld` (looks like the real photos; scored in px against the true line shift).
  * Real photos: line agreement and depth are scored on the 6 truck-tyre frames only (the
    rover references contain background clutter). Depth is read from the predicted groove
    that overlaps the reference groove, and `ok_ref` counts readings within 3 px.
    `false_measure` is split: frames with no laser at all / laser visible but unusable.
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

from . import calib as calib_mod
from . import datasets, datasets2
from .run import LEGAL_MM, fmt, line_scores, resolve


def assign(pred, truth, pad=4.0):
    """Each prediction goes to the true groove it overlaps most. Returns
    (per-truth list of predictions sorted by overlap, predictions on no groove)."""
    per, none = [[] for _ in truth], []
    for p in pred:
        best, best_ov = None, 0.0
        for i, t in enumerate(truth):
            ov = min(p["u1"], t["u1"] + pad) - max(p["u0"], t["u0"] - pad)
            if ov > best_ov:
                best, best_ov = i, ov
        if best is None:
            none.append(p)
        else:
            per[best].append((best_ov, p))
    return [[p for _, p in sorted(lst, key=lambda x: -x[0])] for lst in per], none


def eval_sim2(measure, items, default_cal, unit="mm", limit=None):
    items = items[:limit] if limit else items
    px = unit == "px"
    f1, rmse, errs, true_d, pred_d = [], [], [], [], []
    n_meas = miss = false = split = answered = 0
    v_ok = v_n = false_pass = n_view = 0
    cpu = 0.0
    for it in items:
        tr = it["truth"]
        cal = None if px else (calib_mod.Calib(**it["calib"]) if "calib" in it else default_cal)
        t0 = time.process_time()
        out = measure(it["on"], cal, it["off"])
        cpu += time.process_time() - t0
        in_view = [g for g in tr["grooves"] if g["in_view"]]
        key = "shift_px" if px else "depth_mm"
        pred = [g for g in out["grooves"] if g.get("unit", "mm") == unit]
        true_pass = bool(in_view) and all(g["depth_mm"] >= LEGAL_MM for g in in_view)
        pred_pass = bool(pred) and all(g["depth"] >= LEGAL_MM for g in pred)
        is_answered = out.get("usable", 1.0) >= 0.5
        if in_view and not px:
            n_view += 1
            false_pass += (is_answered and pred_pass and not true_pass)
        if not is_answered:
            continue
        answered += 1
        if out.get("line") is not None:
            s = line_scores(np.asarray(out["line"]["u"], float), np.asarray(out["line"]["v"], float), tr["u"], tr["v"], 1.0)
            if s:
                f1.append(s["f1"])
                if np.isfinite(s["rmse"]):
                    rmse.append(s["rmse"])
        per, none = assign(pred, in_view)
        false += len(none)
        for t, ps in zip(in_view, per):
            measurable = (t.get("shift_px") or 0) > 3 if px else t["floor_visible_px"] >= 3
            if not measurable:
                continue
            n_meas += 1
            if not ps:
                miss += 1
                continue
            split += len(ps) - 1
            errs.append(ps[0]["depth"] - t[key])
            true_d.append(t[key])
            pred_d.append(ps[0]["depth"])
        if in_view and not px:
            v_n += 1
            v_ok += true_pass == pred_pass
    e = np.array(errs)
    a = np.abs(e)
    gross = 3.0 if px else 1.0
    sfx = "px" if px else "mm"
    return {
        "n_frames": len(items), "coverage": answered / len(items), "n_grooves": n_meas, "unit": unit,
        "line_f1": float(np.mean(f1)) if f1 else None, "line_rmse_px": float(np.mean(rmse)) if rmse else None,
        f"depth_mae_{sfx}": float(a.mean()) if len(e) else None, f"depth_med_{sfx}": float(np.median(a)) if len(e) else None,
        f"depth_p95_{sfx}": float(np.percentile(a, 95)) if len(e) else None, f"depth_bias_{sfx}": float(e.mean()) if len(e) else None,
        "gross_rate": float(np.mean(a > gross)) if len(e) else None,
        "miss_rate": miss / max(1, n_meas), "false_per_frame": false / max(1, answered), "split_per_frame": split / max(1, answered),
        "verdict_acc": None if px else v_ok / max(1, v_n), "false_pass_rate": None if px else false_pass / max(1, n_view),
        "cpu_ms": cpu / len(items) * 1000, "_true": [round(x, 3) for x in true_d], "_pred": [round(x, 3) for x in pred_d],
    }


def eval_real2(measure, items):
    f1, derr, per = [], [], {}
    found = ok = n_truck = 0
    hit = n_usable = fm_nolaser = n_nolaser = fm_laser = n_laser = 0
    cpu = 0.0
    for it in items:
        t0 = time.process_time()
        out = measure(it["gray"], None, None)
        cpu += time.process_time() - t0
        pred = [g for g in out["grooves"] if g.get("unit", "px") == "px"]
        usable = out.get("usable", 1.0 if pred else 0.0)
        reports = usable >= 0.5 and bool(pred)
        if it["usable"]:
            n_usable += 1
            hit += usable >= 0.5
        elif it["laser"]:
            n_laser += 1
            fm_laser += reports
        else:
            n_nolaser += 1
            fm_nolaser += reports
        ref = it["ref"]
        if ref is None or ref["mode"] != "groove":
            continue
        n_truck += 1
        if out.get("line") is not None:
            f1.append(line_scores(np.asarray(out["line"]["u"], float), np.asarray(out["line"]["v"], float),
                                  ref["u"], ref["v"], 1.5)["f1"])
        u0, u1 = ref["groove_u"]
        cands = [(min(g["u1"], u1 + 4) - max(g["u0"], u0 - 4), g) for g in pred]
        cands = [c for c in cands if c[0] > 0]
        best = max(cands, key=lambda c: c[0])[1] if (cands and usable >= 0.5) else None
        per[it["name"]] = {"ref_px": ref["depth_px"], "pred_px": None if best is None else round(best["depth"], 2)}
        if best is not None:
            found += 1
            derr.append(abs(best["depth"] - ref["depth_px"]))
            ok += derr[-1] <= 3.0
    pair = [per.get(f"photo_20261030_{k}.jpg", {}).get("pred_px") for k in ("154820", "154854")]
    return {
        "n_frames": len(items), "line_f1_truck": float(np.mean(f1)) if f1 else None,
        "depth_mae_px": float(np.mean(derr)) if derr else None, "found_ref": f"{found}/{n_truck}", "ok_ref": f"{ok}/{n_truck}",
        "repeat_pair_diff_px": abs(pair[0] - pair[1]) if None not in pair else None,
        "usable_recall": hit / max(1, n_usable),
        "false_measure_nolaser": fm_nolaser / max(1, n_nolaser), "false_measure_laser": fm_laser / max(1, n_laser),
        "cpu_ms": cpu / len(items) * 1000, "_per_frame": per,
    }


SIM_KEYS = ["coverage", "line_f1", "line_rmse_px", "depth_mae", "depth_med", "depth_p95", "gross_rate", "miss_rate",
            "false_per_frame", "split_per_frame", "verdict_acc", "false_pass_rate", "cpu_ms"]


def score(module_name, sets, limit=None):
    mod = importlib.import_module(module_name)
    _, measure = resolve(mod)
    cal = calib_mod.default_calib()
    res = {"name": getattr(mod, "NAME", module_name), "module": module_name, "uses_off": getattr(mod, "USES_OFF", False),
           "scorer": 2, "limit": limit, "sets": {}}
    for s in sets:
        if s == "real":
            res["sets"][s] = eval_real2(measure, datasets.load_real())
        elif s == "handheld":
            res["sets"][s] = eval_sim2(measure, datasets2.load_sim2(s), None, "px", limit)
        elif s == "shifted":
            res["sets"][s] = eval_sim2(measure, datasets2.load_sim2(s), cal, "mm", limit)
        else:
            res["sets"][s] = eval_sim2(measure, datasets.load_sim(s), cal, "mm", limit)
    return res


def show(res):
    print(f"\n== {res['name']} (scorer v2) ==")
    print(f"{'set':9s} " + " ".join(f"{k[:11]:>11s}" for k in SIM_KEYS))
    for s, r in res["sets"].items():
        if s == "real":
            continue
        sfx = "_px" if r["unit"] == "px" else "_mm"
        vals = [r.get(k if k in r else k + sfx) for k in SIM_KEYS]
        print(f"{s:9s} " + " ".join(f"{fmt(v):>11s}" for v in vals) + ("   [px]" if r["unit"] == "px" else ""))
    if "real" in res["sets"]:
        r = res["sets"]["real"]
        print("real      " + "  ".join(f"{k}={fmt(r[k])}" for k in ["line_f1_truck", "depth_mae_px", "found_ref", "ok_ref",
                                                                   "repeat_pair_diff_px", "usable_recall",
                                                                   "false_measure_nolaser", "false_measure_laser", "cpu_ms"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--sets", nargs="*", default=["ideal", "stress", "legal", "shifted", "handheld", "real"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    res = score(args.module, args.sets, args.limit)
    show(res)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
