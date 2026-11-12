"""sweep.py - one-factor-at-a-time degradation curves for the peak detectors.

    python -m experiments.peak-detectors.sweep        (from research/; writes sweep.json next to this file)

Base scene: an 'ideal' scene with line_peak_dn 180, blur_px 0.8, speckle 0.2, line_w_mm 0.55,
ambient 0.15. One factor is changed at a time, N scenes (random grooves) per level; the SAME
scenes are used for every level and every detector. Scores: bench.run.line_scores at 1 px.
"""
import importlib
import json
import os
import sys
import time

import numpy as np

from bench import run

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sim"))
import holder_sim as hs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = {"line_peak_dn": 180.0, "blur_px": 0.8, "speckle": 0.2, "line_w_mm": 0.55, "ambient": 0.15}
SWEEPS = {
    "line_peak_dn": [40, 60, 90, 130, 180, 250, 350, 450, 600],
    "blur_px": [0.6, 1.0, 1.4, 1.8, 2.2, 2.5],
    "speckle": [0.1, 0.2, 0.3, 0.38, 0.45],
    "ambient": [0.1, 0.3, 0.6, 1.0],
}
ALGOS = ["baseline_steger", "pk_steger_s1", "pk_cog", "pk_gauss3", "pk_blais8", "pk_fir", "pk_fir2d"]
N = 16

if __name__ == "__main__":
    mods = {a: importlib.import_module("algos." + a) for a in ALGOS}
    rng = np.random.default_rng(7)
    scenes = [hs.sample_scene(rng, "ideal") for _ in range(N)]
    out = {"base": BASE, "n_scenes": N, "sweeps": {}}
    for factor, levels in SWEEPS.items():
        rows = []
        for lv in levels:
            acc = {a: {"f1": [], "rmse": [], "sat": []} for a in ALGOS}
            for s in scenes:
                sc = {**s, **BASE, factor: float(lv)}
                on, _, tr = hs.render(sc, return_off=False)
                for a, m in mods.items():
                    p = m.extract_line(on)
                    r = run.line_scores(np.asarray(p["u"], float), np.asarray(p["v"], float), tr["u"], tr["v"], 1.0)
                    acc[a]["f1"].append(r["f1"])
                    acc[a]["rmse"].append(r["rmse"])
                    acc[a]["sat"].append(float((on[tr["mask"]] >= 250).mean()) if tr["mask"].any() else 0.0)
            row = {"level": lv, "sat_frac": float(np.mean(acc[ALGOS[0]]["sat"]))}
            for a in ALGOS:
                row[a] = {"f1": float(np.mean(acc[a]["f1"])), "rmse": float(np.nanmean(acc[a]["rmse"]))}
            rows.append(row)
            print(factor, lv, f"sat={row['sat_frac']:.2f}", " ".join(f"{a[3:] if a.startswith('pk_') else a}={row[a]['rmse']:.3f}/{row[a]['f1']:.2f}" for a in ALGOS), flush=True)
        out["sweeps"][factor] = rows
    with open(os.path.join(HERE, "sweep.json"), "w") as fh:
        json.dump(out, fh, indent=1)
