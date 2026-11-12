"""e_end_to_end.py - experiment e: what the calibration changes in the benchmark.

  1. TRUE line + each calibration / depth-stage variant, scored by run.eval_sim. No line noise, so this
     is the ceiling ("oracle") for each variant and shows the calibration's own contribution.
  2. the harness results of the algos/calib_*.py variants (baseline Steger line), read from results/.
     Make them first:  python -m bench.run algos.<name> --out results/<name>.json
    python -m experiments.calibration.e_end_to_end    -> out/e_end_to_end.md
"""
import json
import os

import numpy as np

from algos import calib_fit
from bench import calib, datasets, run
from .common import OUT, RESEARCH, table

GEO = dict(mm_space=True, curvature=True)
VARIANTS = {            # name: (calibration, depth-stage flags)
    "default calibration (poly43, 15 plates) = oracle_line": (calib.default_calib(), {}),
    "calib_phys5": (calib_fit.load("phys5_15plates", "phys5"), {}),
    "calib_shims_poly23": (calib_fit.load("poly23_shims", "poly23", calib_fit.SHIMS), {}),
    "calib_shims_phys5": (calib_fit.load("phys5_shims", "phys5", calib_fit.SHIMS), {}),
    "calib_default_geo": (calib.default_calib(), GEO),
    "calib_phys5_geo": (calib_fit.load("phys5_15plates", "phys5"), GEO),
    "calib_rec_phys5_geo": (calib_fit.load("phys5_recommended", "phys5", calib_fit.RECOMMENDED), GEO),
}
SETS = ("ideal", "stress", "legal")
data = {s: datasets.load_sim(s) for s in SETS}
rows = []
for name, (cal, flags) in VARIANTS.items():
    cells = [name]
    for s in SETS:
        truth = {id(it["on"]): it["truth"] for it in data[s]}
        r = run.eval_sim(lambda g, c=None, off=None: calib_fit.measure_line(
            truth[id(g)]["u"][::3].astype(float), truth[id(g)]["v"][::3].astype(float), cal, **flags), data[s], None)
        cells.append(f"{r['depth_mae_mm']:.3f} / {r['depth_p95_mm']:.3f} / {r['depth_bias_mm']:+.3f}")
    rows.append(cells)
md = "**1. TRUE line + each variant: depth MAE / P95 / bias (mm)**\n\n" + table(rows, ["variant"] + list(SETS))

rows = []
for name in ["baseline_steger"] + list(VARIANTS)[1:]:
    path = os.path.join(RESEARCH, "results", f"{name}.json")
    if not os.path.exists(path):
        continue
    r = json.load(open(path))["sets"]
    rows.append([name] + [f"{r[s]['depth_mae_mm']:.3f} / {r[s]['depth_p95_mm']:.3f} / {r[s]['depth_bias_mm']:+.3f} / {r[s]['verdict_acc']:.3f}" for s in SETS]
                + [f"{r['real']['depth_mae_px']:.2f} px, {r['real']['usable_recall']:.2f}, {r['real']['false_measure']:.2f}" if "real" in r else "-"])
md += "\n\n**2. Harness results (baseline Steger line): depth MAE / P95 / bias (mm) / verdict accuracy**\n\n" + table(
    rows, ["algorithm"] + list(SETS) + ["real: depth MAE, usable recall, false measure"])
print(md)
open(f"{OUT}/e_end_to_end.md", "w").write(md + "\n")
