"""tune.py - score peak-detector configurations on a slice of the TRAIN set (never the test sets).

    python -m experiments.peak-detectors.tune            (run from research/)
Prints one row per configuration: line F1, line RMSE, depth MAE / P95, miss, false, ms.
"""
import importlib
import sys

import warnings

import numpy as np

from algos import peakcommon as pc
from bench import calib as calib_mod
from bench import datasets, depth, run


def score(cfg, items, cal):
    def measure(gray, c=None, off=None):
        p = pc.extract(gray, cfg)
        out = depth.measure_from_line(p["u"], p["v"], c)
        out["line"] = p
        return out
    return run.eval_sim(measure, items, cal)


def show(tag, r):
    med = float(np.median(np.abs(np.array(r["_pred_mm"]) - np.array(r["_true_mm"]))))
    print(f"{tag:46s} med={med:.3f} f1={r['line_f1']:.3f} rmse={r['line_rmse_px']:.3f} mae={r['depth_mae_mm']:.3f} "
          f"p95={r['depth_p95_mm']:.3f} miss={r['miss_rate']:.3f} false={r['false_per_frame']:.3f} "
          f"ms={r['ms_per_frame']:.1f}", flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    grid = importlib.import_module("experiments.peak-detectors." + sys.argv[1]).GRID
    n0, n1 = (int(a) for a in sys.argv[2:4]) if len(sys.argv) > 3 else (0, 300)
    items = datasets.load_sim("train")[n0:n1]
    cal = calib_mod.default_calib()
    for tag, cfg in grid:
        show(tag, score(cfg, items, cal))
