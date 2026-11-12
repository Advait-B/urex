"""timing.py - fair timing on a shared machine: the algorithms take turns on the same 40 TRAIN frames, 5 rounds,
and the BEST round of each is reported (the least disturbed one), plus the ratio to the baseline.
    PYTHONPATH=. python experiments/dp-tracking/timing.py
"""
import importlib, time, warnings
import numpy as np
from bench import datasets

warnings.simplefilter("ignore")
names = ["baseline_steger", "dp_dense", "dp_dense_rt", "dp_smooth", "dp_cand", "dp_graph"]
mods = [importlib.import_module("algos." + n) for n in names]
frames = [it["on"] for it in datasets.load_sim("train")[:40]]
best = {n: 1e9 for n in names}
for _ in range(5):
    for n, m in zip(names, mods):
        t = time.perf_counter()
        for g in frames:
            m.extract_line(g)
        best[n] = min(best[n], (time.perf_counter() - t) / len(frames) * 1000)
for n in names:
    print(f"{n:16s} {best[n]:6.1f} ms/frame   x{best[n] / best['baseline_steger']:.2f}")
