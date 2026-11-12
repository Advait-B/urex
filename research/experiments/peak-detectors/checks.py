"""checks.py - three small side experiments (run from research/):  python -m experiments.peak-detectors.checks
  A. signed error of the on-line points (is there a systematic offset?) on the first 60 'ideal' frames
  B. how much the default depth stage moves when the SAME true line is sub-sampled with a different phase
  C. CPU time per stage (time.process_time, so other jobs on the machine do not count), 320x240 frames
"""
import importlib
import time
import warnings

import numpy as np
from scipy.spatial import cKDTree

from algos import peakcommon as pc
from bench import calib as calib_mod
from bench import datasets, depth, run

warnings.simplefilter("ignore")
cal = calib_mod.default_calib()
ideal = datasets.load_sim("ideal")

print("A. signed v error (px, + = detector reads lower in the image / deeper), points within 1 px of the truth")
for a in ["baseline_steger", "pk_steger_s1", "pk_cog", "pk_gauss3", "pk_blais8", "pk_fir", "pk_fir2d"]:
    m = importlib.import_module("algos." + a)
    dv = []
    for it in ideal[:60]:
        t = it["truth"]
        p = m.extract_line(it["on"])
        d, j = cKDTree(np.c_[t["u"], t["v"]]).query(np.c_[p["u"], p["v"]])
        dv.append((np.asarray(p["v"]) - t["v"][j])[d < 1.0])
    dv = np.concatenate(dv)
    print(f"   {a:16s} mean={dv.mean():+.3f} std={dv.std():.3f} rms={np.sqrt((dv ** 2).mean()):.3f} n={len(dv)}")

print("B. oracle line (truth[k::3]) -> default depth stage, 'stress' set")
stress = datasets.load_sim("stress")
truth = {id(it["on"]): it["truth"] for it in stress}
for k in range(3):
    def measure(gray, c=None, off=None, k=k):
        t = truth[id(gray)]
        return depth.measure_from_line(t["u"][k::3], t["v"][k::3], c)
    r = run.eval_sim(measure, stress, cal)
    print(f"   phase {k}: depth_mae_mm={r['depth_mae_mm']:.3f} p95={r['depth_p95_mm']:.3f} miss={r['miss_rate']:.3f} "
          f"false={r['false_per_frame']:.3f} verdict={r['verdict_acc']:.3f}")

print("C. CPU ms per 320x240 frame (mean of 60 ideal frames)")
frames = [it["on"] for it in ideal[:60]]


def cpu(fn):
    t0 = time.process_time()
    out = [fn(g) for g in frames]
    return (time.process_time() - t0) / len(frames) * 1000, out


for bg in ("tophat", "tophat1d", "colpct"):
    print(f"   background {bg:9s} {cpu(lambda g: pc.remove_background(g, bg))[0]:.2f}")
_, imgs = cpu(lambda g: pc.remove_background(g, "tophat"))
it_ = iter(imgs)
print(f"   smooth sigma_v=1.5           {cpu(lambda g: pc.smooth(next(it_), 1.5, 0))[0]:.2f}")
it_ = iter(imgs)
print(f"   smooth sigma_v=1.5 + u=1.5   {cpu(lambda g: pc.smooth(next(it_), 1.5, 1.5))[0]:.2f}")
for a in ["pk_cog", "pk_gauss3", "pk_blais8", "pk_fir", "pk_fir2d", "pk_fir2d_rows"]:
    cfg = importlib.import_module("algos." + a).CFG
    print(f"   {a:14s} no chain {cpu(lambda g: pc.extract(g, {**cfg, 'chain': False}))[0]:.2f}   "
          f"with chain {cpu(lambda g: pc.extract(g, cfg))[0]:.2f}")
for a in ["baseline_steger", "pk_steger_s1"]:
    m = importlib.import_module("algos." + a)
    sig = getattr(m, "SIGMA", 1.5)
    print(f"   {a:14s} steger_points only {cpu(lambda g: pc.sf.steger_points(np.ascontiguousarray(g.T), sig))[0]:.2f}   "
          f"full {cpu(lambda g: m.extract_line(g))[0]:.2f}")
