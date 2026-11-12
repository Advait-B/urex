"""c_shim_errors.py - experiment c: the shims themselves are wrong. How far does each model carry it?

8 plates (+6 .. -22 mm, every 4 mm). The photos are of the TRUE heights; the student writes down a
wrong number, or a plate is not parallel to the feet.
    python -m experiments.calibration.c_shim_errors   -> out/c_shim_errors.md
"""
import numpy as np

from algos import calib_models as cm
from .common import OUT, depth_errors, plate_points, plate_set, rms, table

PLATES = [6, 2, -2, -6, -10, -14, -18, -22]
MODELS = ["poly23", "poly43", "phys5", "phys7", "spline"]
rng = np.random.default_rng(7)
cache = {}
base = [plate_points(z) for z in PLATES]


def fit_all(points, labels):
    u = np.concatenate([p[0] for p in points]); v = np.concatenate([p[1] for p in points])
    z = np.concatenate([np.full(len(p[0]), float(l)) for p, l in zip(points, labels)])
    return {n: depth_errors(cm.make(n).fit(u, v, z)) for n in MODELS}


def swap(i, **tilt):
    pts = list(base)
    pts[i] = plate_points(PLATES[i], **tilt)
    return pts


P = np.array(PLATES, float)
CASES = {
    "none": [(base, P)],
    "calipers: every label +-0.05 mm (sd, 20 trials)": [(base, P + rng.normal(0, 0.05, len(P))) for _ in range(20)],
    "one stack 0.2 mm off (the -10 plate)": [(base, P - 0.2 * (P == -10))],
    "one stack 0.2 mm off (the -22 end plate)": [(base, P - 0.2 * (P == -22))],
    "base shim 0.2 mm off (every plate below -2)": [(base, P - 0.2 * (P < -2))],
    "all labels 1 % too big (calipers / units)": [(base, P * 1.01)],
    "the -10 plate tilted 0.5 deg along Y": [(swap(4, tilt_x=0.5), P)],
    "the -10 plate tilted 0.5 deg along X": [(swap(4, tilt_y=0.5), P)],
    "every plate tilted 0.5 deg, random direction (10 trials)": [],
}
for t in range(10):
    a = rng.uniform(0, 2 * np.pi, len(P))
    CASES["every plate tilted 0.5 deg, random direction (10 trials)"].append(
        ([plate_points(z, seed=10 + t, tilt_x=0.5 * np.sin(k), tilt_y=0.5 * np.cos(k)) for z, k in zip(PLATES, a)], P))

rows = []
for label, trials in CASES.items():
    E = [fit_all(p, l) for p, l in trials]
    rows.append([label] + [f"{rms([e[n] for e in E]):.3f} / {max(np.abs(e[n]).max() for e in E):.3f} / {np.mean([e[n][-1].mean() for e in E]):+.3f}"
                           for n in MODELS])
md = table(rows, ["error in the shims"] + [f"{n}: RMS / max / bias@16" for n in MODELS])
print(md)
open(f"{OUT}/c_shim_errors.md", "w").write(md + "\n")
