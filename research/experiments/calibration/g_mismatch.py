"""g_mismatch.py - is the physical model only good because the simulator IS the physical model?

The simulator's camera is exactly "pinhole + k1, principal point at the image centre" and its laser is
an exact plane, which is precisely what PhysModel assumes. Here the simulated lens is made less ideal
(without editing sim/: holder_sim.project is wrapped) and every model is calibrated again.
    python -m experiments.calibration.g_mismatch     -> out/g_mismatch.md
"""
import numpy as np

from algos import calib_models as cm
from sim import holder_sim as hs
from . import common
from .common import ALL15, OUT, depth_errors, device, plate_set, rms, table

LENS = {"k2": 0.0, "p1": 0.0, "p2": 0.0, "dcx": 0.0, "dcy": 0.0}


def project(P, width, height_px, dev=hs.DEVICE):
    f = hs.focal_px(width, dev)
    zc = dev["Zc"] - P[..., 2]
    x, y = P[..., 0] / zc, -P[..., 1] / zc
    r2 = x * x + y * y
    k = 1 + dev["k1"] * r2 + LENS["k2"] * r2 * r2
    xd = x * k + 2 * LENS["p1"] * x * y + LENS["p2"] * (r2 + 2 * x * x)
    yd = y * k + LENS["p1"] * (r2 + 2 * y * y) + 2 * LENS["p2"] * x * y
    return (width - 1) / 2 + LENS["dcx"] + f * xd, (height_px - 1) / 2 + LENS["dcy"] + f * yd


hs.project = project
CASES = {"simulator lens (k1 only, centred)": ({}, {}),
         "focal length 5 % longer than the data sheet": ({}, {"f_mm": 2.7 * 1.05}),
         "principal point off by (6, -4) px": ({"dcx": 6.0, "dcy": -4.0}, {}),
         "+ second radial term k2 = +0.05": ({"k2": 0.05}, {}),
         "+ tangential distortion p1 = p2 = 0.003": ({"p1": 0.003, "p2": 0.003}, {}),
         "all of the above": ({"dcx": 6.0, "dcy": -4.0, "k2": 0.05, "p1": 0.003, "p2": 0.003}, {"f_mm": 2.7 * 1.05})}
MODELS = ["poly23", "poly43", "poly65", "phys5", "phys7", "spline"]
SETS = {"15": ALL15, "8": [4, 2, 0, -3, -6, -10, -14, -18]}
rows = []
for label, (lens, dv) in CASES.items():
    LENS.update({"k2": 0.0, "p1": 0.0, "p2": 0.0, "dcx": 0.0, "dcy": 0.0}); LENS.update(lens)
    common._LINES.clear()
    dev = device(**dv)
    for sn, hts in SETS.items():
        u, v, z = plate_set(hts, dev)
        E = {n: depth_errors(cm.make(n).fit(u, v, z), dev) for n in MODELS}
        rows.append([label, sn] + [f"{rms(E[n]):.3f} / {np.abs(E[n]).max():.3f}" for n in MODELS])
md = table(rows, ["simulated lens", "plates"] + [f"{n} RMS / max" for n in MODELS])
print(md)
open(f"{OUT}/g_mismatch.md", "w").write(md + "\n")
