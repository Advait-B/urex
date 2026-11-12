"""b_extrapolation.py - experiment b: calibrate on 0 .. -12 mm only, then measure a 16 mm groove.

Shims under the feet can only push the plate AWAY (heights <= 0). On the tyre the tread surface sits
~3.2 mm ABOVE the feet plane and a 16 mm groove floor at -12.8 mm, so both ends are extrapolated.
    python -m experiments.calibration.b_extrapolation   -> out/b_extrapolation.md, figs/b_extrapolation.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algos import calib_models as cm
from .common import FIGS, OUT, UCOLS, depth_errors, plate_set, table, true_line

warnings.filterwarnings("ignore")
PLATES = [0, -2, -4, -6, -8, -10, -12]
MODELS = ["poly23", "poly43", "phys5", "phys7", "spline", "gp"]
ZS = np.arange(8.0, -22.01, -1.0)
SEEDS = range(5)

herr = {n: [] for n in MODELS}            # |height error| vs true height
derr = {n: [] for n in MODELS}            # depth error of 16 mm (surface +3.2) and 8 mm grooves
cache = {}
for name in MODELS:
    for seed in (SEEDS if name != "gp" else [0, 1]):
        m = cm.make(name).fit(*plate_set(PLATES, seed=seed, cache=cache))
        herr[name].append([m.height(UCOLS, true_line(z)) - z for z in ZS])
        derr[name].append(depth_errors(m, depths=(8.0, 16.0)))
rows = []
for n in MODELS:
    d, h = np.array(derr[n]), np.array(herr[n])
    rows.append([n, float(d[:, 0].mean()), float(np.abs(d[:, 0]).max()), float(d[:, 1].mean()), float(np.sqrt((d[:, 1] ** 2).mean())),
                 float(np.abs(d[:, 1]).max()), float(np.sqrt((h[:, ZS == 3.0] ** 2).mean())), float(np.sqrt((h[:, ZS == -18.0] ** 2).mean()))])
md = table(rows, ["model (plates 0..-12)", "8 mm groove: bias", "max", "16 mm groove: bias", "RMS", "max", "height RMS at +3 mm", "height RMS at -18 mm"])
print(md)
open(f"{OUT}/b_extrapolation.md", "w").write(md + "\n")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
col = dict(zip(MODELS, ["#7f8c9a", "#2a78b5", "#d1603d", "#e8a33d", "#4f9d69", "#8e6bb0"]))
for n in MODELS:
    ax.plot(ZS, np.sqrt((np.array(herr[n]) ** 2).mean(axis=(0, 2))), "o-", ms=3, color=col[n], label=n)
ax.axvspan(-12, 0, color="#dfe9f3", label="calibrated range (shims 0 .. -12)")
ax.axvline(3.2, color="k", ls=":", lw=1); ax.text(3.3, 2.0, "tread\nsurface", fontsize=7)
ax.axvline(-12.8, color="k", ls=":", lw=1); ax.text(-12.7, 2.0, "floor of a\n16 mm groove", fontsize=7, ha="right")
ax.set_yscale("log"); ax.set_ylim(5e-4, 5); ax.invert_xaxis(); ax.grid(alpha=0.3, which="both")
ax.set_xlabel("true height above the feet plane (mm)"); ax.set_ylabel("height error RMS over columns (mm)")
ax.set_title("Experiment b: error outside the calibrated range", fontsize=10); ax.legend(fontsize=7, ncol=2, loc="upper center")
fig.tight_layout(); fig.savefig(f"{FIGS}/b_extrapolation.png", dpi=130)
