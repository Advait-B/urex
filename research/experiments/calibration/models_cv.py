"""models_cv.py - experiment 1: which model? 15 plates, leave-one-plate-out cross-validation.

For every model: fit residual, error on a plate it has NOT seen (interior plates = interpolation,
the two end plates = 2 mm of extrapolation) and the calibration-only depth error (common.depth_errors).
    python -m experiments.calibration.models_cv      -> out/models_cv.md, figs/models_cv.png
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algos import calib_models as cm
from .common import ALL15, FIGS, OUT, depth_errors, plate_set, rms, table

warnings.filterwarnings("ignore")
MODELS = ["poly11", "poly12", "poly22", "poly23", "poly43", "poly65", "poly87", "phys4", "phys5", "phys7", "spline", "gp"]
SEEDS = (0, 1, 2)

res = {m: {"fit": [], "cv_in": [], "cv_end": [], "depth": [], "depth_max": []} for m in MODELS}
for seed in SEEDS:
    u, v, z = plate_set(ALL15, seed=seed)
    for name in MODELS:
        if name == "gp" and seed > 0:               # the GP is slow: one seed only
            continue
        m = cm.make(name).fit(u, v, z)
        r = res[name]
        r["n"] = m.n_params
        r["fit"].append(rms(m.height(u, v) - z))
        e = depth_errors(m)
        r["depth"].append(rms(e)); r["depth_max"].append(float(np.abs(e).max()))
        cv_in, cv_end = [], []
        for zk in ALL15:
            tr = z != zk
            mk = cm.make(name).fit(u[tr], v[tr], z[tr])
            (cv_end if zk in (ALL15[0], ALL15[-1]) else cv_in).append(mk.height(u[~tr], v[~tr]) - zk)
        r["cv_in"].append(rms(np.concatenate(cv_in))); r["cv_end"].append(rms(np.concatenate(cv_end)))

rows = [[n, str(res[n]["n"])] + [float(np.mean(res[n][k])) for k in ("fit", "cv_in", "cv_end", "depth", "depth_max")] for n in MODELS]
md = table(rows, ["model", "free numbers", "fit RMS mm", "CV interior RMS mm", "CV end plates RMS mm", "depth err RMS mm", "depth err max mm"])
print(md)
open(f"{OUT}/models_cv.md", "w").write(md + "\n")
json.dump(res, open(f"{OUT}/models_cv.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
x = np.arange(len(MODELS))
for k, lab, c in (("fit", "fit residual", "#9aa5b1"), ("cv_in", "left-out interior plate", "#2a78b5"), ("cv_end", "left-out end plate (extrapolation)", "#d1603d")):
    ax.plot(x, [np.mean(res[n][k]) for n in MODELS], "o-", label=lab, color=c)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({res[n]['n']})" for n in MODELS], fontsize=8)
ax.set_ylabel("height error RMS (mm)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
ax.set_title("Calibration models, 15 plates: fit vs leave-one-plate-out (free numbers in brackets)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGS}/models_cv.png", dpi=130)
