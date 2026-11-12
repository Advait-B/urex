"""a_plates.py - experiment a: accuracy vs number of plates and vs the range they cover.

Depth error (calibration only, RMS over 5 depths x 71 columns, surface at +3.2 mm) averaged over 5
re-photographed plate sets. A polynomial of degree 3 in v needs >= 4 plates; with fewer the degree
in v is cut to (plates - 1), which is what a student would have to do.
    python -m experiments.calibration.a_plates       -> out/a_plates.md, figs/a_plates.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algos import calib_models as cm
from .common import ALL15, FIGS, OUT, depth_errors, plate_set, rms, table

warnings.filterwarnings("ignore")
COUNT = {15: ALL15, 8: [6, 2, -2, -6, -10, -14, -18, -22], 5: [6, -1, -8, -15, -22], 3: [6, -8, -22], 2: [6, -22]}
RANGE = {"+6..-22 (8)": COUNT[8], "+4..-18 (8, recommended)": [4, 2, 0, -3, -6, -10, -14, -18],
         "0..-20 (10, calibrate.py)": [0, -2, -4, -6, -8, -10, -12, -15, -18, -20],
         "0..-20 (6)": [0, -4, -8, -12, -16, -20], "+4..-8 (7)": [4, 2, 0, -2, -4, -6, -8]}
MODELS = ["poly23", "poly43", "phys4k", "phys5", "phys7", "spline", "gp"]
SEEDS = range(5)


def build(name, n_plates):
    if name.startswith("poly"):
        return cm.PolyModel(int(name[4]), min(int(name[5]), n_plates - 1))
    return cm.make(name)


def run(sets):
    out, cache = {}, {}
    for label, hts in sets.items():
        for name in MODELS:
            e = []
            for seed in (SEEDS if name != "gp" else [0]):
                u, v, z = plate_set(hts, seed=seed, cache=cache)
                e.append(depth_errors(build(name, len(hts)).fit(u, v, z)))
            e = np.array(e)
            out[label, name] = (rms(e), float(np.abs(e).max()), float(e[:, -1].mean()))   # RMS, max, bias at 16 mm
    return out


def md(res, sets, what):
    rows = [[str(lab)] + [f"{res[lab, n][0]:.3f} / {res[lab, n][1]:.3f}" for n in MODELS] for lab in sets]
    return table(rows, [what] + [f"{n} RMS / max" for n in MODELS])


rc, rr = run(COUNT), run(RANGE)
text = md(rc, COUNT, "plates (+6..-22)") + "\n\n" + md(rr, RANGE, "range (plates)")
print(text)
open(f"{OUT}/a_plates.md", "w").write(text + "\n")

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
col = dict(zip(MODELS, ["#7f8c9a", "#2a78b5", "#c9a227", "#d1603d", "#e88a6a", "#4f9d69", "#8e6bb0"]))
for n in MODELS:
    ax[0].plot(range(len(COUNT)), [rc[k, n][0] for k in COUNT], "o-", color=col[n], label=n)
ax[0].set_xticks(range(len(COUNT))); ax[0].set_xticklabels(list(COUNT)); ax[0].set_xlabel("number of plates, spread over +6 .. -22 mm")
ax[0].set_yscale("log"); ax[0].set_ylabel("depth error from calibration, RMS (mm)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
wd = 0.115
for i, n in enumerate(MODELS):
    ax[1].bar(np.arange(len(RANGE)) + (i - 3) * wd, [rr[k, n][0] for k in RANGE], wd, color=col[n], label=n)
ax[1].set_xticks(range(len(RANGE))); ax[1].set_xticklabels([k.replace(" (", "\n(") for k in RANGE], fontsize=7)
ax[1].set_yscale("log"); ax[1].grid(alpha=0.3, which="both", axis="y"); ax[1].set_xlabel("plate range (number of plates); tread surface is at +3.2 mm")
fig.suptitle("Experiment a: how many plates, and where", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIGS}/a_plates.png", dpi=130)
