"""figs.py - the figures in notes/figs/dp-tracking/.   PYTHONPATH=. python experiments/dp-tracking/figs.py cache.pkl"""
import pickle, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench import calib, datasets, depth
from algos import baseline_steger, dp_dense, dp_dense_rt

OUT = "notes/figs/dp-tracking/"
cal = calib.default_calib()


def panel(a, it, mod, title, zoom=None):
    g = it["on"]
    p = mod.extract_line(g)
    out = depth.measure_from_line(p["u"], p["v"], cal)
    a.imshow(g, cmap="gray_r", vmin=0, vmax=160)                       # inverted: light background
    a.plot(it["truth"]["u"], it["truth"]["v"], "-", c="#1f9bd1", lw=3.5, alpha=0.45, label="true line")
    a.plot(p["u"], p["v"], ".", c="#d62728", ms=2.5, label="kept points")
    for q in out["grooves"]:
        a.axvspan(q["u0"], q["u1"], color="#f2c200", alpha=0.25, lw=0)
        a.text((q["u0"] + q["u1"]) / 2, a.get_ylim()[1] + 4 if zoom is None else zoom[1] + 4, f"{q['depth']:.1f}", ha="center", va="top", fontsize=8, color="#7a5f00")
    for q in it["truth"]["grooves"]:
        a.text((q["u0"] + q["u1"]) / 2, (zoom[0] if zoom else 239) - 14, f"true {q['depth_mm']:.1f}" + ("" if q["floor_visible_px"] >= 3 else "*"), ha="center", fontsize=8, color="#0b5c82")
    if zoom: a.set_ylim(*zoom)
    a.set_aspect("auto")
    a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])


data = pickle.load(open(sys.argv[1], "rb"))
# figure 1: baseline vs dp on four stress factors (same clean scene underneath)
rows = [("12 glints", 3), ("2 debris gaps", 3), ("saturated (600 DN)", 3), ("gaps + glints", 3)]
if len(sys.argv) > 2: rows = [(r, int(sys.argv[2])) for r, _ in rows]
fig, ax = plt.subplots(len(rows), 2, figsize=(11, 2.5 * len(rows) + 0.4))
for (name, i), (a0, a1) in zip(rows, ax):
    it = data[name][i]
    lo, hi = it["truth"]["v"].min() - 45, it["truth"]["v"].max() + 30
    panel(a0, it, baseline_steger, f"{name}: baseline_steger", (hi, lo))
    panel(a1, it, dp_dense, f"{name}: dp_dense", (hi, lo))
ax[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("Yellow band + number on top = reported groove (mm); 'true' = truth (* = floor hidden, not scored)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "factors_baseline_vs_dp.png", dpi=110, facecolor="white"); plt.close(fig)


# figure 2: where dp_dense still fails (frames of the `stress` TEST set, picked after scoring: analysis only)
stress = datasets.load_sim("stress")
fails = [(7, "line hidden at both ends, glints there: the path hops onto them (107.9 mm!)"),
         (20, "paint stripe just above a weak line: the stripe's edge out-scores the laser"),
         (87, "two debris gaps, each with a glint nearby")]
fig, ax = plt.subplots(len(fails), 2, figsize=(11, 3.3 * len(fails)))
for (i, why), (a0, a1) in zip(fails, ax):
    it = stress[i]
    panel(a0, it, baseline_steger, f"stress[{i}] baseline_steger", (239, 0))
    panel(a1, it, dp_dense, f"stress[{i}] dp_dense", (239, 0))
    a1.text(160, 232, why, ha="center", fontsize=8, color="#7a1010", bbox=dict(fc="white", ec="none", alpha=0.8))
ax[0, 0].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "stress_failures.png", dpi=110, facecolor="white"); plt.close(fig)

# figure 3: two real truck-tyre photos WITHOUT a reference where the two methods disagree
real = {it["name"][6:-4]: it for it in datasets.load_real()}
fig, ax = plt.subplots(2, 2, figsize=(8.5, 10.5))
for (name, (a0, a1)) in zip(("20261030_154700", "20261030_154748"), ax):
    g = real[name]["gray"]
    for a, mod in ((a0, baseline_steger), (a1, dp_dense)):
        p = mod.extract_line(g)
        out = depth.measure_from_line(p["u"], p["v"])
        a.imshow(g, cmap="gray_r", vmin=0, vmax=255)
        a.plot(p["u"], p["v"], ".", c="#d62728", ms=2.5)
        d = ", ".join(f"{q['depth']:.1f}" for q in out["grooves"]) or "none"
        a.set_title(f"{name} {mod.NAME}: depth {d} px", fontsize=9); a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(OUT + "real_saturated.png", dpi=110, facecolor="white"); plt.close(fig)
