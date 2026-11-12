"""figs.py - the figures and the results table for notes/peak-detectors.md (run from research/)."""
import importlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench import datasets

OUT = "notes/figs/peak-detectors"
os.makedirs(OUT, exist_ok=True)
ALGOS = ["baseline_steger", "pk_steger_s1", "pk_cog", "pk_gauss3", "pk_blais4", "pk_blais8", "pk_fir", "pk_fir2d",
         "pk_fir2d_rows", "pk_fir2d_th1d", "pk_fir2d_colpct", "pk_fir2d_nobg", "oracle_line"]
STYLE = {"baseline_steger": ("k", "o", "-"), "pk_steger_s1": ("0.55", "s", "--"), "pk_cog": ("#1f77b4", "^", "-"),
         "pk_gauss3": ("#d62728", "v", "-"), "pk_blais8": ("#2ca02c", "D", "-"), "pk_fir": ("#ff7f0e", "P", "-"),
         "pk_fir2d": ("#9467bd", "*", "-")}
R = {a: json.load(open(f"results/{a}.json"))["sets"] for a in ALGOS}

# ---- table
print("| algorithm | set | line F1 | line RMSE px | depth MAE mm | median abs err mm* | P95 mm | bias mm | miss | false/frame | verdict acc | false pass |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for a in ALGOS:
    for s in ("ideal", "stress", "legal"):
        r = R[a][s]
        med = float(np.median(np.abs(np.array(r["_pred_mm"]) - np.array(r["_true_mm"]))))
        print(f"| {a} | {s} | {r['line_f1']:.3f} | {r['line_rmse_px']:.3f} | {r['depth_mae_mm']:.3f} | {med:.3f} | {r['depth_p95_mm']:.3f} | "
              f"{r['depth_bias_mm']:.3f} | {r['miss_rate']:.3f} | {r['false_per_frame']:.3f} | {r['verdict_acc']:.3f} | {r['false_pass_rate']:.3f} |")
print()
print("| algorithm | line_f1_ref | depth_mae_px | found_ref | repeat_pair_diff_px | usable_recall | false_measure |")
print("|---|---|---|---|---|---|---|")
for a in ALGOS[:-1]:
    r = R[a]["real"]
    print(f"| {a} | {r['line_f1_ref']:.3f} | {r['depth_mae_px']:.3f} | {r['found_ref']} | {r['repeat_pair_diff_px']:.3f} | "
          f"{r['usable_recall']:.3f} | {r['false_measure']:.3f} |")

# ---- fig 1: sweeps
sw = json.load(open("experiments/peak-detectors/sweep.json"))
fig, ax = plt.subplots(2, 4, figsize=(15, 6.4), sharey="row")
names = {"line_peak_dn": "laser peak (DN before clipping at 255)", "blur_px": "defocus blur sigma (px)",
         "speckle": "speckle contrast", "ambient": "ambient light level"}
for j, (f, rows) in enumerate(sw["sweeps"].items()):
    x = [r["level"] for r in rows]
    for a, (c, m, ls) in STYLE.items():
        ax[0, j].plot(x, [r[a]["rmse"] for r in rows], color=c, marker=m, ls=ls, ms=5, lw=1.3, label=a)
        ax[1, j].plot(x, [r[a]["f1"] for r in rows], color=c, marker=m, ls=ls, ms=5, lw=1.3)
    ax[1, j].set_xlabel(names[f])
    ax[0, j].grid(alpha=0.3)
    ax[1, j].grid(alpha=0.3)
    if f == "line_peak_dn":
        ax[0, j].set_xscale("log")
        ax[1, j].set_xscale("log")
        for r in rows[::2]:
            ax[0, j].annotate(f"{100 * r['sat_frac']:.0f}% sat", (r["level"], 0.42), ha="center", fontsize=7, color="0.4")
ax[0, 0].set_ylabel("line RMSE (px), harness metric")
ax[1, 0].set_ylabel("line F1 at 1 px")
ax[0, 0].set_ylim(0.08, 0.45)
ax[0, 3].legend(fontsize=7, loc="upper left")
fig.suptitle("One factor at a time, 16 scenes per level (base: peak 180 DN, blur 0.8 px, speckle 0.2, ambient 0.15)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/sweeps.png", dpi=110)

# ---- fig 2: real frames
real = {r["name"][15:21]: r for r in datasets.load_real() if r["ref"] is not None}
mods = [importlib.import_module("algos." + a) for a in ("pk_fir2d", "pk_fir2d_rows")]
keys = ["154854", "160117", "154510", "162117", "162148"]
fig, ax = plt.subplots(2, 5, figsize=(16, 7.4))
for j, k in enumerate(keys):
    r = real[k]
    for i, m in enumerate(mods):
        p = m.extract_line(r["gray"])
        a = ax[i, j]
        a.imshow(r["gray"], cmap="gray")
        a.plot(r["ref"]["u"], r["ref"]["v"], ".", color="#00d000", ms=4, label="reference (baseline)")
        a.plot(p["u"], p["v"], ".", color="red", ms=2, label=m.NAME)
        if k != "162148":
            v0 = float(np.median(r["ref"]["v"]))
            a.set_ylim(v0 + 60, v0 - 45)
        a.set_title(f"{k}  ({r['ref']['mode']})", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        if j == 0:
            a.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(f"{OUT}/real_frames.png", dpi=100)

# ---- fig 3: does line RMSE predict depth error?
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for j, s in enumerate(("ideal", "legal", "stress")):
    for a in ALGOS[:-1]:
        r = R[a][s]
        med = float(np.median(np.abs(np.array(r["_pred_mm"]) - np.array(r["_true_mm"]))))
        ax[j].plot(r["line_rmse_px"], r["depth_mae_mm"], "o", color="#1f77b4")
        ax[j].plot(r["line_rmse_px"], med, "s", color="#ff7f0e", mfc="none")
        ax[j].annotate(a.replace("pk_", "").replace("baseline_", ""), (r["line_rmse_px"], r["depth_mae_mm"]), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    ax[j].set_title(s)
    ax[j].set_xlabel("line RMSE (px)")
    ax[j].set_yscale("log")
    ax[j].grid(alpha=0.3, which="both")
ax[0].set_ylabel("depth error (mm)")
ax[0].plot([], [], "o", color="#1f77b4", label="depth MAE (harness)")
ax[0].plot([], [], "s", color="#ff7f0e", mfc="none", label="median |error| (from the same JSON)")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/rmse_vs_depth.png", dpi=110)
