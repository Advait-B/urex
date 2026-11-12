"""figs.py - the figures of notes/end-to-end.md (reads the result JSONs; run after ood.py, real_check.py)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from algos import e2e_cnn, e2e_gbm  # noqa: E402
from bench import datasets  # noqa: E402

OUT = "notes/figs/end-to-end/"
E = "experiments/end-to-end/"
C = {"e2e_cnn": "#1f6fb4", "e2e_gbm": "#d9792b", "baseline": "#555555"}
SETC = {"ideal": "#2a9d8f", "stress": "#c8553d", "legal": "#3d5a80"}
ood = json.load(open(E + "ood_results.json"))
real = json.load(open(E + "real_results.json"))

# 1. predicted vs true -------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
for a, (name, title) in zip(ax, (("e2e_cnn", "CNN: every groove"), ("e2e_gbm", "GBM: shallowest groove of the frame"),
                                 ("baseline_steger", "Geometric baseline: every groove"))):
    for s in ("stress", "ideal", "legal"):
        if name == "e2e_gbm":
            t, p = ood[s]["e2e_gbm"]["_true_min"], ood[s]["e2e_gbm"]["_pred_min"]
        else:
            r = json.load(open(f"results/{name}.json"))["sets"][s]
            t, p = r["_true_mm"], r["_pred_mm"]
        a.plot(t, np.clip(p, -1, 24), ".", ms=4, alpha=0.6, color=SETC[s], label=s)
    a.plot([0, 19], [0, 19], "k-", lw=0.6)
    a.axhline(1.6, color="r", lw=0.6, ls="--"); a.axvline(1.6, color="r", lw=0.6, ls="--")
    a.set(xlim=(0, 19), ylim=(-1, 24), xlabel="true depth (mm)", ylabel="predicted depth (mm)", title=title)
    a.grid(alpha=0.3)
ax[0].legend(frameon=False)
ax[2].text(0.5, 23, "points above 24 mm are clipped", fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "pred_vs_true.png", dpi=130); plt.close(fig)

# 2. out-of-distribution ------------------------------------------------------------------------
shifts = ["in_dist", "laser_37deg", "deep_18_22", "four_grooves", "res_640x480", "uneven_wear"]
labels = ["control\n(like train)", "laser 37 deg\n(another unit)", "grooves\n18-22 mm", "4 grooves", "640x480", "uneven wear"]
fair = {"laser_37deg": "baseline_recal", "res_640x480": "baseline_downscaled"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(shifts))
for a, key, yl in ((ax[0], "depth_mae_mm", "depth MAE (mm), log scale"), (ax[1], "verdict_acc", "verdict accuracy")):
    for i, (algo, lab) in enumerate((("e2e_cnn", "CNN"), ("e2e_gbm", "GBM (one number per frame)"), ("baseline", "geometric baseline (re-calibrated where the device changed)"))):
        v = [ood[s][fair.get(s, algo) if algo == "baseline" else algo][key] for s in shifts]
        a.bar(x + (i - 1) * 0.27, v, 0.25, color=C[algo], label=lab)
        for xx, vv in zip(x + (i - 1) * 0.27, v):
            a.text(xx, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=6.5)
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=8); a.set_ylabel(yl); a.grid(axis="y", alpha=0.3)
ax[0].set_yscale("log"); ax[0].set_ylim(0.1, 60); ax[1].set_ylim(0.5, 1.05)
ft = ood["laser_37deg"]["e2e_cnn_finetune50"]["depth_mae_mm"]
ax[0].plot([1 - 0.27], [ft], "v", color="k"); ax[0].text(1 - 0.22, ft * 0.8, "after fine-tuning\non 50 labelled frames", fontsize=6.5, va="top")
ax[0].legend(frameon=False, fontsize=7, loc="upper left")
ax[0].set_title("Error under each shift (100 frames each)"); ax[1].set_title("PASS / FAIL at 1.6 mm")
fig.tight_layout(); fig.savefig(OUT + "ood_error.png", dpi=130); plt.close(fig)

# 3. real photos + laser-off --------------------------------------------------------------------
items = {it["name"]: it for it in datasets.load_real()}
rows = real["rows"]
bad = [r for r in rows if not r["usable"] and r["cnn_n"]]
pick = [next(r for r in bad if r["scene"] == sc) for sc in ("people", "bench", "rover_tyre")]
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
for a, r in zip(ax[0], pick):
    a.imshow(items[r["name"]]["gray"], cmap="gray", vmin=0, vmax=255); a.axis("off")
    a.set_title(f"real, not measurable ({r['scene']})\nCNN: {r['cnn_n']} groove(s), {r['cnn_max_mm']:.1f} mm, conf {r['cnn_conf']:.2f} -> PASS\n"
                f"GBM: {r['gbm_mm']:.1f} mm -> PASS", fontsize=8)
ref = [r for r in rows if r["ref_px"] is not None]
a = ax[1, 0]
a.plot([r["ref_px"] for r in ref], [r["cnn_max_mm"] or 0 for r in ref], "o", color=C["e2e_cnn"], label=f"CNN (Spearman {real['spearman_cnn_max_mm'][0]:.2f})")
a.plot([r["ref_px"] for r in ref], [r["gbm_mm"] for r in ref], "s", color=C["e2e_gbm"], label=f"GBM (Spearman {real['spearman_gbm_mm'][0]:.2f})")
a.set(xlabel="reference depth (px)", ylabel="predicted depth (mm)", title="6 truck-tyre reference frames"); a.grid(alpha=0.3); a.legend(frameon=False, fontsize=8)
it = datasets.load_sim("ideal")[3]
for a, key in zip(ax[1, 1:], ("on", "off")):
    p, d = e2e_cnn.profile(it[key])
    a.imshow(it[key][60:200], cmap="gray", extent=(0, 320, 200, 60), aspect="auto", vmin=0, vmax=160)
    a2 = a.twinx(); a2.plot(np.where(p > 0.5, d, np.nan), color="#ffb000", lw=2); a2.set_ylim(0, 40); a2.set_ylabel("CNN depth (mm)")
    true = ", ".join(f"{g['depth_mm']:.1f}" for g in it["truth"]["grooves"] if g["in_view"])
    got = ", ".join(f"{g['depth']:.1f}" for g in e2e_cnn.measure(it[key])["grooves"]) or "nothing"
    a.set_title(f"simulated, laser {key.upper()}: true {true} mm, CNN says {got}", fontsize=8); a.set_yticks([])
fig.tight_layout(); fig.savefig(OUT + "real_and_laser_off.png", dpi=110); plt.close(fig)
print("figures written")
