"""figs.py - the figures for notes/pixel-classifier.md."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from bench import datasets, run
from algos import cand_clf as cc, baseline_steger as bs
OUT = "notes/figs/pixel-classifier"
cc.MODEL = "hgb"

def panel(ax, img, title, kept=None, rej=None, ref=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    if rej is not None: ax.plot(*rej, ".", color="#e4572e", ms=1.5, label="rejected")
    if ref is not None: ax.plot(*ref, ".", color="#ffd400", ms=1.2, label="reference")
    if kept is not None: ax.plot(*kept, ".", color="#19d3f3", ms=1.5, label="kept")
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])

# 1. stress frames where the baseline is worst
stress = datasets.load_sim("stress")
sc = []
for i, it in enumerate(stress[:80]):
    b = bs.extract_line(it["on"]); s = run.line_scores(b["u"], b["v"], it["truth"]["u"], it["truth"]["v"], 1.0)
    sc.append(s["f1"])
pick = np.argsort(sc)[[0, 3, 8]]
fig, axs = plt.subplots(3, 3, figsize=(10, 7.6))
for r, i in enumerate(pick):
    it = stress[i]; tr = it["truth"]
    b = bs.extract_line(it["on"]); c, F, p, k = cc.extract_line(it["on"], return_all=True)
    s2 = run.line_scores(c["u"][k], c["v"][k], tr["u"], tr["v"], 1.0)
    panel(axs[r, 0], it["on"], f"stress #{i}: image (laser peak {tr['scene']['line_peak_dn']:.0f} DN)")
    panel(axs[r, 1], it["on"], f"baseline chain selection, line F1 {sc[i]:.2f}", kept=(b["u"], b["v"]))
    panel(axs[r, 2], it["on"], f"cand_hgb: {k.sum()} kept / {(~k).sum()} rejected, F1 {s2['f1']:.2f}", kept=(c["u"][k], c["v"][k]), rej=(c["u"][~k], c["v"][~k]))
axs[0, 2].legend(fontsize=7, loc="upper right", markerscale=5)
fig.tight_layout(); fig.savefig(f"{OUT}/stress_kept_rejected.png", dpi=130); plt.close(fig)

# 2. real photos: 2 truck-groove references, 1 lug-step reference, 1 unusable frame that still gets a depth
real = datasets.load_real()
by = {it["name"]: it for it in real}
names = ["photo_20261030_154446.jpg", "photo_20261030_154510.jpg", [n for n in by if n.endswith("162148.jpg")][0]]
fig, axs = plt.subplots(2, 3, figsize=(9, 8.4))
for j, n in enumerate(names):
    it = by[n]; ref = it["ref"]
    b = bs.extract_line(it["gray"]); c, F, p, k = cc.extract_line(it["gray"], return_all=True)
    s = run.line_scores(c["u"][k], c["v"][k], ref["u"], ref["v"], 1.5)
    panel(axs[0, j], it["gray"], f"{n[6:-4]} ({ref['mode']}): baseline = reference", kept=(b["u"], b["v"]))
    panel(axs[1, j], it["gray"], f"cand_hgb: P {s['precision']:.2f} R {s['recall']:.2f} vs reference", kept=(c["u"][k], c["v"][k]), rej=(c["u"][~k], c["v"][~k]))
axs[1, 0].legend(fontsize=7, loc="upper right", markerscale=5)
fig.tight_layout(); fig.savefig(f"{OUT}/real_kept_rejected.png", dpi=130); plt.close(fig)

# 3. permutation importance, with and without the modal-row features
rep = json.load(open("experiments/pixel-classifier/train_report.json"))["permutation_ap_drop"]
loc = json.load(open("experiments/pixel-classifier/importance_hgb_local.json"))
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, d, t in ((axs[0], rep, "cand_hgb (30 features)"), (axs[1], loc, "cand_hgb_local (no dv_mode / abs_dv_mode)")):
    d = d[:12][::-1]
    ax.barh([x[0] for x in d], [x[1] for x in d], xerr=[x[2] for x in d], color="#3b6ea5")
    ax.set_xscale("log"); ax.set_xlabel("drop in average precision when the feature is shuffled (held-out train frames)")
    ax.set_title(t, fontsize=9); ax.tick_params(labelsize=8); ax.xaxis.label.set_size(7)
fig.tight_layout(); fig.savefig(f"{OUT}/feature_importance.png", dpi=130); plt.close(fig)

# numbers quoted in the notes
tr = datasets.load_sim("train")
vm = [np.median(it["truth"]["v"]) for it in tr if len(it["truth"]["v"])]
dmax = [np.percentile(it["truth"]["v"], 99) - np.median(it["truth"]["v"]) for it in tr if len(it["truth"]["v"])]
print("sim train: median line row mean %.1f sd %.1f min %.1f max %.1f; line depth below its median row: p50 %.1f p95 %.1f max %.1f px" % (np.mean(vm), np.std(vm), np.min(vm), np.max(vm), np.median(dmax), np.percentile(dmax, 95), np.max(dmax)))
for it in real:
    if it["ref"] is not None:
        v = it["ref"]["v"]; print(it["name"][6:-4], it["ref"]["mode"], it["gray"].shape, "ref v median %.0f, deepest below median %.0f px, slope span %.0f" % (np.median(v), v.max() - np.median(v), np.ptp(v)))
a, b_ = by["photo_20261030_154820.jpg"]["gray"], by["photo_20261030_154854.jpg"]["gray"]
print("repeat pair identical pixels:", a.shape == b_.shape and bool((a == b_).all()), "mean abs diff", float(np.mean(np.abs(a.astype(int) - b_.astype(int)))) if a.shape == b_.shape else None)
