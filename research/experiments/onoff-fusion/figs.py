"""figs.py - the four figures of notes/onoff-fusion.md (reads the JSON files written by the other scripts)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from perturb import perturb_off
from bench import datasets
from algos import baseline_steger, onoff_cog, onoff_common as oc

E = "experiments/onoff-fusion/"
OUT = "notes/figs/onoff-fusion/"
os.makedirs(OUT, exist_ok=True)
C = {"single+steger": "#555555", "plain+steger": "#b07aa1", "plain+cog": "#e15759", "reg+cog": "#4e79a7", "cons+cog": "#f28e2b", "robust+cog": "#59a14f"}

# 1 ---- example: on / off / difference under sunlight with glints
items = datasets.load_sim("stress")
sc = [(len(it["truth"]["scene"]["glints"]) * it["truth"]["scene"]["ambient"], i) for i, it in enumerate(items)]
i = sorted(sc)[-3][1]
it = items[i]
on, off = it["on"], it["off"]
moved = perturb_off(off, 2.0, 0.0, 1.0, np.random.default_rng(1))
panels = [("laser ON", on, baseline_steger.extract_line(on), "baseline points"),
          ("laser OFF", off, None, ""),
          ("ON - OFF (no motion)", oc.difference(on, off, "plain"), onoff_cog.extract_line(on, off), "centre-of-gravity points"),
          ("ON - OFF, off frame moved 2 px", oc.difference(on, moved, "plain"), oc.cog_line(oc.difference(on, moved, "plain")), "plain subtraction"),
          ("moved 2 px, registered + 1 px guard", oc.difference(on, moved, "robust"), oc.cog_line(oc.difference(on, moved, "robust")), "robust subtraction")]
fig, ax = plt.subplots(1, 5, figsize=(17, 3.2))
for a, (t, img, ln, lab) in zip(ax, panels):
    a.imshow(img, cmap="gray", vmin=0, vmax=255)
    a.plot(it["truth"]["u"], it["truth"]["v"], color="#59a14f", lw=0.6, label="true line")
    if ln is not None:
        a.plot(ln["u"], ln["v"], ".", color="#e15759", ms=2, label=lab)
        a.legend(fontsize=6, loc="lower right")
    a.set_title(t, fontsize=9)
    a.axis("off")
fig.suptitle(f"stress frame {i}: ambient {it['truth']['scene']['ambient']:.2f}, {len(it['truth']['scene']['glints'])} glints", fontsize=9)
fig.tight_layout()
fig.savefig(OUT + "example_on_off_diff.png", dpi=130, facecolor="white")

# 2 ---- error vs motion
m = {**json.load(open(E + "motion_a.json")), **json.load(open(E + "motion_b.json"))}
sh = [0, 0.5, 1, 2, 3, 5]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, key, lab in zip(ax, ["depth_mae_mm", "false_per_frame", "line_f1"], ["depth MAE (mm, log)", "false grooves per frame", "line F1"]):
    for v in ["plain+steger", "plain+cog", "reg+cog", "cons+cog", "robust+cog"]:
        a.plot(sh, [m[f"t{s}|{v}"][key] for s in sh], "o-", color=C[v], label=v, ms=4)
    a.axhline(m["t0|single+steger"][key], color=C["single+steger"], ls="--", label="single-frame baseline")
    a.set_xlabel("off-frame shift (px)")
    a.set_title(lab, fontsize=10)
    a.grid(alpha=0.3)
ax[0].set_yscale("log")
ax[0].legend(fontsize=7)
fig.suptitle("stress set (200 frames): how much motion between the ON and OFF frame can each subtraction take?", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "error_vs_motion.png", dpi=130, facecolor="white")

# 3 ---- error vs N
f = json.load(open(E + "fusion.json"))
NS = [1, 3, 5, 9]
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for a, (cond, algo) in zip(ax, [("stress", "onoff"), ("legal_stress", "onoff"), ("legal", "single"), ("stress", "single")]):
    r = f[f"{cond}|{algo}"]
    for rule, col in zip(["mean", "median", "median+MAD", "quality"], ["#e15759", "#4e79a7", "#59a14f", "#f28e2b"]):
        a.plot(NS, [r[f"{N}|{rule}"]["mae"] for N in NS], "o-", color=col, label=rule, ms=4)
    base = r["1|mean"]["mae"]
    a.plot(NS, base / np.sqrt(NS), "k--", lw=1, label="1/sqrt(N) (pure noise)")
    a.set_title(f"{cond} scenes, {'on/off COG' if algo == 'onoff' else 'single-frame baseline'}", fontsize=9)
    a.set_xlabel("presses N")
    a.set_xticks(NS)
    a.set_ylim(0, None)
    a.grid(alpha=0.3)
ax[0].set_ylabel("depth MAE (mm)")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT + "error_vs_presses.png", dpi=130, facecolor="white")

# 4 ---- one factor at a time
fa = json.load(open(E + "factors.json"))
names = list(fa)
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for a, key, lab in zip(ax, ["line_f1", "depth_mae_mm"], ["line F1", "depth MAE (mm, log)"]):
    for k, (v, col) in enumerate(zip(["single+steger", "single+cog", "plain+steger", "plain+cog"], ["#555555", "#bab0ac", "#b07aa1", "#e15759"])):
        a.bar(np.arange(len(names)) + (k - 1.5) * 0.2, [fa[n][v][key] for n in names], 0.2, color=col, label=v.replace("single", "ON only").replace("plain", "ON-OFF"))
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    a.set_title(lab, fontsize=10)
    a.grid(axis="y", alpha=0.3)
ax[1].set_yscale("log")
ax[0].set_ylim(0.4, 1)
ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle("40 ideal scenes with ONE stress factor added", fontsize=10)
fig.tight_layout()
fig.savefig(OUT + "factors.png", dpi=130, facecolor="white")
print("figures written; example frame", i)
