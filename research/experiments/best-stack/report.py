"""report.py - the ablation tables (markdown) and the three figures, straight from results2/*.json.
    python experiments/best-stack/report.py     (from research/)"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
ROWS = [("0 baseline", "baseline_steger"), ("1a + DP line only", "dp_dense"), ("1b + robust depth only", "rd_final"),
        ("2 DP line + robust depth", "stack_dp_rd"), ("2' FIR line + robust depth", "stack_fir_rd"), ("3 + cap", "stack_dp_rd_cap"),
        ("4 + lower-bound rule", "stack_dp_rd_lb"), ("5 + risk gate", "stack_dp_rd_risk"), ("6 + frame gate (RECOMMENDED)", "stack_dp_rd_gate"),
        ("7 row 4 with laser on/off line", "stack_onoff_rd")]
KEYS = ["coverage", "depth_mae", "depth_med", "depth_p95", "gross_rate", "miss_rate", "false_per_frame", "verdict_acc", "false_pass_rate", "cpu_ms"]
REAL = ["line_f1_truck", "depth_mae_px", "found_ref", "ok_ref", "repeat_pair_diff_px", "usable_recall", "false_measure_nolaser", "false_measure_laser"]
FIG = os.path.join("notes", "figs", "best-stack")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
R = {m: json.load(open(os.path.join("results2", m + ".json"))) for _, m in ROWS}


def fmt(v, k=""):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.1f}" if k == "cpu_ms" else f"{v:.3f}"


def tables():
    for s in ("ideal", "stress", "legal", "shifted", "handheld"):
        sfx = "_px" if s == "handheld" else "_mm"
        print(f"\n**{s}**" + (" (pixels, no calibration)" if s == "handheld" else "") + "\n")
        print("| step | module | " + " | ".join(k + (sfx if k.startswith("depth") else "") for k in KEYS) + " |")
        print("|---|---|" + "---|" * len(KEYS))
        for lab, m in ROWS:
            d = R[m]["sets"][s]
            print(f"| {lab} | `{m}` | " + " | ".join(fmt(d.get(k, d.get(k + sfx)), k) for k in KEYS) + " |")
    print("\n**real photos**\n")
    print("| step | module | " + " | ".join(REAL) + " |")
    print("|---|---|" + "---|" * len(REAL))
    for lab, m in ROWS:
        d = R[m]["sets"]["real"]
        print(f"| {lab} | `{m}` | " + " | ".join(fmt(d[k]) for k in REAL) + " |")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color="#e9e8e3", lw=0.6)
    ax.set_axisbelow(True)


def fig_ablation():
    rows = [r for r in ROWS if r[1] != "stack_fir_rd"]
    panels = [("stress", "depth_mae_mm", "depth MAE, mm"), ("stress", "gross_rate", "share of grooves wrong by > 1 mm"),
              ("stress", "verdict_acc", "PASS/FAIL verdict accuracy"), ("shifted", "depth_mae_mm", "depth MAE, mm"),
              ("handheld", "depth_mae_px", "depth MAE, px"), ("stress", "coverage", "coverage (share of presses answered)")]
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (s, k, title) in zip(axs.ravel(), panels):
        v = [R[m]["sets"][s][k] for _, m in rows]
        ax.barh(y, v, height=0.62, color=[ORANGE if m == "stack_dp_rd_gate" else BLUE for _, m in rows])
        for yi, vi in zip(y, v):
            ax.text(vi + 0.01 * max(v), yi, f"{vi:.3f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [lab for lab, _ in rows])
        ax.set_title(f"{s}: {title}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, max(v) * 1.16)
        style(ax)
    fig.suptitle("Ablation: one piece added per row (scorer v2). Orange = recommended single-frame stack.", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "ablation.png"), dpi=130, facecolor="white")


def fig_scatter():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, s in zip(axs, ("stress", "shifted")):
        for m, c, lab, mk in (("baseline_steger", BLUE, "baseline", "o"), ("stack_dp_rd_gate", ORANGE, "recommended stack", "s")):
            d = R[m]["sets"][s]
            t, p = np.array(d["_true"]), np.array(d["_pred"])
            e = np.clip(p - t, -12, 12)
            ax.scatter(t, e, s=16, color=c, marker=mk, alpha=0.75, edgecolors="white", linewidths=0.4,
                       label=f"{lab}: n = {len(t)}, MAE {d['depth_mae_mm']:.3f} mm, > 1 mm: {d['gross_rate']:.3f}, coverage {d['coverage']:.2f}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{s}: depth error of every scored groove", fontsize=10, loc="left", color=INK)
        ax.set_xlabel("true groove depth, mm", fontsize=9, color=MUTED)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16))
        style(ax)
        ax.grid(axis="y", color="#e9e8e3", lw=0.6)
    axs[0].set_ylabel("measured - true, mm (clipped at +-12)", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "error_vs_depth.png"), dpi=130, facecolor="white", bbox_inches="tight")


def fig_real():
    from algos import baseline_steger, dp_dense, stackcommon as sc
    from bench import datasets
    items = {it["name"]: it for it in datasets.load_real()}
    names = ["photo_20261030_154820.jpg", "photo_20261030_154510.jpg", "photo_20261028_162148.jpg"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 6.0))
    for ax, n in zip(axs, names):
        it = items[n]
        ax.imshow(it["gray"], cmap="gray", vmin=0, vmax=255)
        txt = []
        for fn, c, lab, dv in ((baseline_steger.extract_line, BLUE, "baseline", 0), (dp_dense.extract_line, ORANGE, "recommended", 0)):
            ln = fn(it["gray"])
            ax.plot(ln["u"], ln["v"], ".", ms=2.2, color=c, label=f"{lab} line ({len(ln['u'])} points)")
            if lab == "baseline":
                from bench import depth
                g = depth.measure_from_line(ln["u"], ln["v"], None)["grooves"]
            else:
                g = sc.measure(it["gray"], None, None, line_fn=fn, depth="rd", cap=True)["grooves"]
            txt.append(f"{lab}: " + (", ".join(f"{x['depth']:.1f}" for x in g) or "no groove") + " px")
        ref = it["ref"]["depth_px"] if it["ref"] and it["ref"].get("depth_px") else None
        ax.set_title(f"{n[6:-4]} ({it['scene']})\n" + "\n".join(txt) + (f"\nreference: {ref:.1f} px" if ref else ""), fontsize=8, loc="left", color=INK)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "real_overlay.png"), dpi=120, facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    tables()
    fig_ablation(); fig_scatter(); fig_real()
