"""evaluate.py - things the harness does not print: readout comparison on the held-out part of
`train`, mask statistics on the real photos, and the figures for the notes.

    cd research
    python experiments/cnn-segmentation/evaluate.py val  [weights.pt]   # readouts on the 200 held-out train frames
    python experiments/cnn-segmentation/evaluate.py real [weights.pt]   # mask area / false lines on real photos
    python experiments/cnn-segmentation/evaluate.py figs                # notes/figs/cnn-segmentation/*.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from algos import baseline_steger, cnn_line  # noqa: E402
from bench import calib, datasets, depth, run  # noqa: E402

FIG = "notes/figs/cnn-segmentation"


def measure_with(readout, path):
    def measure(gray, cal=None, off=None):
        p = cnn_line.extract_line(gray, readout=readout, path=path)
        out = depth.measure_from_line(p["u"], p["v"], cal)
        out["line"] = p
        return out
    return measure


def val(path):
    items = datasets.load_sim("train")[-200:]
    cal = calib.default_calib()
    keys = ["line_f1", "line_rmse_px", "depth_mae_mm", "depth_p95_mm", "miss_rate", "false_per_frame", "verdict_acc", "ms_per_frame"]
    print("held-out train (200 frames)  " + " ".join(f"{k[:12]:>12s}" for k in keys))
    _, base = run.resolve(baseline_steger)
    for name, m in [("baseline", base)] + [(r, measure_with(r, path)) for r in ("steger", "heat", "gated", "column")]:
        r = run.eval_sim(m, items, cal)
        print(f"{name:28s} " + " ".join(f"{r[k]:12.3f}" for k in keys))


def real(path):
    items = datasets.load_real()
    rows = []
    for it in items:
        p = cnn_line.extract_line(it["gray"], path=path)
        g = depth.measure_from_line(p["u"], p["v"], None)["grooves"]
        rows.append((it["usable"], it["laser"], float(p["mask"].mean()), float((p["heat"] > 0.5).mean()), len(p["u"]), bool(g), it["name"]))
    a = np.array([r[:6] for r in rows], float)
    for lab, sel in [("usable (19)", a[:, 0] == 1), ("unusable, laser visible (12)", (a[:, 0] == 0) & (a[:, 1] == 1)),
                     ("unusable, no laser (55)", (a[:, 0] == 0) & (a[:, 1] == 0)), ("all unusable (67)", a[:, 0] == 0)]:
        s = a[sel]
        print(f"{lab:30s} n={len(s):2d}  mask area {100 * s[:, 2].mean():.3f} % of frame (heat>0.5: {100 * s[:, 3].mean():.3f} %)  "
              f"frames with any line points {int((s[:, 4] > 0).sum())}  with a groove reported {int(s[:, 5].sum())}")
    print("unusable frames with a groove:", [r[6][6:21] + ("(laser)" if r[1] else "") for r in rows if not r[0] and r[5]])


def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)

    def panel(ax, gray, title, truth=None, ref=None):
        p = cnn_line.extract_line(gray)
        b = baseline_steger.extract_line(gray)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax.contour(p["heat"], levels=[cnn_line.THR], colors="#00a0ff", linewidths=0.6)
        ax.plot(b["u"], b["v"], ".", color="#ff9d00", ms=2.2, label="baseline")
        ax.plot(p["u"], p["v"], ".", color="#ff2bd0", ms=1.0, label="CNN-gated points")
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])

    st = datasets.load_sim("stress")
    cal = calib.default_calib()
    # the stress frames where the baseline line F1 is worst: glints, paint, dim laser
    f1b = [run.line_scores(*(lambda q: (q["u"], q["v"]))(baseline_steger.extract_line(it["on"])), it["truth"]["u"], it["truth"]["v"], 1.0)["f1"] for it in st]
    order = np.argsort(f1b)[:8]
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.2))
    for ax, i in zip(axs.ravel(), order):
        q = cnn_line.extract_line(st[i]["on"])
        f1c = run.line_scores(q["u"], q["v"], st[i]["truth"]["u"], st[i]["truth"]["v"], 1.0)["f1"]
        panel(ax, st[i]["on"], f"stress #{i}: line F1 baseline {f1b[i]:.2f} -> CNN {f1c:.2f}")
    axs[0, 0].legend(fontsize=6, loc="lower left", markerscale=4)
    fig.suptitle("Simulated stress frames where the baseline is worst. Orange = baseline points, magenta = CNN-gated points, blue = CNN mask outline", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sim_stress.png", dpi=130, facecolor="white")
    plt.close(fig)

    R = datasets.load_real()
    refs = [r for r in R if r["ref"] is not None]
    fig, axs = plt.subplots(2, 6, figsize=(13, 6.6))
    for ax, it in zip(axs.ravel(), refs):
        q = cnn_line.extract_line(it["gray"])
        s = run.line_scores(q["u"], q["v"], it["ref"]["u"], it["ref"]["v"], 1.5)
        panel(ax, it["gray"], f"{it['name'][6:21]} ({it['scene']})\nF1 vs ref {s['f1']:.2f}")
    fig.suptitle("The 12 real reference photos (never used for training). Orange = baseline (= reference), magenta = CNN-gated points, blue = CNN mask outline", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/real_refs.png", dpi=130, facecolor="white")
    plt.close(fig)

    bad = [r for r in R if not r["usable"]]
    area = [float(cnn_line.extract_line(r["gray"])["mask"].mean()) for r in bad]
    order = np.argsort(area)[::-1][:12]
    fig, axs = plt.subplots(2, 6, figsize=(13, 6.6))
    for ax, i in zip(axs.ravel(), order):
        panel(ax, bad[i]["gray"], f"{bad[i]['name'][6:21]} laser={bad[i]['laser']}\nmask {100 * area[i]:.2f} % of frame")
    fig.suptitle("Unusable real photos with the LARGEST predicted mask (the worst false lines; laser=1 means a laser line is really there)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/real_unusable.png", dpi=130, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    path = sys.argv[2] if len(sys.argv) > 2 else None
    {"val": lambda: val(path), "real": lambda: real(path), "figs": figs}[sys.argv[1]]()
