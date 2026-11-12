"""figures.py - the PNGs in notes/figs/robust-depth/ (run floor.py stress and uncertainty.py ideal,stress,legal,uneven_test first)."""
import functools, json, os, sys
sys.path.insert(0, "experiments/robust-depth")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common as C
from algos import robust_depth as rd
C.memo_surface()
OUT = "notes/figs/robust-depth"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# 1. geometric bias vs true depth (true line, ideal + stress, bench floor estimator)
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
for a, (tag, kw) in zip(ax, (("no correction (bench default)", {"correct": False}), ("surface fitted in lateral X", {"curvature": False}),
                             ("lateral X + curvature term (R = 480)", {}))):
    d, e = [], []
    for name in ("ideal", "stress"):
        r = C.score(functools.partial(rd.stage, **kw), name, "oracle")
        P = [(t, p) for t, p, i in C.pairs(name, r["_outs"]) if p and t["floor_visible_px"] >= 3]
        d += [t["depth_mm"] for t, p in P]; e += [p["depth"] - t["depth_mm"] for t, p in P]
    d, e = np.array(d), np.array(e)
    a.scatter(d, e, s=5, alpha=0.4, color="#4878a8", lw=0)
    b = np.arange(0, 19, 3)
    a.plot(b[:-1] + 1.5, [np.mean(e[(d >= x) & (d < x + 3)]) for x in b[:-1]], "o-", color="#c44e52", ms=4, label="mean per 3 mm bin")
    a.axhline(0, color="k", lw=0.6); a.set_ylim(-0.5, 0.6); a.set_title(f"{tag}\nmean {e.mean():+.3f} mm, MAE {np.abs(e).mean():.3f} mm", fontsize=9)
    a.set_xlabel("true groove depth (mm)")
ax[0].set_ylabel("depth error (mm), true line"); ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/bias_vs_depth.png", dpi=130); plt.close(fig)

# 2. floor estimators on stress: cumulative |error|
rec = json.load(open(os.path.join(C.CACHE, "floor_stress.json")))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for a, src in zip(ax, ("oracle", "baseline")):
    for fl, col in zip(rd.FLOORS, ("#222222", "#8c8c8c", "#55a868", "#8172b2", "#dd8452", "#ccb974", "#c44e52")):
        e = np.sort(np.abs([x["pred"] - x["true"] for x in rec if x["floor"] == fl and x["src"] == src]))
        a.plot(e, np.arange(1, len(e) + 1) / len(e), color=col, lw=2 if fl in ("band", "dbscan") else 1.1, label=f"{fl} (MAE {e.mean():.2f})")
    a.set_xscale("log"); a.set_xlim(0.01, 20); a.set_xlabel("|depth error| (mm)")
    a.set_title(f"stress, {'true line' if src == 'oracle' else 'baseline line'}", fontsize=9); a.legend(frameon=False, fontsize=7.5, loc="lower right")
ax[0].set_ylabel("share of grooves with error below x")
fig.tight_layout(); fig.savefig(f"{OUT}/floor_estimators_stress.png", dpi=130); plt.close(fig)

# 3. interval coverage
U = json.load(open(os.path.join(C.CACHE, "unc_ideal_stress_legal_uneven_test.json")))
fig, ax = plt.subplots(figsize=(4.6, 4))
nom = np.linspace(0.05, 0.99, 60)
from scipy.stats import norm
zq = norm.ppf(0.5 + nom / 2)
for (k, v), col in zip([(k, v) for k, v in U.items() if "uneven" not in k], ("#4878a8", "#4878a8", "#c44e52", "#c44e52", "#55a868", "#55a868")):
    z = np.abs(np.array(v["e"])) / np.array(v["s"])
    ax.plot(nom, [(z < q).mean() for q in zq], color=col, ls="-" if "oracle" in k else "--", label=k.replace("/", ", ").replace("oracle", "true line").replace("baseline", "baseline line"))
ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.axvline(0.95, color="k", lw=0.4)
ax.set_xlabel("nominal coverage of depth +- z*sigma"); ax.set_ylabel("observed coverage"); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/interval_coverage.png", dpi=130); plt.close(fig)

# 4. the legal boundary: predicted vs true on uneven_test (baseline line), bench default vs recommended
from bench import depth
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
for a, (tag, st) in zip(ax, (("bench default stage", depth.measure_from_line),
                             ("recommended stage", functools.partial(rd.stage, floor="dbscan", detect="shape", lo_k=3.5)))):
    r = C.score(st, "uneven_test", "baseline")
    P = C.pairs("uneven_test", r["_outs"])
    f = [(t["depth_mm"], p["depth"]) for t, p, i in P if p]; m = [t["depth_mm"] for t, p, i in P if p is None]
    a.scatter(*zip(*f), s=6, color="#4878a8", alpha=0.6, lw=0, label="found")
    a.scatter(m, np.full(len(m), -0.12), s=14, marker="x", color="#c44e52", lw=0.8, label=f"missed ({len(m)})")
    a.plot([0, 4.2], [0, 4.2], "k:", lw=0.7); a.axhline(1.6, color="k", lw=0.5); a.axvline(1.6, color="k", lw=0.5)
    a.set_title(f"{tag}: verdict acc {r['verdict_acc']:.3f}, false PASS {r['false_pass_rate']:.3f}", fontsize=9)
    a.set_xlabel("true depth (mm)"); a.legend(frameon=False, loc="upper left")
ax[0].set_ylabel("measured depth (mm), baseline line")
fig.tight_layout(); fig.savefig(f"{OUT}/legal_boundary.png", dpi=130); plt.close(fig)
