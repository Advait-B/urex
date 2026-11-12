"""figs.py - figures + the cross-validated harness-style numbers for the notes (reads cache/part_a.json, part_b.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("notes", "figs", "quality-gate")
A = json.load(open(os.path.join(HERE, "cache", "part_a.json")))
B = json.load(open(os.path.join(HERE, "cache", "part_b.json")))
COL = {"rule1 (multi_frac)": "#888888", "tree depth 3": "#d98a00", "LR": "#1f6fb4", "HGB": "#c0392b"}
cp = lambda k, n: (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1), 1.0 if k == n else beta.ppf(.975, k + 1, n - k))  # noqa: E731

# 1 risk-coverage
fig, ax = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
for j, s in enumerate(("ideal", "stress", "legal")):
    for name, c in COL.items():
        cv = B["curves"][f"{s}|{name}"]
        x = [q["coverage"] for q in cv]
        ax[0, j].plot(x, [q["p95"] for q in cv], color=c, label=name)
        ax[1, j].plot(x, [q["bad_rate"] for q in cv], color=c)
        op = B["sets"][s][name]
        ax[0, j].plot(op["coverage"], op["p95"], "o", color=c); ax[1, j].plot(op["coverage"], op["bad_rate"], "o", color=c)
    ng = B["sets"][s]["no gate"]
    ax[0, j].axhline(ng["p95"], color="k", ls=":", lw=1); ax[1, j].axhline(ng["bad_rate"], color="k", ls=":", lw=1)
    ax[0, j].set_title(f"{s}  (dotted: no gate)"); ax[0, j].set_yscale("log"); ax[1, j].set_xlabel("share of presses answered")
    ax[0, j].grid(alpha=.3); ax[1, j].grid(alpha=.3)
ax[0, 0].set_ylabel("P95 |depth error| (mm)"); ax[1, 0].set_ylabel("share of answers with an error > 0.5 mm")
ax[0, 0].legend(fontsize=8)
fig.suptitle("Risk-coverage: answer only the presses the gate trusts most (dots = shipped threshold, fixed on train)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "risk_coverage.png"), dpi=110); plt.close(fig)

# 2 feature importance
imp = B["perm_importance_stress"]
keys = sorted(imp["LR"], key=lambda k: -max(imp["LR"][k][0], imp["HGB"][k][0]))[:14][::-1]
fig, ax = plt.subplots(figsize=(7, 5))
yy = np.arange(len(keys))
ax.barh(yy + .2, [imp["LR"][k][0] for k in keys], .4, xerr=[imp["LR"][k][1] for k in keys], color=COL["LR"], label="LR (shipped)")
ax.barh(yy - .2, [imp["HGB"][k][0] for k in keys], .4, xerr=[imp["HGB"][k][1] for k in keys], color=COL["HGB"], label="HGB")
ax.set_yticks(yy); ax.set_yticklabels(keys); ax.set_xlabel("drop in AUC on `stress` when the feature is shuffled"); ax.legend(); ax.grid(alpha=.3, axis="x")
ax.set_title("Risk gate: which features matter (permutation importance)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "risk_importance.png"), dpi=110); plt.close(fig)

# 3 part A ROC (LOSO)
y = np.array(A["y"])
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
for a, pn in zip(ax, ("LOO", "LOSO")):
    for name in ("LR C=0.03 (existing)", "LR nested C", "HistGradientBoosting", "SVM rbf C=1", "k-NN k=5", "tiny CNN 48x48", "LR C=0.03 (existing) +sim"):
        p = np.array(A["preds"][f"{pn}|{name}"])
        f, t, _ = roc_curve(y, p)
        a.plot(f, t, label=f"{name}  AUC {A['scores'][pn][name]['auc']:.2f}", lw=2.2 if name.startswith("LR C=0.03 (existing)") and "sim" not in name else 1.2)
    a.plot([0, 1], [0, 1], "k:", lw=.8); a.set_title(f"real photos, {pn} (19 usable / 67 not)"); a.set_xlabel("false positive rate"); a.grid(alpha=.3)
    a.legend(fontsize=7, loc="lower right")
ax[0].set_ylabel("recall of usable photos")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "frame_gate_roc.png"), dpi=110); plt.close(fig)

# cross-validated versions of the harness numbers on real (usable_recall / false_measure need "has a groove" too)
r = np.load(os.path.join(HERE, "cache", "real.npz"), allow_pickle=True)
has = r["n_pred"] > 0
print("baseline, no gate: usable_recall", (has & (y == 1)).sum(), "/ 19   false_measure", (has & (y == 0)).sum(), "/ 67")
for pn in ("LOO", "LOSO"):
    for name in ("LR C=0.03 (existing)", "LR nested C", "LR C=1 (untuned)"):
        ok = (np.array(A["preds"][f"{pn}|{name}"]) >= 0.5) & has
        k1, k0 = int((ok & (y == 1)).sum()), int((ok & (y == 0)).sum())
        print(f"{pn:5s} {name:22s} usable_recall {k1}/19 = {k1 / 19:.3f} [{cp(k1, 19)[0]:.2f},{cp(k1, 19)[1]:.2f}]   "
              f"false_measure {k0}/67 = {k0 / 67:.3f} [{cp(k0, 67)[0]:.2f},{cp(k0, 67)[1]:.2f}]")
for s in ("ideal", "stress", "legal"):
    for n in ("no gate", "LR"):
        q = B["sets"][s][n]; k = round(q["bad_rate"] * q["n_answered"])
        print(s, n, f"answered {q['n_answered']} bad {k} -> {q['bad_rate']:.3f} [{cp(k, q['n_answered'])[0]:.2f},{cp(k, q['n_answered'])[1]:.2f}]")

# bootstrap (frames resampled, 2000x) intervals for MAE / P95 at the shipped risk threshold
M = json.load(open(os.path.join("models", "risk_lr.json")))
rng = np.random.default_rng(0)
for s in ("ideal", "stress", "legal"):
    d = np.load(os.path.join(HERE, "cache", s + ".npz"), allow_pickle=True)
    risk = 1 / (1 + np.exp(-(((d["X"] - M["mean"]) / M["scale"]) @ np.array(M["coef"]) + M["intercept"])))
    for tag, keep in (("no gate", d["meta"][:, 0] > 0), ("LR gate", (d["meta"][:, 0] > 0) & (risk <= M["threshold"]))):
        per = [np.abs(d["errs"][d["err_frame"] == i]) for i in np.nonzero(keep)[0]]
        bs = []
        for _ in range(2000):
            e = np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))])
            bs.append((e.mean(), np.percentile(e, 95)))
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        print(f"{s:6s} {tag:8s} MAE CI [{lo[0]:.3f},{hi[0]:.3f}]  P95 CI [{lo[1]:.3f},{hi[1]:.3f}]")

# frame gate: permutation importance (model fitted on all 86 photos and shuffled on the same 86: IN-SAMPLE, indicative only)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Xr = r["X"][:, :12]
m = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", max_iter=2000)).fit(Xr, y)
pi = permutation_importance(m, Xr, y, scoring="roc_auc", n_repeats=50, random_state=0)
names = M["features"][:12]
print("frame gate permutation importance (AUC drop, in-sample):",
      ", ".join(f"{names[j]} {pi.importances_mean[j]:+.3f}" for j in np.argsort(-pi.importances_mean)))
