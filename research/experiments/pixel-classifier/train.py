"""train.py - fit the candidate classifiers on train frames 0..899, validate on 900..1199.
Writes models/cand_hgb.pkl, models/cand_lr.json, models/cand_thresh.json (thresholds filled by tune.py)."""
import json, os, pickle, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.inspection import permutation_importance
sys.path.insert(0, os.getcwd())
from algos import cand_clf as cc

d = np.load("experiments/pixel-classifier/cache/train.npz")
F, y, fr = d["F"], d["y"], d["frame"]
use = y >= 0                                   # drop the ambiguous 1..2 px ring
tr, va = use & (fr < 900), use & (fr >= 900)
rng = np.random.default_rng(0)
sub = rng.choice(np.nonzero(tr)[0], 300_000, replace=False)
print("train pts", tr.sum(), "val pts", va.sum(), "pos frac", y[tr].mean())

t0 = time.time()
hgb = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=50,
                                     class_weight="balanced", random_state=0).fit(F[sub], y[sub])
p = hgb.predict_proba(F[va])[:, 1]
print("hgb fit s", round(time.time() - t0), "val AUC %.4f AP %.4f F1@0.5 %.4f" % (roc_auc_score(y[va], p), average_precision_score(y[va], p), f1_score(y[va], p > 0.5)))
with open("models/cand_hgb.pkl", "wb") as fh:
    pickle.dump({"clf": hgb, "thresh": 0.5, "features": cc.FEATURES}, fh)

L = lambda X: np.sign(X) * np.log1p(np.abs(X))
mean, scale = L(F[sub]).mean(0), L(F[sub]).std(0) + 1e-6
lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced").fit((L(F[sub]) - mean) / scale, y[sub])
pl = lr.predict_proba((L(F[va]) - mean) / scale)[:, 1]
print("lr  val AUC %.4f AP %.4f F1@0.5 %.4f" % (roc_auc_score(y[va], pl), average_precision_score(y[va], pl), f1_score(y[va], pl > 0.5)))
json.dump({"mean": mean.tolist(), "scale": scale.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
           "thresh": 0.5, "features": cc.FEATURES}, open("models/cand_lr.json", "w"))

# simplest alternative: one threshold on one feature (best point-F1 on the training part)
rows = []
for j, name in enumerate(cc.FEATURES):
    x = F[sub, j]; auc = roc_auc_score(y[sub], x); sign = 1 if auc >= 0.5 else -1
    cuts = np.quantile(x, np.linspace(0.02, 0.98, 97))
    f1s = [f1_score(y[sub], (x >= c) if sign > 0 else (x <= c)) for c in cuts]
    k = int(np.argmax(f1s)); xv = F[va, j]
    rows.append((name, max(auc, 1 - auc), sign, float(cuts[k]), f1_score(y[va], (xv >= cuts[k]) if sign > 0 else (xv <= cuts[k]))))
rows.sort(key=lambda r: -r[4])
print("single-feature thresholds (val point F1):")
for r in rows[:10]:
    print("  %-20s AUC %.3f sign %+d cut %.4g F1 %.4f" % r)
json.dump({"feature": rows[0][0], "sign": rows[0][2], "cut": rows[0][3], "thresh": 0.5}, open("models/cand_thresh.json", "w"))

# permutation importance on the validation frames (drop in average precision)
vs = rng.choice(np.nonzero(va)[0], 60_000, replace=False)
pi = permutation_importance(hgb, F[vs], y[vs], scoring="average_precision", n_repeats=3, random_state=0, n_jobs=1)
imp = sorted(zip(cc.FEATURES, pi.importances_mean, pi.importances_std), key=lambda r: -r[1])
for n_, m_, s_ in imp:
    print("  %-20s %.4f +- %.4f" % (n_, m_, s_))
json.dump({"permutation_ap_drop": [[n_, float(m_), float(s_)] for n_, m_, s_ in imp],
           "single_feature": [list(r) for r in rows], "lr_coef": dict(zip(cc.FEATURES, lr.coef_[0].tolist()))},
          open("experiments/pixel-classifier/train_report.json", "w"), indent=1)
