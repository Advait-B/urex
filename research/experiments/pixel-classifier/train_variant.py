"""train_variant.py NAME drop1,drop2,... - HGB on a feature subset -> models/cand_NAME.pkl (+ permutation importance)."""
import json, os, pickle, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
sys.path.insert(0, os.getcwd())
from algos import cand_clf as cc

name, drop = sys.argv[1], sys.argv[2].split(",")
cols = [i for i, f in enumerate(cc.FEATURES) if f not in drop]
d = np.load("experiments/pixel-classifier/cache/train.npz")
F, y, fr = d["F"][:, cols], d["y"], d["frame"]
use = y >= 0
rng = np.random.default_rng(0)
sub = rng.choice(np.nonzero(use & (fr < 900))[0], 300_000, replace=False)
va = rng.choice(np.nonzero(use & (fr >= 900))[0], 60_000, replace=False)
hgb = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=50,
                                     class_weight="balanced", random_state=0).fit(F[sub], y[sub])
p = hgb.predict_proba(F[va])[:, 1]
print(name, "val AUC %.4f AP %.4f" % (roc_auc_score(y[va], p), average_precision_score(y[va], p)))
pickle.dump({"clf": hgb, "thresh": float(sys.argv[3]) if len(sys.argv) > 3 else 0.9, "cols": cols,
             "features": [cc.FEATURES[i] for i in cols]}, open(f"models/cand_{name}.pkl", "wb"))
pi = permutation_importance(hgb, F[va], y[va], scoring="average_precision", n_repeats=3, random_state=0, n_jobs=1)
imp = sorted(zip([cc.FEATURES[i] for i in cols], pi.importances_mean, pi.importances_std), key=lambda r: -r[1])
for r in imp[:14]:
    print("  %-20s %.4f +- %.4f" % r)
json.dump([[a, float(b), float(c)] for a, b, c in imp], open(f"experiments/pixel-classifier/importance_{name}.json", "w"))
