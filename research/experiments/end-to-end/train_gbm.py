"""train_gbm.py - fit algos/e2e_gbm.py on `train` (first 1000 frames; last 200 = validation).

    python experiments/end-to-end/train_gbm.py
Compares a few feature sets / models on the validation split, then saves the best one
(refitted on the same 1000 training frames) to models/e2e_gbm.joblib.
"""
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())
from algos import e2e_gbm as M  # noqa: E402
from bench import datasets  # noqa: E402

items = datasets.load_sim("train")
y = np.array([min(g["depth_mm"] for g in it["truth"]["grooves"] if g["in_view"]) for it in items])
tr, va = slice(0, 1000), slice(1000, 1200)


def report(name, model, X):
    model.fit(X[tr], y[tr])
    p = model.predict(X[va])
    err = np.abs(p - y[va])
    acc = np.mean((p >= 1.6) == (y[va] >= 1.6))
    print(f"{name:28s} val MAE {err.mean():.3f} mm  P95 {np.percentile(err, 95):.3f}  verdict acc {acc:.3f}", flush=True)
    return err.mean()


def hgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                         loss="absolute_error", random_state=0)


def true_line(it):
    """The exact line from the simulator, one v per column: how good could step 2 be with a perfect step 1?"""
    t, v = it["truth"], np.full(M.W, np.nan)
    c = np.round(t["u"]).astype(int)
    for k in np.unique(c):
        v[k] = t["v"][c == k].max()
    v[np.isnan(v)] = np.nanmedian(v)
    return v, np.ones(M.W)


L = [M.cheap_line(it["on"]) for it in items]
F = {k: np.stack([M.describe(v, a, k) for v, a in L]) for k in ("quantiles", "profile", "full")}
for k, X in F.items():
    report("HGB " + k, hgb(), X)
report("MLP full", make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=0)), F["full"])
report("HGB full, TRUE line (oracle)", hgb(), np.stack([M.describe(*true_line(it), "full") for it in items]))
m = hgb().fit(F["full"][tr], y[tr])
joblib.dump(m, M.MODEL, compress=3)
print("saved HGB full:", os.path.getsize(M.MODEL) // 1024, "kB")
