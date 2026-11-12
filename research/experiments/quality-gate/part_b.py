"""part_b.py - "can I trust this measurement?" Error predictor for the baseline, trained on sim `train` only.
    python experiments/quality-gate/part_b.py     (from research/, after build.py; ~1 min)
    -> models/risk_lr.json, cache/part_b.json, notes/figs/quality-gate/*.png

Target: BAD = some reported groove depth is wrong by more than 0.5 mm (a phantom groove counts as wrong).
Only frames where the baseline reports at least one groove are used ("answered"); the rest are already "retake".
Models (all settings fixed, nothing tuned on a test set):
  rule1   one feature: multi_frac (share of columns where the line has two branches > 3 px apart)
  tree    depth-3 decision tree = a plain if/else rule a person can read
  LR      logistic regression on the 30 standardised features, C = 1
  HGB     HistGradientBoosting, max_depth 3, defaults otherwise
The shipped threshold is chosen on out-of-fold predictions of `train` (5-fold): the largest coverage whose accepted
frames are BAD at most 10 % of the time. ideal / stress / legal are only scored.
"""
import json
import os
import sys

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.getcwd())
from algos import gate_features as gf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOL, TARGET_BAD = 0.5, 0.10


def load(n):
    d = np.load(os.path.join(HERE, "cache", n + ".npz"), allow_pickle=True)
    m = dict(zip(d["meta_cols"], d["meta"].T))
    return d["X"], m, d["errs"], d["err_frame"]


class Rule1(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.02).astype(int)

    def predict_proba(self, X):
        r = X[:, gf.ALL.index("multi_frac")]
        return np.c_[1 - r, r]


def models():
    return {"rule1 (multi_frac)": Rule1(),
            "tree depth 3": DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, random_state=0),
            "LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
            "HGB": HistGradientBoostingClassifier(max_depth=3, random_state=0)}


def at(keep, m, errs, ef):
    """metrics when the device answers only the frames in `keep` (bool per frame). Coverage is over ALL presses."""
    e = np.abs(errs[keep[ef]])
    n_v = int((m["true_pass"] >= 0).sum())
    fp = int((keep & (m["pred_pass"] == 1) & (m["true_pass"] == 0)).sum())
    return {"coverage": float(keep.mean()), "n_answered": int(keep.sum()), "n_errs": int(len(e)),
            "mae": float(e.mean()) if len(e) else None, "p95": float(np.percentile(e, 95)) if len(e) else None,
            "bad_rate": float((m["strict_worst"][keep] > TOL).mean()) if keep.any() else None,
            "false_pass": fp, "false_pass_rate": fp / n_v,
            "found_frac": float(m["n_found"][keep].sum() / max(1, m["n_meas"].sum()))}


def curve(risk, ans, m, errs, ef):
    order = np.argsort(np.where(ans, risk, np.inf), kind="stable")
    out = []
    for k in range(5, int(ans.sum()) + 1, 5):
        keep = np.zeros(len(ans), bool)
        keep[order[:k]] = True
        out.append(at(keep, m, errs, ef))
    return out


def main():
    X, m, _, _ = load("train")
    ans = m["n_pred"] > 0
    Xa, ya = X[ans], (m["strict_worst"][ans] > TOL).astype(int)
    print(f"train: {len(X)} frames, {ans.sum()} answered, {ya.mean():.3f} of those BAD")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    fitted, thr, res = {}, {}, {"train_oof": {}, "sets": {}}
    for name, mod in models().items():
        oof = cross_val_predict(mod, Xa, ya, cv=cv, method="predict_proba")[:, 1]
        o = np.argsort(oof, kind="stable")
        cum = np.cumsum(ya[o]) / np.arange(1, len(o) + 1)
        k = int(np.nonzero(cum <= TARGET_BAD)[0].max())            # largest prefix with <= 10 % BAD
        thr[name] = float(oof[o[k]])
        fitted[name] = mod.fit(Xa, ya)
        res["train_oof"][name] = {"auc": float(roc_auc_score(ya, oof)), "threshold": thr[name], "coverage_of_answered": (k + 1) / len(o)}
        print(f"  {name:20s} OOF AUC {roc_auc_score(ya, oof):.3f}  threshold {thr[name]:.3f} keeps {(k + 1) / len(o):.2f} of answered train frames")
    print(export_text(fitted["tree depth 3"], feature_names=gf.ALL))

    curves = {}
    for s in ("ideal", "stress", "legal"):
        Xs, ms, errs, ef = load(s)
        a = ms["n_pred"] > 0
        yb = ms["strict_worst"] > TOL
        res["sets"][s] = {"no gate": at(a, ms, errs, ef)}
        r0 = res["sets"][s]["no gate"]
        print(f"\n{s}: no gate      cov {r0['coverage']:.3f} MAE {r0['mae']:.3f} P95 {r0['p95']:.3f} BAD {r0['bad_rate']:.3f} false_pass {r0['false_pass']}")
        for name, mod in fitted.items():
            risk = mod.predict_proba(Xs)[:, 1]
            keep = a & (risk <= thr[name])
            r = at(keep, ms, errs, ef)
            r["auc"] = float(roc_auc_score(yb[a], risk[a])) if 0 < yb[a].sum() < a.sum() else None
            res["sets"][s][name] = r
            curves[(s, name)] = curve(risk, a, ms, errs, ef)
            # same coverage but random abstention = what the gate must beat is simply the no-gate row
            print(f"  {name:20s} cov {r['coverage']:.3f} MAE {r['mae']:.3f} P95 {r['p95']:.3f} BAD {r['bad_rate']:.3f} "
                  f"false_pass {r['false_pass']} found {r['found_frac']:.2f} AUC {r['auc'] if r['auc'] is None else round(r['auc'], 3)}")
        if s == "stress":
            for c in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
                row = min(curves[(s, "LR")], key=lambda q: abs(q["coverage"] - c))
                print(f"    LR stress @ coverage {row['coverage']:.2f}: MAE {row['mae']:.3f} P95 {row['p95']:.3f} BAD {row['bad_rate']:.3f} false_pass {row['false_pass']}")
    res["curves"] = {f"{s}|{n}": c for (s, n), c in curves.items()}

    # ---- which features matter: permutation importance of the shipped LR, on stress (drop in AUC)
    Xs, ms, _, _ = load("stress")
    a = ms["n_pred"] > 0
    imp = {}
    for name in ("LR", "HGB"):
        pi = permutation_importance(fitted[name], Xs[a], (ms["strict_worst"][a] > TOL).astype(int), scoring="roc_auc", n_repeats=30, random_state=0)
        imp[name] = {k: [float(pi.importances_mean[j]), float(pi.importances_std[j])] for j, k in enumerate(gf.ALL)}
    res["perm_importance_stress"] = imp
    print("\npermutation importance (AUC drop on stress), LR:", ", ".join(f"{k} {v[0]:+.3f}" for k, v in sorted(imp["LR"].items(), key=lambda t: -t[1][0])[:8]))
    print("HGB:", ", ".join(f"{k} {v[0]:+.3f}" for k, v in sorted(imp["HGB"].items(), key=lambda t: -t[1][0])[:8]))

    lr = fitted["LR"]
    json.dump({"features": gf.ALL, "mean": lr[0].mean_.tolist(), "scale": lr[0].scale_.tolist(), "coef": lr[1].coef_[0].tolist(),
               "intercept": float(lr[1].intercept_[0]), "threshold": thr["LR"], "target": "P(some reported depth wrong by > 0.5 mm)",
               "trained_on": f"sim train, {int(ans.sum())} answered frames"}, open(os.path.join("models", "risk_lr.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(HERE, "cache", "part_b.json"), "w"))


if __name__ == "__main__":
    main()
