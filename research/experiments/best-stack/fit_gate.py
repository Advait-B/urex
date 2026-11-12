"""fit_gate.py - on sim `train` ONLY: (1) what the lower_bound flag is worth, (2) re-fit the risk gate for the stack.
    python experiments/best-stack/fit_gate.py        (after build.py train)  -> models/stack_risk_lr.json
Protocol copied from experiments/quality-gate/part_b.py: logistic regression (C = 1, fixed), threshold = the largest
coverage at which 5-fold out-of-fold predictions on `train` are BAD at most TARGET_BAD of the time. part_b used 10 %;
this stack is already at 6.5 % BAD on train with no gate, so 10 % would keep every frame. TARGET_BAD = 3 % (about half
the ungated rate) was chosen on train before any test set was scored with the gate.
BAD frame = a phantom groove, OR a groove with a seen floor wrong by > 0.5 mm, OR a lower-bound groove that reads
more than 0.5 mm DEEPER than the truth (">= x" is then false), OR an unsafe PASS."""
import json
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())
from algos import stackcommon as sc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOL, TARGET_BAD = 0.5, 0.03


def load(name):
    return pickle.load(open(os.path.join(HERE, "cache", name + ".pkl"), "rb"))


def bad(r, tol=TOL):
    if r["n_phantom"] or (r["pred_pass"] and r["has_view"] and not r["true_pass"]):
        return True
    for g in r["grooves"]:
        if g["pred"] is None:
            continue
        e = g["pred"] - g["true"]
        if (e > tol) if g["lower_bound"] else (abs(e) > tol and g["measurable"]):
            return True
    return False


def answered(r):
    return r["n_pred"] > 0 and r["verdict"] != "RETAKE"


def flag_table(rows, label):
    e = np.array([(g["pred"] - g["true"], g["lower_bound"]) for r in rows for g in r["grooves"] if g["pred"] is not None and g["measurable"]])
    for name, m in (("seen floor", e[:, 1] == 0), ("lower_bound", e[:, 1] == 1)):
        a = np.abs(e[m, 0])
        if not len(a):
            print(f"  {label} {name:11s} n=   0")
            continue
        print(f"  {label} {name:11s} n={m.sum():4d}  MAE {a.mean():.3f}  median {np.median(a):.3f}  P95 {np.percentile(a, 95):.3f}  "
              f">1mm {np.mean(a > 1):.3f}  reads too deep by >0.5: {np.mean(e[m, 0] > 0.5):.3f}  too shallow by >0.5: {np.mean(e[m, 0] < -0.5):.3f}")
    v = [r["verdict"] for r in rows if r["has_view"]]
    naive_fp = sum(r["pred_pass"] and not r["true_pass"] for r in rows if r["has_view"])
    retake = [r for r in rows if r["has_view"] and r["verdict"] == "RETAKE"]
    print(f"  {label} verdicts: " + ", ".join(f"{k} {v.count(k)}" for k in ("PASS", "FAIL", "RETAKE", "NONE")) +
          f" | of the RETAKE frames, truly PASS: {sum(r['true_pass'] for r in retake)}/{len(retake)} | unsafe PASS: {naive_fp}")


def main():
    rows = load("train")
    print(f"train: {len(rows)} frames; frames with a capped (> 25 mm) reading: {sum(r['n_capped'] > 0 for r in rows)}")
    flag_table(rows, "train")
    ans = np.array([answered(r) for r in rows])
    X = np.array([[r["f"][k] for k in sc.FEATURES] for r in rows])
    y = np.array([bad(r) for r in rows]).astype(int)
    Xa, ya = X[ans], y[ans]
    print(f"answered {ans.sum()}, BAD share of answered {ya.mean():.3f}")
    mod = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    oof = cross_val_predict(mod, Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    o = np.argsort(oof, kind="stable")
    cum = np.cumsum(ya[o]) / np.arange(1, len(o) + 1)
    k = int(np.nonzero(cum <= TARGET_BAD)[0].max())
    thr = float(oof[o[k]])
    print(f"LR: OOF AUC {roc_auc_score(ya, oof):.3f}; threshold {thr:.3f} keeps {(k + 1) / len(o):.3f} of answered train frames "
          f"(BAD among kept {cum[k]:.3f}, OOF)")
    for t in (0.02, 0.05, 0.10):
        kk = np.nonzero(cum <= t)[0]
        print(f"   (for reference: BAD <= {t:.2f} would keep {(kk.max() + 1) / len(o) if len(kk) else 0:.3f})")
    mod.fit(Xa, ya)
    lr = mod[1]
    json.dump({"features": sc.FEATURES, "mean": mod[0].mean_.tolist(), "scale": mod[0].scale_.tolist(), "coef": lr.coef_[0].tolist(),
               "intercept": float(lr.intercept_[0]), "threshold": thr, "target": "P(frame BAD), see experiments/best-stack/fit_gate.py",
               "trained_on": f"sim train, {int(ans.sum())} answered frames, stack = dp_dense + rd_final + cap + lower-bound rule"},
              open(os.path.join("models", "stack_risk_lr.json"), "w"), indent=1)
    w = sorted(zip(sc.FEATURES, lr.coef_[0]), key=lambda t: -abs(t[1]))
    print("largest standardised weights:", ", ".join(f"{k} {c:+.2f}" for k, c in w[:8]))


if __name__ == "__main__":
    main()
