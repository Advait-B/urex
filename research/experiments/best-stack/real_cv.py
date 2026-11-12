"""real_cv.py - honest real-photo numbers for the gated stack: the frame gate is re-fitted WITHOUT the photos it is
scored on (leave-one-session-out by date, and leave-one-out), same model as models/frame_lr.json (StandardScaler +
logistic regression, C = 0.03, class_weight balanced, the 12 frame features). The stack itself has no real-photo
training. Metrics as bench/run2.eval_real2.     python experiments/best-stack/real_cv.py   (from research/)"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())
from algos import gate_features as gf, stackcommon as sc  # noqa: E402
from algos.dp_dense import extract_line  # noqa: E402
from bench import datasets  # noqa: E402

items = datasets.load_real()
X = np.array([[gf.features(it["gray"])[0][k] for k in gf.FRAME] for it in items])
y = np.array([it["usable"] for it in items])
laser = np.array([it["laser"] for it in items])
has = np.array([bool(sc.measure(it["gray"], None, None, line_fn=extract_line, depth="rd", cap=True)["grooves"]) for it in items])
day = np.array([it["name"].split("_")[1] for it in items])


def report(name, p):
    use = (p >= 0.5) & has                       # the gated stack answers: gate says yes AND a groove was found
    nl, la = (y == 0) & (laser == 0), (y == 0) & (laser == 1)
    print(f"{name:28s} usable_recall {use[y == 1].sum()}/{(y == 1).sum()} = {use[y == 1].mean():.3f}   false_measure_nolaser "
          f"{use[nl].sum()}/{nl.sum()} = {use[nl].mean():.3f}   false_measure_laser {use[la].sum()}/{la.sum()} = {use[la].mean():.3f}")


def fit(tr):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", max_iter=2000)).fit(X[tr], y[tr])


report("no gate (stack finds a groove)", np.ones(len(y)))
report("in-sample (all 86)", fit(np.ones(len(y), bool)).predict_proba(X)[:, 1])
p = np.zeros(len(y))
for d in np.unique(day):
    te = day == d
    p[te] = fit(~te).predict_proba(X[te])[:, 1] if len(np.unique(y[~te])) == 2 else 0.0
report("leave-one-session-out", p)
p = np.zeros(len(y))
for i in range(len(y)):
    tr = np.arange(len(y)) != i
    p[i] = fit(tr).predict_proba(X[i:i + 1])[0, 1]
report("leave-one-out", p)
print("sessions:", {d: (int((day == d).sum()), int(y[day == d].sum())) for d in np.unique(day)})
