"""fit_frame.py - fit the shipped frame gate (logistic regression, C = 0.03, the 12 frame features) on ALL 86
labelled real photos -> models/frame_lr.json. Scoring this model on the same 86 photos is IN-SAMPLE."""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
r = np.load(os.path.join(HERE, "cache", "real.npz"), allow_pickle=True)
names = json.load(open(os.path.join("models", "risk_lr.json")))["features"][:12]
X, y = r["X"][:, :12], r["y"]
sc = StandardScaler().fit(X)
lr = LogisticRegression(C=0.03, class_weight="balanced", max_iter=2000).fit(sc.transform(X), y)
json.dump({"features": names, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
           "intercept": float(lr.intercept_[0]), "threshold": 0.5, "trained_on": "86 real photos (19 usable), all labels"},
          open(os.path.join("models", "frame_lr.json"), "w"), indent=1)
p = lr.predict_proba(sc.transform(X))[:, 1] >= 0.5
print("in-sample: recall", (p & (y == 1)).sum() / y.sum(), "false positives", int((p & (y == 0)).sum()), "of", int((y == 0).sum()))
