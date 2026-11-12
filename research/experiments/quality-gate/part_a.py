"""part_a.py - "is this photo usable?" on the 86 real photos: honest model comparison.
    python experiments/quality-gate/part_a.py     (from research/, after build.py; ~4 min) -> cache/part_a.json

Protocols: LOO (86 folds) and leave-one-session-out (4 days; only 28 Oct and 30 Oct contain usable photos).
Hyper-parameters: everything is a FIXED, untuned setting written here before looking at the scores, except
  "LR C=0.03 (existing)"  whose C was picked on this very LOSO (optimistic), and
  "LR nested C"           whose C is picked inside each training fold (inner stratified 5-fold, log-loss).
The tiny CNN is too slow for 86 LOO fits on this machine: its "LOO" row is stratified 10-fold instead (stated in the table).
"+sim" = 43 simulated positives + 43 simulated negatives appended to every training fold (never to a test fold).
"""
import json
import os
import sys
import warnings

import numpy as np
import torch
from scipy.stats import beta
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
C = lambda n: np.load(os.path.join(HERE, "cache", n + ".npz"), allow_pickle=True)  # noqa: E731
NF = 12                                                         # the 12 frame features


def cp(k, n, a=0.05):
    """exact (Clopper-Pearson) 95 % interval for k successes out of n"""
    return (0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k)))


def sim_extra(seed=0, n=43):
    """Simulated training photos. Positive: laser-on frame with a normal laser (100-250 DN peak: outside that range the baseline is wrong on most frames) on which the
    baseline found a groove. Negative: half laser-OFF frames of the same scenes (sunlit tread, glints, paint but
    no line = the nearest thing the simulator has to "junk"), half frames with a very dim laser (< 55 DN)."""
    t = C("train")
    m = dict(zip(t["meta_cols"], t["meta"].T))
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.nonzero((m["line_peak_dn"] > 100) & (m["line_peak_dn"] < 250) & (m["n_pred"] > 0))[0], n, replace=False)
    dim = rng.choice(np.nonzero(m["line_peak_dn"] < 55)[0], n // 2, replace=False)
    off = rng.choice(len(t["Xoff"]), n - n // 2, replace=False)
    X = np.r_[t["X"][pos, :NF], t["X"][dim, :NF], t["Xoff"][off, :NF]]
    T = np.r_[t["thumbs"][pos], t["thumbs"][dim], t["thumbs_off"][off]]
    return X, np.r_[np.ones(n, int), np.zeros(n, int)], T


def models():
    sc = StandardScaler
    return {
        "LR C=0.03 (existing)": make_pipeline(sc(), LogisticRegression(C=0.03, class_weight="balanced", max_iter=2000)),
        "LR C=1 (untuned)": make_pipeline(sc(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        "LR nested C": make_pipeline(sc(), LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=5, scoring="neg_log_loss",
                                                               class_weight="balanced", max_iter=2000)),
        "random forest (existing)": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, min_samples_leaf=5,
                                                               class_weight="balanced", random_state=0),
        "SVM rbf C=1": make_pipeline(sc(), SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=0)),
        "k-NN k=5": make_pipeline(sc(), KNeighborsClassifier(5)),
    }


class Net(torch.nn.Module):
    """3 conv layers, 2.6 k weights, 48x48 grey thumbnail -> logit"""
    def __init__(self):
        super().__init__()
        L = torch.nn
        self.f = L.Sequential(L.Conv2d(1, 8, 3, padding=1), L.ReLU(), L.MaxPool2d(2), L.Conv2d(8, 16, 3, padding=1), L.ReLU(),
                              L.MaxPool2d(2), L.Conv2d(16, 16, 3, padding=1), L.ReLU(), L.AdaptiveMaxPool2d(1), L.Flatten(),
                              L.Dropout(0.3), L.Linear(16, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def cnn_fit_predict(Ttr, ytr, Tte, seed=0, epochs=40):
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=1e-3)
    x = torch.tensor(Ttr[:, None] / 255.0, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    pw = torch.tensor((len(ytr) - ytr.sum()) / max(1, ytr.sum()), dtype=torch.float32)
    for _ in range(epochs):
        for b in torch.randperm(len(x)).split(32):
            xb = x[b]
            if torch.rand(1) < 0.5:
                xb = xb.flip(3)                                    # mirror left-right: still a valid photo
            xb = xb * (0.8 + 0.4 * torch.rand(len(b), 1, 1, 1))    # exposure jitter
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(xb), y[b], pos_weight=pw).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Tte[:, None] / 255.0, dtype=torch.float32))).numpy()


def cv_scores(fit_predict, y, splits):
    p = np.zeros(len(y))
    for tr, te in splits:
        p[te] = fit_predict(tr, te)
    return p


def summarise(y, p, rng):
    d = p >= 0.5
    tp, fp, fn, tn = int((d & (y == 1)).sum()), int((d & (y == 0)).sum()), int((~d & (y == 1)).sum()), int((~d & (y == 0)).sum())
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], p[i]))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "acc": (tp + tn) / len(y), "precision": tp / max(1, tp + fp),
            "recall": tp / (tp + fn), "recall_ci": cp(tp, tp + fn), "fpr": fp / (fp + tn), "fpr_ci": cp(fp, fp + tn),
            "auc": float(roc_auc_score(y, p)), "auc_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main():
    r = C("real")
    X, y, T, sess = r["X"][:, :NF], r["y"], r["thumbs"], r["session"]
    Xs, ys, Ts = sim_extra()
    protos = {"LOO": list(LeaveOneOut().split(X)), "LOSO": list(LeaveOneGroupOut().split(X, y, sess))}
    rng = np.random.default_rng(0)
    out, preds = {}, {}
    rule = ((X[:, 2] >= 0.6) & (X[:, 4] >= 90) & (X[:, 3] >= 0.3)).astype(float)
    for pn, splits in protos.items():
        res = {"hand rule": summarise(y, rule, rng)}
        for name, m in models().items():
            for tag, ex in (("", None), (" +sim", (Xs, ys))):
                def fp(tr, te, m=m, ex=ex):
                    Xtr, ytr = (X[tr], y[tr]) if ex is None else (np.r_[X[tr], ex[0]], np.r_[y[tr], ex[1]])
                    return clone(m).fit(Xtr, ytr).predict_proba(X[te])[:, 1]
                p = cv_scores(fp, y, splits)
                res[name + tag] = summarise(y, p, rng); preds[f"{pn}|{name}{tag}"] = p.tolist()
        for tag, ex in (("", None), (" +sim", (Ts, ys))):
            ps = []
            cs = splits if pn == "LOSO" else list(StratifiedKFold(10, shuffle=True, random_state=0).split(X, y))
            for seed in range(3 if pn == "LOSO" else 1):           # the net is seed-dependent: average 3 on LOSO
                def fp(tr, te, ex=ex, seed=seed):
                    Ttr, ytr = (T[tr], y[tr]) if ex is None else (np.r_[T[tr], ex[0]], np.r_[y[tr], ex[1]])
                    return cnn_fit_predict(Ttr, ytr, T[te], seed)
                ps.append(cv_scores(fp, y, cs))
            p = np.mean(ps, 0)
            res["tiny CNN 48x48" + tag] = summarise(y, p, rng); preds[f"{pn}|tiny CNN 48x48{tag}"] = p.tolist()
            print(pn, "cnn" + tag, "done", flush=True)
        out[pn] = res
        print(f"\n{pn}\n{'model':32s} acc  prec  recall [95% CI]      FPR [95% CI]        AUC [95% CI]      tp fp fn tn")
        for k, s in res.items():
            print(f"{k:32s} {s['acc']:.2f} {s['precision']:.2f}  {s['recall']:.2f} [{s['recall_ci'][0]:.2f},{s['recall_ci'][1]:.2f}]  "
                  f"{s['fpr']:.2f} [{s['fpr_ci'][0]:.2f},{s['fpr_ci'][1]:.2f}]  {s['auc']:.3f} [{s['auc_ci'][0]:.2f},{s['auc_ci'][1]:.2f}]  "
                  f"{s['tp']:2d} {s['fp']:2d} {s['fn']:2d} {s['tn']:2d}", flush=True)
    json.dump({"scores": out, "preds": preds, "y": y.tolist(), "names": r["names"].tolist()},
              open(os.path.join(HERE, "cache", "part_a.json"), "w"))


if __name__ == "__main__":
    main()
