"""frame_classifier.py - ML: is this photo usable for a depth measurement?

Most photos a hand-held device takes are no good: no laser, laser too close
and blurred, reflections, the line off the tyre. A hand-written rule ("line
covers 60 % of rows and is bright") breaks easily. The 30 Oct tread_depth.py
check had only a 1.26x vs 1.43x margin. So we LEARN the decision from labelled
photos.

  features   12 numbers per photo measured by single_frame.py: how many
             laser points, how much of the frame they cover, how well they
             fit one surface, how bright/sharp/thick the line is, saturation,
             background brightness, image sharpness...
  labels     data/photo_labels.csv: 'usable' = 1 if a person looking at the
             photo would measure it (19 of 86 photos).
  models     logistic regression (12 weights; simple enough to run on the
             Nucleo later) and a random forest for comparison, plus the old
             hand rule as a baseline.
  testing    leave-one-out (train on 85, test on the 1 left out, repeat 86
             times) and leave-one-SESSION-out (train on the other days, test
             on a whole unseen day). The second is the honest one: a new day
             means new lighting and a new tyre.

    python frame_classifier.py PHOTO_FOLDER          train + evaluate + save model.json
    python frame_classifier.py PHOTO_FOLDER --report  also print every photo's score

The saved model (data/frame_model.json) is just the mean/scale/weights, so
predict() needs only numpy.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

import single_frame as sf

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "data", "photo_labels.csv")
MODEL = os.path.join(HERE, "data", "frame_model.json")

FEATURES = ["log_points", "keep_frac", "row_coverage", "inlier_frac", "mean_contrast", "mean_strength",
            "line_width_px", "sat_frac", "n_pieces", "bg_mean", "log_sharpness", "longest_run_frac"]


def features(gray, res=None):
    """12 numbers describing the photo and its laser line."""
    if res is None:
        res = sf.analyse(gray)
    h, w = gray.shape
    x, y = res["_x"], res["_y"]
    th = sf.tophat(gray, sf.DEFAULTS["tophat"])
    widths, sat = [], 0
    for xi, yi in zip(x[::3], y[::3]):            # every 3rd point is plenty
        r, c = int(round(yi)), int(round(xi))
        if not (0 <= r < h and 0 <= c < w):
            continue
        row = th[r, max(0, c - 15): c + 16]
        widths.append(float((row > 0.5 * th[r, c]).sum()) if th[r, c] > 0 else 0.0)
        sat += gray[r, c] >= 250
    rows = np.unique(np.clip(np.round(y).astype(int), 0, h - 1)) if len(y) else np.array([], int)
    longest = 0
    if len(rows):
        runs = np.split(rows, np.nonzero(np.diff(rows) > 3)[0] + 1)
        longest = max(len(r) for r in runs)
    return {
        "log_points": float(np.log1p(res["n_points"])),
        "keep_frac": float(res["n_points"] / max(1, res["n_points_all"])),
        "row_coverage": float(res["row_coverage"]),
        "inlier_frac": float(res["surface"]["inlier_frac"]) if res["surface"] else 0.0,
        "mean_contrast": float(res["mean_contrast"]),
        "mean_strength": float(res["mean_strength"]),
        "line_width_px": float(np.median(widths)) if widths else 0.0,
        "sat_frac": float(sat / max(1, len(widths))),
        "n_pieces": float(res["n_pieces"]),
        "bg_mean": float(gray.mean()),
        "log_sharpness": float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
        "longest_run_frac": float(longest / h),
    }


def rule_baseline(f):
    """the kind of hand rule used before: line covers most rows, bright, one surface"""
    return int(f["row_coverage"] >= 0.6 and f["mean_contrast"] >= 90 and f["inlier_frac"] >= 0.3)


def load_dataset(folder):
    rows = list(csv.DictReader(open(LABELS)))
    X, y, names, sessions, cache = [], [], [], [], {}
    for r in rows:
        p = os.path.join(folder, r["file"])
        if not os.path.exists(p):
            continue
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        res = sf.analyse(g)
        f = features(g, res)
        X.append([f[k] for k in FEATURES])
        y.append(int(r["usable"]))
        names.append(r["file"])
        sessions.append(r["file"].split("_")[1])          # photo_YYYYMMDD_...
        cache[r["file"]] = (f, res)
    return np.array(X), np.array(y), names, np.array(sessions), cache


def make_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        # C = 0.03 (strong regularisation): picked because it generalised best to a held-out
        # DAY (C = 1: precision 0.60, 0.3: 0.69, 0.1: 0.72, 0.03: 0.82). Chosen on the same
        # cross-validation it is scored on, so treat its score as slightly optimistic.
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced",
                                                                                  max_iter=2000)),
        "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                                random_state=0),
    }


def scores(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    return {"accuracy": (tp + tn) / len(y), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(X, y, sessions, feats):
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
    out = {}
    base = np.array([rule_baseline(f) for f in feats])
    out["hand rule (baseline)"] = {"LOO": scores(y, base), "leave-one-session-out": scores(y, base)}
    for name, m in make_models().items():
        p_loo = cross_val_predict(clone(m), X, y, cv=LeaveOneOut())
        # leave-one-session-out: a session with only one class still gets predicted by a
        # model trained on the others (that's the point: new day, unseen)
        p_lso = np.zeros_like(y)
        for tr, te in LeaveOneGroupOut().split(X, y, sessions):
            if len(np.unique(y[tr])) < 2:
                p_lso[te] = y[tr][0]
                continue
            p_lso[te] = clone(m).fit(X[tr], y[tr]).predict(X[te])
        out[name] = {"LOO": scores(y, p_loo), "leave-one-session-out": scores(y, p_lso), "_pred_loo": p_loo}
    return out


def fit_and_save(X, y, path=MODEL):
    m = make_models()["logistic regression"].fit(X, y)
    sc, lr = m.steps[0][1], m.steps[1][1]
    model = {"features": FEATURES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
             "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), "threshold": 0.5,
             "trained_on": int(len(y)), "positives": int(y.sum())}
    with open(path, "w") as fh:
        json.dump(model, fh, indent=2)
    return model


def predict(f, model=None):
    """P(usable) for one photo's feature dict, using only numpy."""
    if model is None:
        with open(MODEL) as fh:
            model = json.load(fh)
    v = (np.array([f[k] for k in model["features"]]) - np.array(model["mean"])) / np.array(model["scale"])
    z = float(v @ np.array(model["coef"]) + model["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with the labelled photos")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    X, y, names, sessions, cache = load_dataset(args.folder)
    feats = [cache[n][0] for n in names]
    print(f"{len(y)} photos, {y.sum()} usable, sessions {sorted(set(sessions))}\n")
    ev = evaluate(X, y, sessions, feats)
    print(f"{'model':24s} {'test':24s} {'acc':>5s} {'prec':>5s} {'recall':>6s}   tp fp fn tn")
    for name, d in ev.items():
        for k in ("LOO", "leave-one-session-out"):
            s = d[k]
            print(f"{name:24s} {k:24s} {s['accuracy']:5.2f} {s['precision']:5.2f} {s['recall']:6.2f}   "
                  f"{s['tp']:2d} {s['fp']:2d} {s['fn']:2d} {s['tn']:2d}")
    model = fit_and_save(X, y)
    print(f"\nsaved logistic-regression model -> {MODEL}")
    w = sorted(zip(FEATURES, model["coef"]), key=lambda t: -abs(t[1]))
    print("biggest weights (standardised):", ", ".join(f"{k} {v:+.2f}" for k, v in w[:5]))
    if args.report:
        p_loo = ev["logistic regression"]["_pred_loo"]
        for n, yy, pp in zip(names, y, p_loo):
            flag = "" if yy == pp else "  <-- wrong (LOO)"
            print(f"{n:30s} label {yy} pred {pp}{flag}")


if __name__ == "__main__":
    main()
