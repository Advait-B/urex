"""build.py - candidates + features + labels for the train set -> cache/train.npz
label 1 = within 1 px of the true line (the harness tolerance), -1 = 1..2 px (ignored), 0 = clutter."""
import os, sys, time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.getcwd())
from bench import datasets
from algos import cand_clf as cc

name = sys.argv[1] if len(sys.argv) > 1 else "train"
items = datasets.load_sim(name)
Fs, ys, fr, uv, rec = [], [], [], [], []
t0 = time.time()
for i, it in enumerate(items):
    c = cc.candidates(it["on"]); F = cc.features(c)
    tr = it["truth"]
    T = np.c_[tr["u"], tr["v"]]
    if len(F) and len(T):
        d, _ = cKDTree(T).query(np.c_[c["u"], c["v"]])
    else:
        d = np.full(len(F), 99.0)
    y = np.where(d < 1.0, 1, np.where(d < 2.0, -1, 0))
    if len(T):
        Ts = np.unique(np.round(T * 2) / 2, axis=0)
        rec.append(np.mean(cKDTree(np.c_[c["u"], c["v"]]).query(Ts)[0] < 1.0) if len(F) else 0.0)
    Fs.append(F); ys.append(y); fr.append(np.full(len(F), i)); uv.append(np.c_[c["u"], c["v"]])
print(name, "frames", len(items), "cands", sum(map(len, ys)), "pos frac", np.mean(np.concatenate(ys) == 1),
      "candidate recall of truth", np.mean(rec), "ms/frame", (time.time() - t0) / len(items) * 1000)
np.savez_compressed(f"experiments/pixel-classifier/cache/{name}.npz", F=np.concatenate(Fs), y=np.concatenate(ys),
                    frame=np.concatenate(fr), uv=np.concatenate(uv))
