"""harness_checks.py - evidence for the 'things to know about the harness / baseline' section of the notes.
    PYTHONPATH=. python experiments/dp-tracking/harness_checks.py
"""
import warnings
import numpy as np
from scipy.spatial import cKDTree
from bench import calib, datasets, run
from algos import baseline_steger, dp_dense

warnings.simplefilter("ignore")
cal = calib.default_calib()

# 1. How much of the TRUE line is too dim to see? (truth["mask"] = pixels where the clean line image is
#    above max(8 DN, 25 % of the line peak)).  Truth points outside it still count in line_f1's recall.
for name in ("ideal", "stress", "legal"):
    items = datasets.load_sim(name)
    share = []
    for it in items:
        t = it["truth"]
        vis = t["mask"][np.clip(np.round(t["v"]).astype(int), 0, 239), np.clip(np.round(t["u"]).astype(int), 0, 319)]
        Ts = np.unique(np.round(np.c_[t["u"], t["v"]] * 2) / 2, axis=0)          # the scorer's 0.5 px resampling
        vis_s = t["mask"][np.clip(np.round(Ts[:, 1]).astype(int), 0, 239), np.clip(np.round(Ts[:, 0]).astype(int), 0, 319)]
        share.append(1 - vis_s.mean())
    print(f"1. {name}: share of scored truth samples that are dimmer than 25 % of the line peak: mean {np.mean(share):.3f}, max {np.max(share):.3f}")

# 2. false grooves: phantom (no true groove there) or a true groove reported twice (split by the depth stage)?
for mod in (baseline_steger, dp_dense):
    _, measure = run.resolve(mod)
    for name in ("ideal", "stress", "legal"):
        split = phantom = 0
        for it in datasets.load_sim(name):
            out = measure(it["on"], cal, None)
            tr = [g for g in it["truth"]["grooves"] if g["in_view"]]
            _, extra = run.match_grooves(out["grooves"], tr)
            for e in extra:
                inside = any(min(e["u1"], g["u1"] + 4) - max(e["u0"], g["u0"] - 4) > 0 for g in it["truth"]["grooves"])
                split += inside; phantom += not inside
        print(f"2. {mod.NAME} {name}: false grooves = {split + phantom}: {split} overlap a true groove (duplicate / out-of-view groove), {phantom} are phantoms")

# 3. Steger's 'sub-pixel offset within +-0.5 px in both axes' test drops points when the line centre sits
#    half-way between two rows. Count path columns (dp_dense path, ideal set) with NO pixel passing the test.
n = bad = 0
for it in datasets.load_sim("ideal")[:60]:
    t = dp_dense.track(it["on"])
    cols = np.nonzero(t["on"])[0]
    ok = np.zeros(len(cols), bool)
    for k in (-1, 0, 1):
        r = np.clip(t["path"][cols] + k, 0, 239)
        ok |= (np.abs(t["du"][r, cols]) <= 0.5) & (np.abs(t["dv"][r, cols]) <= 0.5) & (t["reward"][r, cols] > 0.2)
    n += len(cols); bad += (~ok).sum()
print(f"3. ideal[:60]: {bad} of {n} line columns ({bad / n:.1%}) have no pixel that passes the +-0.5 px test")

# 4. real references: agreement per frame, split by photo day
_, measure = run.resolve(dp_dense)
for it in datasets.load_real():
    if it["ref"] is None:
        continue
    out = measure(it["gray"], None, None)
    s = run.line_scores(np.asarray(out["line"]["u"], float), np.asarray(out["line"]["v"], float), it["ref"]["u"], it["ref"]["v"], 1.5)
    print(f"4. {it['name']} {it['ref']['mode']:7s} ref points {len(it['ref']['u']):4d}  dp_dense P {s['precision']:.2f} R {s['recall']:.2f} F1 {s['f1']:.2f}")
