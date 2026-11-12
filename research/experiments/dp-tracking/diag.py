"""diag.py - per-frame problems on TRAIN: false grooves, misses, big depth errors; and a picture of the worst.
    PYTHONPATH=. python experiments/dp-tracking/diag.py algos.dp_dense 200 out.png
"""
import importlib, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench import calib, datasets, run

def frame_problems(measure, it, cal):
    out = measure(it["on"], cal, None)
    tr = it["truth"]
    iv = [g for g in tr["grooves"] if g["in_view"]]
    pairs, extra = run.match_grooves(out["grooves"], iv)
    miss = sum(1 for t, p in pairs if t["floor_visible_px"] >= 3 and p is None)
    err = [p["depth"] - t["depth_mm"] for t, p in pairs if t["floor_visible_px"] >= 3 and p is not None]
    s = run.line_scores(np.asarray(out["line"]["u"], float), np.asarray(out["line"]["v"], float), tr["u"], tr["v"], 1.0)
    return {"false": len(extra), "miss": miss, "maxerr": max(np.abs(err), default=0.0), "f1": s["f1"], "prec": s["precision"],
            "rec": s["recall"], "out": out}

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1]); n = int(sys.argv[2])
    items = datasets.load_sim(sys.argv[4] if len(sys.argv) > 4 else "train")[:n]
    cal = calib.default_calib(); _, measure = run.resolve(mod)
    bad = []
    for i, it in enumerate(items):
        r = frame_problems(measure, it, cal)
        if r["false"] or r["miss"] or r["maxerr"] > 1.0:
            sc = it["truth"]["scene"]
            print(i, f"false {r['false']} miss {r['miss']} maxerr {r['maxerr']:.2f} f1 {r['f1']:.2f} P {r['prec']:.2f} R {r['rec']:.2f} | peak {sc['line_peak_dn']:.0f} amb {sc['ambient']:.2f} "
                  f"blur {sc['blur_px']:.1f} glints {len(sc['glints'])} gaps {len(sc['gaps'])} paint {bool(sc['paint'])} stones {len(sc['stones'])} spikes {sc['spikes']} zz {sc['zigzag_mm']}")
            bad.append((i, r))
    bad = bad[:16]
    fig, ax = plt.subplots(4, 4, figsize=(18, 13)); 
    for a in ax.ravel(): a.axis("off")
    for a, (i, r) in zip(ax.ravel(), bad):
        it = items[i]
        a.imshow(it["on"], cmap="gray", vmin=0, vmax=120)
        a.plot(it["truth"]["u"], it["truth"]["v"], ".", c="cyan", ms=0.6)
        a.plot(r["out"]["line"]["u"], r["out"]["line"]["v"], ".", c="red", ms=1.5)
        for g in r["out"]["grooves"]:
            a.axvspan(g["u0"], g["u1"], color="yellow", alpha=0.15); a.text(g["u0"], 20, f"{g['depth']:.1f}", color="yellow", fontsize=8)
        for g in it["truth"]["grooves"]:
            a.text(g["u0"], 230, f"{g['depth_mm']:.1f}" + ("" if g["floor_visible_px"] >= 3 else "x"), color="cyan", fontsize=8)
        a.set_title(f"{i}: false {r['false']} miss {r['miss']} err {r['maxerr']:.1f}", fontsize=9)
    fig.tight_layout(); fig.savefig(sys.argv[3], dpi=70, facecolor="white")
