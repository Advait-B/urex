"""uncertainty.py - is the per-groove sigma calibrated?  python experiments/robust-depth/uncertainty.py train 600
Coverage of the 68 % / 95 % intervals (depth +- 1 / 1.96 sigma) for grooves with a visible floor that the
stage did not flag as lower bounds. Writes _cache/unc_<set>.json for figures.py."""
import functools, json, os, sys
sys.path.insert(0, "experiments/robust-depth")
import numpy as np
import common as C
from algos import robust_depth as rd
C.memo_surface()

rec = {}
for name in sys.argv[1].split(","):
    limit = (int(sys.argv[2]) or None) if len(sys.argv) > 2 else None
    for src in ("oracle", "baseline"):
        r = C.score(functools.partial(rd.stage, floor="dbscan", detect="shape", lo_k=3.5), name, src, limit)
        P = [(t, p) for t, p, i in C.pairs(name, r["_outs"], limit) if p and t["floor_visible_px"] >= 3 and not p["lower_bound"]]
        e = np.array([p["depth"] - t["depth_mm"] for t, p in P]); s = np.array([p["sigma"] for t, p in P])
        z = np.abs(e) / s
        print(f"{name:12s} {src:8s} n={len(z):4d} cover68={np.mean(z < 1):.3f} cover95={np.mean(z < 1.96):.3f} |z|>5: {np.mean(z > 5):.3f} "
              f"median sigma={np.median(s):.3f} median|e|={np.median(np.abs(e)):.3f}  scale for 95%: {np.percentile(z, 95) / 1.96:.2f}", flush=True)
        rec[f"{name}/{src}"] = {"e": e.round(4).tolist(), "s": s.round(4).tolist(), "d": [round(t["depth_mm"], 3) for t, p in P]}
with open(os.path.join(C.CACHE, f"unc_{sys.argv[1].replace(',', '_')}.json"), "w") as fh:
    json.dump(rec, fh)
