"""detect.py - shallow-groove detection and the PASS/FAIL verdict.
    python experiments/robust-depth/detect.py uneven_tune          (tuning)   /   uneven_test, legal, ideal (reporting)
Prints harness verdict numbers for each rule and the miss rate by true depth."""
import functools, sys
sys.path.insert(0, "experiments/robust-depth")
import numpy as np
import common as C
from algos import robust_depth as rd
from bench import depth
C.memo_surface()

name = sys.argv[1]
limit = (int(sys.argv[2]) or None) if len(sys.argv) > 2 else None
BINS = [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 1.6), (1.6, 5), (5, 20)]
print(f"{'rule':30s} {'src':8s} verdict falsePASS falseFAIL  miss false/fr | miss by true depth " + " ".join(f"{a}-{b}" for a, b in BINS))
for src in ("oracle", "baseline"):
    rules = {"bench default": depth.measure_from_line, "dbscan + default rule": functools.partial(rd.stage, floor="dbscan")}
    for k in (sys.argv[3:] or ["2.5"]):
        rules[f"dbscan + shape rule k={k}"] = functools.partial(rd.stage, floor="dbscan", detect="shape", lo_k=float(k))
    for tag, st in rules.items():
        r = C.score(st, name, src, limit)
        P = [(t, p) for t, p, i in C.pairs(name, r["_outs"], limit) if t["floor_visible_px"] >= 3]
        d = np.array([t["depth_mm"] for t, p in P]); miss = np.array([p is None for t, p in P])
        mb = " ".join(f"{miss[(d >= a) & (d < b)].mean():5.2f}" if ((d >= a) & (d < b)).any() else "   - " for a, b in BINS)
        ff = 1 - r["verdict_acc"] - r["false_pass_rate"]
        print(f"{tag:30s} {src:8s} {r['verdict_acc']:7.3f} {r['false_pass_rate']:9.3f} {ff:9.3f} {r['miss_rate']:5.3f} {r['false_per_frame']:8.3f} | {mb}", flush=True)
