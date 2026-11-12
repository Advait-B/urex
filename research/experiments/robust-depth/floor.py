"""floor.py - compare floor estimators.  python experiments/robust-depth/floor.py train 600
Prints MAE split by condition and writes per-groove records to _cache/floor_<set>.json (for figures.py)."""
import functools, json, os, sys
sys.path.insert(0, "experiments/robust-depth")
import numpy as np
import common as C
from algos import robust_depth as rd
C.memo_surface()

name = sys.argv[1]
limit = (int(sys.argv[2]) or None) if len(sys.argv) > 2 else None
rec = []
print(f"{'floor':12s} {'src':8s} {'MAE':>6s} {'med':>6s} {'P95':>6s} {'bias':>7s} | {'ideal':>6s} {'stress':>6s} | stone: MAE   bias    n | not-LB MAE  LB share")
for src in ("oracle", "baseline"):
    for fl in rd.FLOORS:
        r = C.score(functools.partial(rd.stage, floor=fl), name, src, limit)
        P = [(t, p, C.items(name)[i]["truth"]["scene"]) for t, p, i in C.pairs(name, r["_outs"], limit) if p and t["floor_visible_px"] >= 3]
        e = np.array([p["depth"] - t["depth_mm"] for t, p, _ in P])
        stone = np.array([any(abs(st["x"] - t["x_mm"]) < t["w_mm"] / 2 for st in s["stones"]) for t, _, s in P])
        stress = np.array([s["mode"] == "stress" for *_, s in P])
        lb = np.array([p["lower_bound"] for _, p, _ in P])
        a = lambda m: f"{np.abs(e[m]).mean():6.3f}" if m.any() else "   -  "
        print(f"{fl:12s} {src:8s} {a(e == e)} {np.median(np.abs(e)):6.3f} {np.percentile(np.abs(e), 95):6.3f} {e.mean():7.3f} | {a(~stress)} {a(stress & ~stone)} |"
              f"       {a(stone)} {e[stone].mean() if stone.any() else 0:6.3f} {stone.sum():4d} | {a(~lb)}      {lb.mean():.3f}", flush=True)
        rec += [{"floor": fl, "src": src, "true": t["depth_mm"], "pred": p["depth"], "sigma": p["sigma"], "lb": bool(p["lower_bound"]),
                 "stone": bool(s_), "stress": bool(st)} for (t, p, _), s_, st in zip(P, stone, stress)]
with open(os.path.join(C.CACHE, f"floor_{name}.json"), "w") as fh:
    json.dump(rec, fh)
