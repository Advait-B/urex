"""surface.py - does the surface model matter?  python experiments/robust-depth/surface.py [set] [limit]
Depth error (floor=dbscan, bias-corrected) for each surface model, split by rocking and by the share of
the view taken up by grooves."""
import functools, sys, time
sys.path.insert(0, "experiments/robust-depth")
import numpy as np
import common as C
from algos import robust_depth as rd
C.memo_surface()

name = sys.argv[1] if len(sys.argv) > 1 else "train"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
print(f"{'surface':12s} {'src':8s} {'MAE':>6s} {'med':>6s} {'P95':>6s} | rock<0.5 rock>1  | fill<.2 fill>.3 | miss  ms")
for src in ("oracle", "baseline"):
    for sf in (sys.argv[3:] or ("ransac", "huber", "theilsen", "ransac_irls")):
        t0 = time.time()
        r = C.score(functools.partial(rd.stage, surface=sf, floor="dbscan"), name, src, limit)
        ms = 1000 * (time.time() - t0) / r["n_frames"]
        P = [(t, p, C.items(name)[i]["truth"]) for t, p, i in C.pairs(name, r["_outs"], limit) if p and t["floor_visible_px"] >= 3
             and not p["lower_bound"]]
        e = np.abs([p["depth"] - t["depth_mm"] for t, p, _ in P])
        rock = np.array([max(abs(tr["scene"]["tilt_x_deg"]), abs(tr["scene"]["tilt_y_deg"])) for *_, tr in P])
        fill = np.array([sum(g["u1"] - g["u0"] for g in tr["grooves"]) / 320 for *_, tr in P])
        f = lambda m: f"{e[m].mean():7.3f}" if m.any() else "    -  "
        print(f"{sf:12s} {src:8s} {e.mean():6.3f} {np.median(e):6.3f} {np.percentile(e, 95):6.3f} | {f(rock < 0.5)} {f(rock > 1)} |"
              f" {f(fill < 0.2)} {f(fill > 0.3)} | {r['miss_rate']:.3f} {ms:4.0f}", flush=True)
