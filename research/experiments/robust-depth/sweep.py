"""sweep.py - score stage variants on a set.  python experiments/robust-depth/sweep.py train 300 floor"""
import sys, time, functools
sys.path.insert(0, "experiments/robust-depth")
import common as C
from algos import robust_depth as rd
from bench import depth
C.memo_surface()

KEYS = ["depth_mae_mm", "depth_p95_mm", "depth_bias_mm", "miss_rate", "false_per_frame", "verdict_acc", "false_pass_rate"]


def row(tag, stage, name, src, limit):
    t = time.time()
    r = C.score(stage, name, src, limit)
    print(f"{tag:34s} {src:8s} {name:6s} " + " ".join(f"{r[k]:7.3f}" for k in KEYS) + f"  {1000*(time.time()-t)/r['n_frames']:5.1f}ms", flush=True)
    return r


if __name__ == "__main__":
    name, limit, what = sys.argv[1], int(sys.argv[2]) or None, sys.argv[3]
    print(" " * 51 + " ".join(f"{k[6:13] if k.startswith('depth') else k[:7]:>7s}" for k in KEYS))
    for src in ("oracle", "baseline"):
        row("bench default", depth.measure_from_line, name, src, limit)
        if what == "floor":
            for f in rd.FLOORS:
                row(f"floor={f}", functools.partial(rd.stage, floor=f, correct=False), name, src, limit)
        if what == "surface":
            for s in ("ransac", "huber", "theilsen", "ransac_irls"):
                row(f"surface={s}", functools.partial(rd.stage, surface=s, correct=False), name, src, limit)
        if what == "correct":                      # bench stage (band floor) + each correction term
            row("no correction", functools.partial(rd.stage, correct=False), name, src, limit)
            row("curvature term only", functools.partial(rd.stage, lateral=False, parallax=False), name, src, limit)
            row("parallax (u-shift) only", functools.partial(rd.stage, lateral=False, curvature=False), name, src, limit)
            row("parallax (u-shift) + curvature", functools.partial(rd.stage, lateral=False), name, src, limit)
            row("lateral-X fit only", functools.partial(rd.stage, curvature=False), name, src, limit)
            for R in (384, 480, 576):
                row(f"lateral-X fit + curvature R={R}", functools.partial(rd.stage, tyre_R=R), name, src, limit)
        if what == "detect":
            row("dbscan, detect=default", functools.partial(rd.stage, floor="dbscan"), name, src, limit)
            row("dbscan, detect=shape", functools.partial(rd.stage, floor="dbscan", detect="shape"), name, src, limit)
