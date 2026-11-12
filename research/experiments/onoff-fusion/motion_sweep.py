"""motion_sweep.py - stress set with a moved / re-exposed off frame. Usage: motion_sweep.py out.json cond [cond ...]"""
import json
import sys

from variants import make
from perturb import perturbed_items
from bench import datasets, calib as calib_mod
from bench.run import eval_sim

CONDS = {"t0": (0, 0, 0), "t0.5": (0.5, 0, 0), "t1": (1, 0, 0), "t2": (2, 0, 0), "t3": (3, 0, 0), "t5": (5, 0, 0),
         "t1_r0.3": (1, 0.3, 0), "exp20": (0, 0, 0.2), "t2_r0.3_exp20": (2, 0.3, 0.2)}
VARS = [("single", "steger"), ("plain", "steger"), ("plain", "cog"), ("reg", "cog"), ("cons", "cog"), ("robust", "cog")]
items = datasets.load_sim("stress")
sunny = [i for i, it in enumerate(items) if it["truth"]["scene"]["ambient"] >= 0.55]
cal = calib_mod.default_calib()
res = {}
for c in sys.argv[2:]:
    pit = perturbed_items(items, *CONDS[c])
    for mode, det in VARS:
        if mode == "single" and c != "t0":
            continue
        m = make(mode, det)
        cache = {}

        def cm(g, cal_, off, m=m, cache=cache):
            if id(g) not in cache:
                cache[id(g)] = m(g, cal_, off)
            return cache[id(g)]
        r = eval_sim(cm, pit, cal)
        rs = eval_sim(cm, [pit[i] for i in sunny], cal)
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        r["sunny"] = {k: v for k, v in rs.items() if not k.startswith("_")}
        res[f"{c}|{mode}+{det}"] = r
        print(f"{c:14s} {mode + '+' + det:14s} F1 {r['line_f1']:.3f} MAE {r['depth_mae_mm']:.3f} P95 {r['depth_p95_mm']:.3f} "
              f"miss {r['miss_rate']:.3f} false {r['false_per_frame']:.3f} acc {r['verdict_acc']:.3f} fp {r['false_pass_rate']:.3f}"
              f" | sunny({len(sunny)}) MAE {rs['depth_mae_mm']:.3f} P95 {rs['depth_p95_mm']:.3f} acc {rs['verdict_acc']:.3f}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
