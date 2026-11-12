"""tune.py - score a module on a slice of TRAIN with parameter overrides (never on the test sets).
    PYTHONPATH=. python experiments/dp-tracking/tune.py algos.dp_dense 150 J=0.5,1,2 A=0.1,0.2
"""
import importlib, itertools, sys
import numpy as np
from bench import calib, datasets, run

def score(mod, items, cal):
    _, measure = run.resolve(mod)
    r = run.eval_sim(measure, items, cal)
    return (f"F1 {r['line_f1']:.3f} rmse {r['line_rmse_px']:.3f} mae {r['depth_mae_mm']:.3f} p95 {r['depth_p95_mm']:.3f} "
            f"miss {r['miss_rate']:.3f} false {r['false_per_frame']:.3f} verd {r['verdict_acc']:.3f} ms {r['ms_per_frame']:.1f}")

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1])
    n = int(sys.argv[2])
    items = datasets.load_sim("train")[:n]
    cal = calib.default_calib()
    grid = {a.split("=")[0]: [float(x) for x in a.split("=")[1].split(",")] for a in sys.argv[3:]}
    base = dict(mod.P) if hasattr(mod, "P") else {}
    for combo in itertools.product(*grid.values()):
        for k, v in zip(grid, combo):
            mod.P[k] = int(v) if isinstance(base[k], int) else v
        print(dict(zip(grid, combo)), score(mod, items, cal), flush=True)
