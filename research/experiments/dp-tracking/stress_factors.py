"""stress_factors.py - one stress factor at a time. The SAME 40 clean scenes (seed 7) are re-rendered with
exactly one thing made bad, then scored with the harness' own eval_sim. Extra data, not a test set.
    PYTHONPATH=. python experiments/dp-tracking/stress_factors.py render cache.pkl
    PYTHONPATH=. python experiments/dp-tracking/stress_factors.py score cache.pkl algos.baseline_steger algos.dp_dense
"""
import copy, importlib, json, pickle, sys
import numpy as np
from bench import calib, run
from sim import holder_sim as hs

N = 40


def glints(rng, n):
    return [{"u": float(rng.uniform(0, 1)), "v": float(rng.uniform(0.25, 0.75)), "len": float(rng.uniform(8, 25)),
             "wid": float(rng.uniform(0.8, 2.5)), "ang": float(rng.uniform(0, np.pi)), "amp": float(rng.uniform(150, 300))}
            for _ in range(n)]


def gaps(rng):
    return [(float(a), float(a + rng.uniform(0.06, 0.10))) for a in rng.uniform(0.05, 0.85, 2)]


FACTORS = {
    "clean": lambda s, r: {},
    "12 glints": lambda s, r: {"ambient": 0.8, "glints": glints(r, 12)},
    "paint stripe": lambda s, r: {"ambient": 0.6, "paint": {"v": float(r.uniform(0.3, 0.7)), "wid": 10.0, "amp": 160.0}},
    "2 debris gaps": lambda s, r: {"gaps": gaps(r)},
    "gaps + glints": lambda s, r: {"ambient": 0.8, "glints": glints(r, 12), "gaps": gaps(r)},
    "dim laser (30 DN)": lambda s, r: {"ambient": 0.6, "line_peak_dn": 30.0},
    "saturated (600 DN)": lambda s, r: {"line_peak_dn": 600.0},
    "strong blur": lambda s, r: {"blur_px": 2.4, "line_w_mm": 1.2},
}

if sys.argv[1] == "render":
    rng = np.random.default_rng(7)
    base = [hs.sample_scene(rng, "ideal") for _ in range(N)]
    data = {}
    for name, f in FACTORS.items():
        r = np.random.default_rng(8)
        data[name] = []
        for s in base:
            s2 = {**copy.deepcopy(s), **f(s, r)}
            on, off, t = hs.render(s2)
            data[name].append({"on": on, "off": off, "truth": t})
        print(name, flush=True)
    pickle.dump(data, open(sys.argv[2], "wb"))
else:
    data = pickle.load(open(sys.argv[2], "rb"))
    cal = calib.default_calib()
    res = {}
    for m in sys.argv[3:]:
        mod = importlib.import_module(m)
        _, measure = run.resolve(mod)
        print(f"\n{mod.NAME}\n{'factor':20s} {'F1':>6s} {'rmse':>6s} {'MAE mm':>7s} {'P95':>6s} {'miss':>6s} {'false':>6s}")
        for name, items in data.items():
            r = run.eval_sim(measure, items, cal)
            res[f"{mod.NAME}|{name}"] = {k: v for k, v in r.items() if not k.startswith("_")}
            print(f"{name:20s} {r['line_f1']:6.3f} {r['line_rmse_px']:6.3f} {r['depth_mae_mm']:7.3f} {r['depth_p95_mm']:6.3f} {r['miss_rate']:6.3f} {r['false_per_frame']:6.3f}")
    json.dump(res, open("experiments/dp-tracking/stress_factors.json", "w"), indent=1)
