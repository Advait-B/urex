"""factors.py - one stress factor at a time on otherwise ideal scenes: what does the off frame fix?"""
import copy
import json
import sys

import numpy as np

from variants import make
from bench import calib as calib_mod
from bench.run import eval_sim
from sim import holder_sim as hs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
G = [{"u": u, "v": v, "len": 20.0, "wid": 2.0, "ang": a, "amp": 300.0}
     for u, v, a in [(.2, .45, .3), (.5, .55, 1.2), (.8, .5, 2.0), (.35, .6, 2.6), (.65, .4, .8), (.5, .5, 0.1)]]
FACTORS = {
    "none": {},
    "sunlight (ambient 1.0)": {"ambient": 1.0},
    "sunlight + gradient": {"ambient": 1.0, "ambient_grad": (0.9, 0.9)},
    "6 wet glints near line": {"ambient": 0.8, "glints": G},
    "paint stripe near line": {"ambient": 0.8, "paint": {"v": 0.55, "wid": 10.0, "amp": 160.0}},
    "saturated laser (600 DN)": {"line_peak_dn": 600.0},
    "dim laser (40 DN)": {"line_peak_dn": 40.0},
    "blur 2.4 px + wide line": {"blur_px": 2.4, "line_w_mm": 1.2},
    "5 laser spikes": {"spikes": 5},
    "rocking 1.5 deg": {"tilt_x_deg": 1.5, "tilt_y_deg": -1.5},
}
VARS = [("single", "steger"), ("single", "cog"), ("plain", "steger"), ("plain", "cog")]
cal = calib_mod.default_calib()
rng = np.random.default_rng(5150)
scenes = [hs.sample_scene(rng, "ideal") for _ in range(N)]
res = {}
for name, upd in FACTORS.items():
    items = []
    for s in scenes:
        s2 = copy.deepcopy(s)
        s2.update(copy.deepcopy(upd))
        on, off, t = hs.render(s2)
        items.append({"on": on, "off": off, "truth": t})
    res[name] = {}
    for mode, det in VARS:
        r = eval_sim(make(mode, det), items, cal)
        res[name][f"{mode}+{det}"] = {k: v for k, v in r.items() if not k.startswith("_")}
    print(f"{name:26s} " + " | ".join(f"{k} F1 {r['line_f1']:.3f} MAE {r['depth_mae_mm']:.3f} miss {r['miss_rate']:.2f} false {r['false_per_frame']:.2f}"
                                      for k, r in res[name].items()), flush=True)
json.dump(res, open("experiments/onoff-fusion/factors.json", "w"), indent=1)
