"""presses.py - render P presses of the same tyre position and measure each one.
    presses.py <ideal|stress|legal|legal_stress> [n_scenes=40] [presses=9]  -> presses_<cond>.pkl
Between presses: new noise/speckle seed, tilt +-0.3 deg (both axes), x offset +-1 mm, exposure +-10 %.
Grooves, stones, glints, paint and debris belong to the tyre and stay."""
import copy
import pickle
import sys

import numpy as np

from bench import calib as calib_mod, datasets, depth
from bench.run import match_grooves
from sim import holder_sim as hs
from algos import baseline_steger, onoff_cog

cond = sys.argv[1]
n_sc = int(sys.argv[2]) if len(sys.argv) > 2 else 40
P = int(sys.argv[3]) if len(sys.argv) > 3 else 9
cal = calib_mod.default_calib()
rng = np.random.default_rng({"ideal": 11, "stress": 22, "legal": 33, "legal_stress": 44}[cond])


def base_scene():
    if cond == "legal":
        return datasets._legal_scene(rng, hs)
    s = hs.sample_scene(rng, "stress" if "stress" in cond else "ideal")
    if cond == "legal_stress":                       # stress conditions, depths around the 1.6 mm limit
        b = rng.uniform(0.6, 3.5)
        for g in s["grooves"]:
            g["d"] = float(np.clip(b + rng.normal(0, 0.25), 0.4, 4.0))
        s["stones"] = []
    return s


def press(s):
    p = copy.deepcopy(s)
    p["seed"] = int(rng.integers(1 << 30))
    p["tilt_x_deg"] += float(rng.uniform(-0.3, 0.3))
    p["tilt_y_deg"] += float(rng.uniform(-0.3, 0.3))
    dx = float(rng.uniform(-1, 1))
    for g in p["grooves"] + p["stones"]:
        g["x"] += dx
    e = float(rng.uniform(0.9, 1.1))
    p["line_peak_dn"] *= e
    p["rubber_dn"] *= e
    return p


out = []
for k in range(n_sc):
    s = base_scene()
    rec = {"true": [g["d"] for g in s["grooves"]], "press": []}
    for _ in range(P):
        on, off, t = hs.render(press(s))
        pr = {"scorable": [g["in_view"] and g["floor_visible_px"] >= 3 for g in t["grooves"]],
              "in_view": [g["in_view"] for g in t["grooves"]]}
        for name, ex in (("single", lambda: baseline_steger.extract_line(on)), ("onoff", lambda: onoff_cog.extract_line(on, off))):
            ln = ex()
            m = depth.measure_from_line(ln["u"], ln["v"], cal)
            pairs, extra = match_grooves(m["grooves"], t["grooves"])
            pr[name] = {"d": [None if p is None else p["depth"] for _, p in pairs], "n_false": len(extra),
                        "false_min": min([g["depth"] for g in extra], default=None),
                        "inlier": m["inlier_frac"], "n_pts": len(ln["u"]), "cover": len(np.unique(np.round(ln["u"]))) / on.shape[1]}
        rec["press"].append(pr)
    out.append(rec)
pickle.dump(out, open(f"experiments/onoff-fusion/presses_{cond}.pkl", "wb"))
print(cond, len(out), "scenes x", P, "presses done")
