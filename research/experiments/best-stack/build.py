"""build.py - run a stack on a simulated set and store, per frame, the gate features and what was right or wrong.
    python experiments/best-stack/build.py train            (from research/; ~1.5 min for train)
-> experiments/best-stack/cache/<set>.pkl   (list of per-frame dicts)
The stack here is dp_dense + rd_final + cap, with NO abstaining, so every later rule can be tried offline."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from algos import stackcommon as sc  # noqa: E402
from algos.dp_dense import extract_line  # noqa: E402
from bench import calib as calib_mod, datasets, datasets2  # noqa: E402
from bench.run2 import assign  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name):
    items = datasets2.load_sim2(name) if name in ("shifted", "handheld") else datasets.load_sim(name)
    cal0, rows = calib_mod.default_calib(), []
    for i, it in enumerate(items):
        px = name == "handheld"
        cal = None if px else (calib_mod.Calib(**it["calib"]) if "calib" in it else cal0)
        out = sc.measure(it["on"], cal, None, line_fn=extract_line, depth="rd", cap=True)
        tr = it["truth"]
        in_view = [g for g in tr["grooves"] if g["in_view"]]
        key = "shift_px" if px else "depth_mm"
        per, none = assign(out["grooves"], in_view)
        gro = []
        for t, ps in zip(in_view, per):
            meas = (t.get("shift_px") or 0) > 3 if px else t["floor_visible_px"] >= 3
            p = ps[0] if ps else None
            gro.append({"true": t[key], "measurable": bool(meas), "floor_visible_px": t["floor_visible_px"], "u0": t["u0"], "u1": t["u1"],
                        "pred": None if p is None else p["depth"], "lower_bound": None if p is None else bool(p.get("lower_bound")),
                        "sigma": None if p is None else p.get("sigma"), "pu0": None if p is None else p["u0"], "pu1": None if p is None else p["u1"]})
        rows.append({"i": i, "f": sc.stack_features(it["on"], out["line"], out), "grooves": gro, "n_phantom": len(none),
                     "phantom": [g["depth"] for g in none], "n_capped": len(out["capped"]), "n_pred": len(out["grooves"]),
                     "verdict": out["verdict"], "true_pass": bool(in_view) and all(g["depth_mm"] >= 1.6 for g in in_view), "has_view": bool(in_view),
                     "pred_pass": bool(out["grooves"]) and all(g["depth"] >= 1.6 for g in out["grooves"]),
                     "scene": {k: v for k, v in tr["scene"].items() if not isinstance(v, (list, dict, np.ndarray))}})
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    pickle.dump(rows, open(os.path.join(HERE, "cache", name + ".pkl"), "wb"))
    print(name, len(rows), "frames")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
