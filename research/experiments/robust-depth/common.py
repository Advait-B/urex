"""common.py - shared helpers for the robust-depth experiments (run from research/).

lines(set, src): line points per frame, src = "oracle" (true line, every 3rd point, as bench/oracle.py)
or "baseline" (algos.baseline_steger, cached in experiments/robust-depth/_cache/).
score(stage, set, src): run a "line -> grooves" stage through bench.run.eval_sim.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from bench import calib as calib_mod, datasets, run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache")
CAL = calib_mod.default_calib()
_items = {}


def memo_surface():
    """Experiments only: remember surface fits (RANSAC is deterministic and is most of the run time)."""
    from algos import robust_depth as rd
    from bench import depth
    for mod, fn in ((rd, "fit_surface"), (depth, "ransac_quadratic"), (rd, "ransac_quadratic")):
        f, memo = getattr(mod, fn), {}
        if getattr(f, "_memo", False):
            continue

        def g(u, q, *a, _f=f, _m=memo, **k):
            key = (len(u), float(np.sum(u)), float(np.sum(q)), repr(a), repr(k))
            if key not in _m:
                _m[key] = _f(u, q, *a, **k)
            return _m[key]
        g._memo = True
        setattr(mod, fn, g)


def _extra(name):
    """Extra simulated sets for this topic (cached in _cache/). *_tune = tuning, *_test = reporting.
    wide_*    3-4 wide grooves (up to ~60 % of the view) and +-1.5 deg rocking: surface-model test
    uneven_*  legal-style conditions, every groove worn independently: one groove 0.15-1.6 mm
              next to 1.6-4 mm ones (the case where a missed shallow groove turns FAIL into PASS)"""
    path = os.path.join(CACHE, f"sim_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    sys.path.insert(0, "sim")
    import holder_sim as hs
    kind, part = name.split("_")
    rng = np.random.default_rng({"wide_tune": 11, "wide_test": 12, "uneven_tune": 21, "uneven_test": 22}[name])
    out = []
    for _ in range(150 if kind == "wide" else 300):
        s = hs.sample_scene(rng, "ideal")
        if kind == "wide":
            n = int(rng.choice([3, 4]))
            cx = np.linspace(-40, 40, n) + rng.uniform(-3, 3, n)
            base = rng.uniform(2, 14)
            s["grooves"] = [{"x": float(c), "w": float(rng.uniform(13, 17)), "d": float(max(1.0, base + rng.normal(0, 0.8))),
                             "wall_deg": float(rng.uniform(8, 22))} for c in cx]
            s["tilt_x_deg"], s["tilt_y_deg"] = float(rng.normal(0, 1.5)), float(rng.normal(0, 1.5))
        else:
            mode = rng.choice(["one_worn", "all_ok", "all_near"], p=[0.45, 0.2, 0.35])
            base = rng.uniform(0.2, 3.0)
            worn = int(rng.integers(len(s["grooves"])))
            for j, g in enumerate(s["grooves"]):
                g["d"] = float({"one_worn": rng.uniform(0.15, 1.6) if j == worn else rng.uniform(1.6, 4.0),
                                "all_ok": rng.uniform(1.6, 4.0), "all_near": max(0.15, base + rng.normal(0, 0.25))}[mode])
            s["ambient"], s["blur_px"] = float(rng.uniform(0.05, 0.5)), float(rng.uniform(0.6, 1.5))
            s["tilt_x_deg"], s["tilt_y_deg"] = float(rng.normal(0, 0.5)), float(rng.normal(0, 0.5))
        on, off, t = hs.render(s)
        out.append({"on": on, "off": off, "truth": t})
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out


def items(name):
    if name not in _items:
        _items[name] = datasets.load_real() if name == "real" else _extra(name) if "_" in name else datasets.load_sim(name)
    return _items[name]


def lines(name, src, limit=None):
    its = items(name)[:limit] if limit else items(name)
    if src == "oracle":
        return [{"u": it["truth"]["u"][::3], "v": it["truth"]["v"][::3]} for it in its]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"baseline_{name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            L = pickle.load(fh)
    else:
        from algos import baseline_steger
        key = "gray" if name == "real" else "on"
        L = [{k: np.asarray(v) for k, v in baseline_steger.extract_line(it[key]).items()} for it in items(name)]
        with open(path, "wb") as fh:
            pickle.dump(L, fh)
    return L[:limit] if limit else L


def score(stage, name, src, limit=None, calib=CAL):
    """stage(u, v, calib) -> {"grooves": [...]}. Returns the bench.run metrics dict (+ '_outs')."""
    its = items(name)[:limit] if limit else items(name)
    L = lines(name, src, limit)
    look = {id(it["gray" if name == "real" else "on"]): l for it, l in zip(its, L)}
    outs = []

    def measure(gray, c=None, off=None):
        l = look[id(gray)]
        out = stage(l["u"], l["v"], c)
        out["line"] = l
        outs.append(out)
        return out
    res = run.eval_real(measure, its) if name == "real" else run.eval_sim(measure, its, calib)
    res["_outs"] = outs
    return res


def pairs(name, outs, limit=None):
    """(truth groove, predicted groove or None, frame index) for every in-view truth groove."""
    its = items(name)[:limit] if limit else items(name)
    out = []
    for i, (it, o) in enumerate(zip(its, outs)):
        tv = [g for g in it["truth"]["grooves"] if g["in_view"]]
        pr, _ = run.match_grooves([g for g in o["grooves"] if g.get("unit", "mm") == "mm"], tv)
        out += [(t, p, i) for t, p in pr]
    return out
