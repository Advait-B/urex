"""build.py - run the baseline once on every frame and cache features + errors (cache/*.npz, not committed).
    python experiments/quality-gate/build.py            (from research/, ~3 min)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from algos import gate_features as gf  # noqa: E402
from bench import calib as calib_mod, datasets, depth, run  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TOL = 0.5


def thumb(g):
    return cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA)


def frame_errors(grooves, truth):
    """harness errors (matched + scored grooves) and the strict worst error over EVERY reported number:
    a reported groove is compared with the true groove it overlaps; overlapping none = phantom (99 mm)."""
    in_view = [g for g in truth["grooves"] if g["in_view"]]
    pairs, _ = run.match_grooves(grooves, in_view)
    errs = [p["depth"] - t["depth_mm"] for t, p in pairs if p is not None and t["floor_visible_px"] >= 3]
    n_meas = sum(t["floor_visible_px"] >= 3 for t, _ in pairs)
    strict = []
    for p in grooves:
        ov = [(min(p["u1"], t["u1"] + 4) - max(p["u0"], t["u0"] - 4), t) for t in truth["grooves"]]
        ov = [o for o in ov if o[0] > 0]
        if not ov:
            strict.append(99.0)
            continue
        t = max(ov, key=lambda o: o[0])[1]
        if t["floor_visible_px"] >= 3:
            strict.append(abs(p["depth"] - t["depth_mm"]))
    true_pass = all(g["depth_mm"] >= run.LEGAL_MM for g in in_view) if in_view else None
    pred_pass = bool(grooves) and all(g["depth"] >= run.LEGAL_MM for g in grooves)
    return errs, n_meas, max(strict, default=0.0), true_pass, pred_pass


def sim(name, cal, with_off):
    items = datasets.load_sim(name)
    X, Xoff, th, thoff, rows, errs_all, idx = [], [], [], [], [], [], []
    for i, it in enumerate(items):
        line, res = gf.analyse(it["on"])
        f, _ = gf.features(it["on"], line, res)
        out = depth.measure_from_line(line["u"], line["v"], cal)
        errs, n_meas, strict, tp, pp = frame_errors(out["grooves"], it["truth"])
        sc = it["truth"]["scene"]
        X.append([f[k] for k in gf.ALL]); th.append(thumb(it["on"]))
        rows.append([len(out["grooves"]), n_meas, len(errs), strict, max(np.abs(errs), default=0.0),
                     -1 if tp is None else tp, pp, sc["line_peak_dn"], sc["ambient"], sc["blur_px"]])
        errs_all += errs; idx += [i] * len(errs)
        if with_off and i < 400:
            fo, oo = gf.features(it["off"])
            Xoff.append([fo[k] for k in gf.ALL] + [len(oo["grooves"])]); thoff.append(thumb(it["off"]))
    np.savez_compressed(os.path.join(CACHE, f"{name}.npz"), X=np.array(X), meta=np.array(rows, float), thumbs=np.array(th),
                        errs=np.array(errs_all), err_frame=np.array(idx), Xoff=np.array(Xoff), thumbs_off=np.array(thoff),
                        meta_cols=["n_pred", "n_meas", "n_found", "strict_worst", "harness_worst", "true_pass", "pred_pass",
                                   "line_peak_dn", "ambient", "blur_px"])
    print(name, len(items), flush=True)


def real():
    items = datasets.load_real()
    X, th, ng = [], [], []
    for it in items:
        f, out = gf.features(it["gray"])
        X.append([f[k] for k in gf.ALL]); th.append(thumb(it["gray"])); ng.append(len(out["grooves"]))
    np.savez_compressed(os.path.join(CACHE, "real.npz"), X=np.array(X), thumbs=np.array(th), n_pred=np.array(ng),
                        y=np.array([it["usable"] for it in items]), laser=np.array([it["laser"] for it in items]),
                        names=np.array([it["name"] for it in items]), scene=np.array([it["scene"] for it in items]),
                        session=np.array([it["name"].split("_")[1] for it in items]))
    print("real", len(items))


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    cal = calib_mod.default_calib()
    real()
    for n in sys.argv[1:] or ["ideal", "stress", "legal", "train"]:
        sim(n, cal, n == "train")
