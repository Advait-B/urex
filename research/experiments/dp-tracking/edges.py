"""edges.py - does smoothing help sub-pixel accuracy, and does it smear groove edges? (TRAIN frames only)
Point error = distance to the nearest true line point, split into points within 4 px (in u) of a true
groove edge and all other points; only points within 3 px of the truth count (the rest are not line).
    PYTHONPATH=. python experiments/dp-tracking/edges.py 300
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
from bench import calib, datasets, depth
from algos import dp_dense, dp_smooth

items = datasets.load_sim("train")[:int(sys.argv[1])]
cal = calib.default_calib()
configs = [("none", None, None)] + [(f"lam {l:g} break {b:g}", l, b) for b in (1.0, 1e9) for l in (1, 4, 16, 64)]
pts = [dp_dense.extract_line(it["on"]) for it in items]
print(f"{'smoother':22s} {'rmse flat':>9s} {'rmse edge':>9s} {'rmse all':>9s} {'depth MAE':>9s} {'bias':>7s} {'P95':>7s}")
for name, lam, brk in configs:
    e_flat, e_edge, derr = [], [], []
    for it, p in zip(items, pts):
        n = p["n_path"]; u = p["u"]; v = p["v"].copy()
        if lam is not None:
            v[:n] = dp_smooth.smooth(u[:n], v[:n], lam, brk)
        tr = it["truth"]
        d, _ = cKDTree(np.c_[tr["u"], tr["v"]]).query(np.c_[u, v])
        edges = np.array([e for g in tr["grooves"] for e in (g["u0"], g["u1"])])
        near = np.abs(u[:, None] - edges[None, :]).min(axis=1) < 4
        ok = d < 3
        e_flat += list(d[ok & ~near]); e_edge += list(d[ok & near])
        out = depth.measure_from_line(u, v, cal)
        for g in tr["grooves"]:
            if not g["in_view"] or g["floor_visible_px"] < 3: continue
            m = [q for q in out["grooves"] if min(q["u1"], g["u1"] + 4) - max(q["u0"], g["u0"] - 4) > 0]
            if m: derr.append(max(m, key=lambda q: q["n"])["depth"] - g["depth_mm"])
    rm = lambda x: float(np.sqrt(np.mean(np.square(x))))
    derr = np.array(derr)
    print(f"{name:22s} {rm(e_flat):9.3f} {rm(e_edge):9.3f} {rm(e_flat + e_edge):9.3f} {np.mean(np.abs(derr)):9.3f} {np.mean(derr):7.3f} {np.percentile(np.abs(derr), 95):7.3f}")
