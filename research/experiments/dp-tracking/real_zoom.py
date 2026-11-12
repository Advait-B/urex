"""real_zoom.py - the 6 truck-tyre reference frames, zoomed on the groove: reference (cyan), path (yellow), points (red)."""
import sys, importlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench import datasets, depth

mod = importlib.import_module(sys.argv[1])
items = [it for it in datasets.load_real() if it["ref"] is not None and it["ref"]["mode"] == "groove"]
fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
for a, it in zip(ax.ravel(), items):
    g = it["gray"]; p = mod.extract_line(g)
    out = depth.measure_from_line(p["u"], p["v"])
    best = max(out["grooves"], key=lambda q: q["depth"]) if out["grooves"] else None
    a.imshow(g, cmap="gray_r", vmin=0, vmax=255)       # inverted: light background
    a.plot(it["ref"]["u"], it["ref"]["v"], "o", c="#1f9bd1", ms=6, mfc="none", mew=0.7, label="reference (baseline)")
    if hasattr(mod, "track"):
        t = mod.track(g); a.plot(np.arange(g.shape[1]), t["path"], "-", c="#e8a400", lw=0.9, label="Viterbi path")
    a.plot(p["u"], p["v"], ".", c="#d62728", ms=3.5, label="kept points")
    ru, rv = it["ref"]["u"], it["ref"]["v"]
    dev = np.abs(rv - np.median(rv)); c = ru[np.argmax(dev)]
    a.set_xlim(max(0, c - 55), min(g.shape[1], c + 55)); a.set_ylim(np.median(rv) + 75, np.median(rv) - 25)
    a.set_title(f"{it['name'][6:-4]}  ref {it['ref']['depth_px']:.1f} px, got {'none' if best is None else '%.1f px' % best['depth']}", fontsize=10)
ax[0, 0].legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=100, facecolor="white")
