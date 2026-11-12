"""look.py - contact sheet: frame + truth (cyan) + algorithm points (red). python -m experiments.dp-tracking.look"""
import sys, importlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench import datasets

def sheet(items, mods, path, cols=4, title=None):
    n = len(items)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.1), squeeze=False)
    for a in ax.ravel(): a.axis("off")
    for a, it in zip(ax.ravel(), items):
        g = it["on"] if "on" in it else it["gray"]
        a.imshow(g, cmap="gray", vmin=0, vmax=255)
        if "truth" in it:
            a.plot(it["truth"]["u"], it["truth"]["v"], ".", c="cyan", ms=0.6)
        elif it.get("ref") is not None:
            a.plot(it["ref"]["u"], it["ref"]["v"], ".", c="cyan", ms=1.5)
        for m, c in zip(mods, ["red", "yellow"]):
            p = m.extract_line(g)
            a.plot(p["u"], p["v"], ".", c=c, ms=1.2)
        a.set_title(it.get("title", it.get("name", "")), fontsize=7)
    if title: fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=80, facecolor="white"); plt.close(fig)

if __name__ == "__main__":
    mods = [importlib.import_module(m) for m in sys.argv[3:]]
    if sys.argv[1] == "real":
        items = [it for it in datasets.load_real() if it["ref"] is not None]
    else:
        items = datasets.load_sim(sys.argv[1])[:16]
    sheet(items, mods, sys.argv[2])
