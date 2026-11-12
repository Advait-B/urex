"""worst.py - the 10 largest depth errors of a stack on one set, with the scene parameters, and a picture of each.
    python experiments/best-stack/worst.py stack_dp_rd_gate stress      (from research/)
-> prints a table, writes experiments/best-stack/cache/worst_<module>_<set>.png (for eyeballing; not committed)"""
import importlib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from bench import calib as calib_mod, datasets, datasets2  # noqa: E402
from bench.run2 import assign  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["line_peak_dn", "ambient", "blur_px", "tilt_x_deg", "tilt_y_deg", "wet", "n_stones", "zigzag", "sun", "paint", "debris"]


def collect(modname, name):
    mod = importlib.import_module("algos." + modname)
    items = datasets2.load_sim2(name) if name in ("shifted", "handheld") else datasets.load_sim(name)
    px, cal0, recs, frames = name == "handheld", calib_mod.default_calib(), [], {}
    for i, it in enumerate(items):
        cal = None if px else (calib_mod.Calib(**it["calib"]) if "calib" in it else cal0)
        out = mod.measure(it["on"], cal, None)
        ans = out.get("usable", 1.0) >= 0.5
        in_view = [g for g in it["truth"]["grooves"] if g["in_view"]]
        per, none = assign(out["grooves"], in_view)
        frames[i] = {"answered": ans, "n_none": len(none), "none": none, "out": out}
        for t, ps in zip(in_view, per):
            meas = (t.get("shift_px") or 0) > 3 if px else t["floor_visible_px"] >= 3
            if ans and meas and ps:
                recs.append({"i": i, "true": t["shift_px" if px else "depth_mm"], "pred": ps[0]["depth"], "lb": bool(ps[0].get("lower_bound")),
                             "floor_px": t["floor_visible_px"], "tu": (t["u0"], t["u1"]), "pu": (ps[0]["u0"], ps[0]["u1"]), "width": t.get("width_mm")})
    return items, recs, frames


def main(modname, name, n=10):
    items, recs, frames = collect(modname, name)
    recs.sort(key=lambda r: -abs(r["pred"] - r["true"]))
    fig, axs = plt.subplots(2, 5, figsize=(20, 7))
    for r, ax in zip(recs[:n], axs.ravel()):
        it = items[r["i"]]
        sc = it["truth"]["scene"]
        desc = {k: (round(sc[k], 2) if isinstance(sc[k], float) else sc[k]) for k in sc if not isinstance(sc[k], (list, dict, np.ndarray))}
        print(f"frame {r['i']:3d} true {r['true']:6.2f} pred {r['pred']:6.2f} err {r['pred'] - r['true']:+6.2f} lb={int(r['lb'])} floor_vis_px={r['floor_px']:.0f} "
              f"true_u={r['tu'][0]:.0f}-{r['tu'][1]:.0f} pred_u={r['pu'][0]:.0f}-{r['pu'][1]:.0f} | {desc} | extras: "
              f"{ {k: (len(v) if isinstance(v, list) else '...') for k, v in sc.items() if isinstance(v, (list, dict))} }")
        ax.imshow(it["on"], cmap="gray", vmin=0, vmax=255)
        ax.plot(it["truth"]["u"], it["truth"]["v"], ".", ms=1.5, color="cyan", alpha=0.6)
        ln = frames[r["i"]]["out"]["line"]
        ax.plot(ln["u"], np.asarray(ln["v"]) + 0.0, ".", ms=1.5, color="red", alpha=0.6)
        ax.axvspan(*r["pu"], color="yellow", alpha=0.15)
        ax.set_title(f"#{r['i']} true {r['true']:.1f} pred {r['pred']:.1f}" + (" LB" if r["lb"] else ""), fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "cache", f"worst_{modname}_{name}.png"), dpi=70, facecolor="white")
    e = np.array([abs(r["pred"] - r["true"]) for r in recs])
    print(f"{modname} {name}: {len(recs)} scored grooves; sum of top-10 |err| = {e[:10].sum():.2f} of total {e.sum():.2f}; "
          f"MAE without the top 10: {e[10:].mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
