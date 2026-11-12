"""f_curvature.py - experiment f: the "curvature bias" of the depth stage. It is two geometric effects.

Take the calibration out completely: feed the default depth stage the TRUE line with its TRUE heights.
What is left is geometry. zs = tread height above the feet (~3 mm), d = groove depth, Zc = 85 mm.

  along the tyre (Y)   The laser comes in at 35 deg, so it meets the floor at Y = (zs - d) tan35 instead of
                       Y = zs tan35, where the tyre (radius R) is lower:
                           + tan^2(35) * ((zs - d)^2 - zs^2) / (2 R)        [+ tan(tilt_x) tan35 d if rocked]
  across the tyre (X)  The default stage compares a floor point with the tread fitted at the SAME IMAGE
                       COLUMN. A floor point at X is d mm further from the lens, so that column's tread is
                       at X' = X (Zc - zs) / (Zc - zs + d), nearer the middle, where the crowned tread
                       (cross radius 900 mm) is higher:
                           + (X^2 - X'^2) / (2 * 900)  -  tan(tilt_y) (X - X')
Fixes (algos/calib_fit.py): mm_space = fit the tread in mm across the tyre (needs the physical model's
X); curvature = subtract the first formula.
    python -m experiments.calibration.f_curvature    -> out/f_curvature.md, figs/f_curvature.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algos import calib_fit
from bench import datasets, depth, run
from sim import holder_sim as hs
from .common import FIGS, OUT, device, table

T35, ZC, CROSS_R = np.tan(np.radians(35.0)), 85.0, 900.0
PHYS = calib_fit.load("phys5_15plates", "phys5")


def grooves(items, **flags):
    """True line through the depth stage. No flags: TRUE heights + default stage. With flags: phys5
    calibration + calib_fit.measure_line(**flags). -> rows (depth, error, zs, tilt_x, tilt_y, X)."""
    out = []
    for it in items:
        t = it["truth"]
        u, v = t["u"][::3].astype(float), t["v"][::3].astype(float)
        o = calib_fit.measure_line(u, v, PHYS, **flags) if flags else depth.measure_from_line(u, v, calib_fit._Fixed(t["height_mm"][::3].astype(float)))
        pairs, _ = run.match_grooves(o["grooves"], [g for g in t["grooves"] if g["in_view"]])
        for g, p in pairs:
            if p is not None and g["floor_visible_px"] >= 3:
                sc, X = t["scene"], g["x_mm"]                                   # tread height at the groove, from the scene
                zs = 3.2 - X ** 2 / (2 * CROSS_R) + np.tan(np.radians(sc["tilt_y_deg"])) * X - sc["dz"]
                out.append((g["depth_mm"], p["depth"] - g["depth_mm"], zs, sc["tilt_x_deg"], sc["tilt_y_deg"], X))
    return np.array(out)


def along(d, zs, R=480.0):
    return T35 ** 2 * ((zs - d) ** 2 - zs ** 2) / (2 * R)


def rock(d, tx):
    return np.tan(np.radians(tx)) * T35 * d


def across(d, zs, X, ty):
    X1 = X * (ZC - zs) / (ZC - zs + d)
    return (X ** 2 - X1 ** 2) / (2 * CROSS_R) - np.tan(np.radians(ty)) * (X - X1)


SETS = {s: datasets.load_sim(s) for s in ("ideal", "stress", "legal")}
g, gf = grooves(SETS["ideal"]), grooves(SETS["ideal"], mm_space=True, curvature=True)
rows = []
for lo, hi in [(0, 2), (2, 4), (4, 8), (8, 12), (12, 15), (15, 19)]:
    k, kf = (g[:, 0] >= lo) & (g[:, 0] < hi), (gf[:, 0] >= lo) & (gf[:, 0] < hi)
    a, c = along(g[k, 0], g[k, 2]), across(g[k, 0], g[k, 2], g[k, 5], g[k, 4])
    rows.append([f"{lo}-{hi} mm", str(int(k.sum())), float(g[k, 1].mean()), float(a.mean()), float(c.mean()), float((a + c).mean()),
                 float(gf[kf, 1].mean()), float(np.abs(gf[kf, 1]).mean())])
md = "**Bias vs depth, 'ideal', true line (mm)**\n\n" + table(
    rows, ["true depth", "grooves", "default stage, true heights: mean error", "formula along (R = 480)", "formula across (crown + tilt_y)",
           "sum of formulas", "both fixes + phys5: mean error", "both fixes: MAE"])

rows = []
for s, items in SETS.items():
    g = grooves(items)
    e, a, r, c = g[:, 1], along(g[:, 0], g[:, 2]), rock(g[:, 0], g[:, 3]), across(g[:, 0], g[:, 2], g[:, 5], g[:, 4])
    k = np.abs(e) < 0.5                                   # leave the few depth-stage outliers (stones) out of the fit
    coef = np.linalg.lstsq(np.c_[a[k], c[k], r[k]], e[k], rcond=None)[0]
    cells = [s, f"{np.abs(e).mean():.3f} / {e.mean():+.3f}"]
    for flags in (dict(curvature=True), dict(mm_space=True), dict(mm_space=True, curvature=True)):
        x = grooves(items, **flags)
        cells.append(f"{np.abs(x[:, 1]).mean():.3f} / {x[:, 1].mean():+.3f}")
    x = grooves(items, mm_space=True, curvature=True)
    cells += [f"{np.abs(x[:, 1] - rock(x[:, 0], x[:, 3])).mean():.3f}", ", ".join(f"{q:.2f}" for q in coef)]
    rows.append(cells)
md += "\n\n**All sets, true line: MAE / bias (mm)**\n\n" + table(
    rows, ["set", "default stage, true heights", "phys5 + curvature fix only", "phys5 + mm_space only", "phys5 + both",
           "both, if tilt_x were ALSO known (not available)", "regression of the default-stage error on (along, across, rocking)"])

# sensitivity to the tyre radius: the ideal scenes on smaller / larger tyres, correction always with R = 480
rows = []
for R in (450.0, 480.0, 520.0):
    dev = device(tyre_R=R)
    items = [{"truth": hs.render(it["truth"]["scene"], dev=dev, return_off=False)[2]} for it in SETS["ideal"][:80]]
    g, x = grooves(items), grooves(items, mm_space=True, curvature=True)
    rows.append([f"{R:.0f}", float(g[g[:, 0] > 12, 1].mean()), float(x[:, 1].mean()), float(x[x[:, 0] > 12, 1].mean()),
                 float(along(16.0, 3.0, R)), float(along(16.0, 3.0, R) - along(16.0, 3.0))])
md += "\n\n**Tyre radius (80 ideal scenes re-rendered, true line; the fix always assumes R = 480)**\n\n" + table(
    rows, ["true R (mm)", "default stage: bias, grooves > 12 mm", "both fixes: bias, all", "both fixes: bias, grooves > 12 mm",
           "formula along @16 mm", "left over @16 mm when 480 is assumed"])
print(md)
open(f"{OUT}/f_curvature.md", "w").write(md + "\n")

g = grooves(SETS["ideal"])
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
d = np.linspace(0.5, 18, 100)
sc = ax[0].scatter(g[:, 0], g[:, 1], s=8, c=np.abs(g[:, 5]), cmap="viridis", label="default stage, true line + true heights")
fig.colorbar(sc, ax=ax[0], label="|X| of the groove (mm)", pad=0.01)
for R, c in ((450, "#d1603d"), (480, "k"), (520, "#2a78b5")):
    ax[0].plot(d, along(d, 3.0, R), color=c, lw=1.2, label=f"along-tyre formula, R = {R}")
ax[0].plot(d, along(d, 3.0) + across(d, 3.0, 40.0, 0.0), "k--", lw=1, label="+ across formula at |X| = 40 mm")
ax[1].scatter(gf[:, 0], gf[:, 1], s=8, color="#2a78b5", label="both fixes (phys5 calibration)")
ax[1].plot(d, rock(d, 0.15), "--", color="#9aa5b1", lw=1, label="+-0.15 deg of rocking (1 sd in 'ideal')"); ax[1].plot(d, -rock(d, 0.15), "--", color="#9aa5b1", lw=1)
for a_ in ax:
    a_.axhline(0, color="k", lw=0.5); a_.set_xlabel("true groove depth (mm)"); a_.grid(alpha=0.3); a_.legend(fontsize=7, loc="upper left"); a_.set_ylim(-0.12, 0.42)
ax[0].set_ylabel("depth error (mm), + = too deep")
fig.suptitle("Experiment f: the 'curvature bias' is geometry in the depth stage, along AND across the tyre", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIGS}/f_curvature.png", dpi=130)
