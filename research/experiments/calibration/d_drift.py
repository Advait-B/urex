"""d_drift.py - experiment d: the device changes AFTER calibration (laser knocked, lens changed).

  1. depth error with the OLD calibration on the exact line of the drifted device: offset or scale?
  2. zero check: one photo of a flat plate at 0 mm (device standing on the plate), read with the old
     calibration. What does it read, and how does that compare with its repeatability without drift?
  3. correction from that one photo:  poly43 + row shift  (shift the image rows so the plate reads 0)
                                      phys5 + re-aim      (PhysModel.refit_plane: re-solve theta, roll)
  4. 100 'ideal' test scenes re-rendered with the drifted device, scored by the harness's eval_sim.
    python -m experiments.calibration.d_drift    -> out/d_drift.md, figs/d_drift.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algos import baseline_steger, calib_models as cm
from bench import datasets, depth, run
from sim import holder_sim as hs
from .common import ALL15, DEPTHS, FIGS, OUT, UCOLS, depth_errors, device, plate_points, plate_set, rms, table

DEV = {"none": device(), "laser +0.25 deg": device(theta_deg=35.25), "laser +0.5 deg": device(theta_deg=35.5),
       "laser +1.0 deg": device(theta_deg=36.0), "laser -0.5 deg": device(theta_deg=34.5),
       "laser roll 0.5 deg": device(laser_roll_deg=0.5), "focal length +1 %": device(f_mm=2.7 * 1.01)}


class RowShift:
    """Old model + a row shift dv(u) (quadratic in u) that makes the zero plate read zero."""

    def __init__(self, model, u, v):
        v0 = np.full(len(u), 120.0)
        for _ in range(8):                                   # Newton: the row where the old model reads 0
            v0 -= model.height(u, v0) / ((model.height(u, v0 + 0.5) - model.height(u, v0 - 0.5)))
        self.model, self.c = model, np.polyfit(u, v - v0, 2)

    def height(self, u, v):
        return self.model.height(u, np.asarray(v, float) - np.polyval(self.c, u))


u, v, z = plate_set(ALL15)
poly, phys = cm.make("poly43").fit(u, v, z), cm.make("phys5").fit(u, v, z)


def zero_reading(model, dev, seed, z=0.0):
    pu, pv = plate_points(z, dev, seed)
    r = model.height(pu, pv) - z
    return float(r.mean()), float(np.polyfit(pu, r, 1)[0] * 300)        # mean (mm), tilt across 300 px (mm)


# repeatability of the check itself (no drift, 12 fresh photos)
rep = np.array([zero_reading(phys, DEV["none"], 200 + i) for i in range(12)])
print(f"zero-check repeatability without drift: mean {rep[:, 0].std():.4f} mm sd, tilt {rep[:, 1].std():.4f} mm sd")

rows1, rows2, rows3, curves = [], [], [], {}
for name, dev in DEV.items():
    e = depth_errors(poly, dev)
    b, a = np.polyfit(DEPTHS, e.mean(1), 1)                              # error = a + b * depth
    rows1.append([name, float(a), f"{100 * b:+.2f} %", float(e[0].mean()), float(e[2].mean()), float(e[4].mean()), float(np.ptp(e[4]))])
    m0, t0 = zero_reading(phys, dev, 100)
    m12, _ = zero_reading(phys, dev, 100, -12.0)
    rows2.append([name, m0, t0, m12, m12 - m0])
    zu, zv = plate_points(0.0, dev, 100)
    fixes = {"poly43 old": poly, "poly43 + row shift": RowShift(poly, zu, zv), "phys5 old": phys, "phys5 + re-aim": phys.refit_plane(zu, zv, 0.0)}
    if name.startswith("focal"):                                         # lens change: needs a second plate
        u2, v2 = plate_points(-12.0, dev, 100)
        m = cm.PhysModel(("k1", "theta", "roll", "y0"), init=dict(phys.p))
        fixes["phys5 + 2-plate refit (0, -12)"] = m.fit(np.r_[zu, u2], np.r_[zv, v2], np.r_[np.zeros(len(zu)), np.full(len(u2), -12.0)])
    curves[name] = {k: depth_errors(f, dev) for k, f in fixes.items()}
    rows3.append([name] + [f"{curves[name][k][4].mean():+.3f} / {rms(curves[name][k]):.3f}" for k in list(fixes)[:4]])
    if name.startswith("focal"):
        k = "phys5 + 2-plate refit (0, -12)"
        print(f"focal +1 %: {k}: bias@16 {curves[name][k][4].mean():+.3f}, RMS {rms(curves[name][k]):.3f}")
    DEV[name] = (dev, fixes)

# 4. 100 'ideal' test scenes re-rendered with the drifted device. Baseline (Steger) line for three
#    calibrations, plus the TRUE line for the re-aimed one (= what is left that is not line noise).
#    The mean is pulled around by a few line outliers, so the median error is reported as well.
def score(items, lines, model):
    r = run.eval_sim(lambda g, c=None, off=None: depth.measure_from_line(*lines[id(g)], model), items, None)
    return f"{r['depth_mae_mm']:.3f} / {r['depth_bias_mm']:+.3f} / {np.median(np.subtract(r['_pred_mm'], r['_true_mm'])):+.3f}"


ideal = datasets.load_sim("ideal")[:100]
rows4 = []
for name, (dev, fixes) in DEV.items():
    items, st, tr = [], {}, {}
    for it in ideal:
        on, off, t = hs.render(it["truth"]["scene"], dev=dev)
        items.append({"on": on, "off": off, "truth": t})
        p = baseline_steger.extract_line(on)
        st[id(on)], tr[id(on)] = (p["u"], p["v"]), (t["u"][::3], t["v"][::3])
    rows4.append([name] + [score(items, st, fixes[k]) for k in ("poly43 old", "poly43 + row shift", "phys5 + re-aim")]
                 + [score(items, tr, fixes["phys5 + re-aim"])])

md = ("**1. Old poly43 calibration on the drifted device (exact line, flat surface +3.2 mm)**\n\n"
      + table(rows1, ["drift", "offset a (mm)", "scale b", "error @1.6", "@8", "@16", "spread across columns @16"])
      + "\n\n**2. Zero check read with the old phys5 calibration**\n\n"
      + table(rows2, ["drift", "0 mm plate reads (mm)", "tilt across 300 px (mm)", "-12 mm plate reads error (mm)", "difference (-12) - (0)"])
      + "\n\n**3. Depth error after a one-photo correction: bias @16 mm / RMS all depths (mm)**\n\n"
      + table(rows3, ["drift", "poly43 old", "poly43 + row shift", "phys5 old", "phys5 + re-aim"])
      + "\n\n**4. 100 'ideal' scenes re-rendered with the drifted device: depth MAE / bias / median error (mm)**\n\n"
      + table(rows4, ["drift", "poly43 old (baseline line)", "poly43 + row shift (baseline line)", "phys5 + re-aim (baseline line)", "phys5 + re-aim (TRUE line)"]))
print(md)
open(f"{OUT}/d_drift.md", "w").write(md + "\n")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
cols = ["#9aa5b1", "#f0b27a", "#d1603d", "#8b2f1c", "#2a78b5", "#4f9d69", "#8e6bb0"]
for a_, key, title in zip(ax, ("poly43 old", "poly43 + row shift", "phys5 + re-aim"),
                          ("old calibration (poly43)", "zero check: row shift (poly43)", "zero check: re-aim the laser plane (phys5)")):
    for (name, c) in zip(DEV, cols):
        a_.plot(DEPTHS, curves[name][key].mean(1), "o-", color=c, label=name, ms=3)
    a_.axhline(0, color="k", lw=0.5); a_.set_title(title, fontsize=9); a_.set_xlabel("groove depth (mm)"); a_.grid(alpha=0.3)
ax[0].set_ylabel("depth error (mm)"); ax[0].legend(fontsize=7)
fig.suptitle("Experiment d: drift after calibration, and what one photo of a flat plate can repair", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIGS}/d_drift.png", dpi=130)
