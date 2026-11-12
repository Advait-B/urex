"""calib_fit.py - helper for the calib_* benchmark variants (not an algorithm itself).

Each variant = baseline Steger line + default depth stage, but with its OWN px -> mm model instead of
the Calib the harness passes in. The model is fitted once and cached as JSON under research/models/.

    model = load("phys5_15plates", "phys5")                  # the 15 plates of the 'calib' set
    model = load("phys5_shims", "phys5", SHIMS)               # plates rendered at the given heights
    measure = make_measure(model, curvature=True)             # + tyre-curvature correction (see notes, f)
"""
import json
import os

import numpy as np

from bench import datasets, depth
from . import baseline_steger, calib_models as cm

MODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
SHIMS = [0, -2, -4, -6, -8, -10, -12, -15, -18, -20]      # what pipeline/calibrate.py tells the students to do
RECOMMENDED = [4, 2, 0, -3, -6, -10, -14, -18]            # this branch's procedure: 2 raised blocks + 6 shim stacks


def plate_points(items):
    """Same point selection as bench/calib.py."""
    U, V, Z = [], [], []
    for it in items:
        p = baseline_steger.extract_line(it["on"], it["off"])
        u, v = np.asarray(p["u"], float), np.asarray(p["v"], float)
        keep = np.abs(v - np.median(v)) < 6
        U.append(u[keep]); V.append(v[keep]); Z.append(np.full(keep.sum(), it["height_mm"]))
    return map(np.concatenate, (U, V, Z))


def load(tag, kind, heights=None):
    path = os.path.join(MODELS, f"calib_{tag}.json")
    cls = cm.PolyModel if kind.startswith("poly") else cm.PhysModel
    if os.path.exists(path):
        with open(path) as fh:
            return cls.from_json(json.load(fh))
    if heights is None:
        items = datasets.load_sim("calib")
    else:                                                   # render the plates exactly as datasets.py does
        from sim import holder_sim as hs
        rng = np.random.default_rng(909)
        items = []
        for z in heights:
            on, off, _ = hs.render(hs.flat_scene(z, rng))
            items.append({"on": on, "off": off, "height_mm": float(z)})
    model = cm.make(kind).fit(*plate_points(items))
    os.makedirs(MODELS, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(model.to_json(), fh, indent=1)
    return model


class CadGeometry:
    """Stand-in when the calibration is not a physical model: the CAD numbers of the holder."""
    p = {"cx": 159.5, "Zc": 85.0}

    def plane_y(self, z):
        return z * np.tan(np.radians(35.0))


class _Fixed:                                               # hands measure_from_line heights computed beforehand
    def __init__(self, z):
        self.z = z

    def height(self, u, v):
        return self.z


def curvature_fix(out, geo, tyre_R=480.0):
    """The laser meets a groove floor further round the tyre than it meets the tread surface, where the
    whole tyre is lower: a groove reads too deep by (Y(floor)^2 - Y(surface)^2) / (2 R). Subtract it."""
    for g in out["grooves"]:
        zs = -np.polyval(out["surface"], 0.5 * (g["u0"] + g["u1"]))                # surface height at the groove (mm)
        g["depth"] -= (geo.plane_y(zs - g["depth"]) ** 2 - geo.plane_y(zs) ** 2) / (2 * tyre_R)
    return out


def surface_columns(u, z, geo):
    """Perspective fix (CAD version, ignores lens distortion; make_measure uses the exact xyz /
    project pair when the calibration is a PhysModel). A floor point d mm further from the lens appears (Zc - zs + d) / (Zc - zs) closer
    to the image centre than the tread directly above it would. The default depth stage compares a floor
    point with the tread fit AT THE SAME COLUMN, i.e. with tread that is really further inboard, and the
    tread is crowned / tilted across the tyre. So move every point to the column it would have at the
    height of the tread (z_ref) before fitting the surface: then 'same column' means 'same X in mm'."""
    z_ref = np.median(z)
    return geo.p["cx"] + (u - geo.p["cx"]) * (geo.p["Zc"] - z) / (geo.p["Zc"] - z_ref)


def measure_line(u, v, cal, curvature=False, mm_space=False, tyre_R=480.0):
    """Line points + calibration -> grooves. Without the two flags this is exactly the default depth stage."""
    geo = cal if hasattr(cal, "plane_y") else CadGeometry()
    if mm_space and len(u) >= 12 and hasattr(cal, "xyz"):
        # physical model: the abscissa is X in mm (drawn at ~px scale so the stage's px thresholds still
        # apply). Exact: neither perspective nor lens distortion is left in it.
        X, _, z = cal.xyz(u, v)
        z_ref = np.median(z)
        k = cal.f_px / (cal.p["Zc"] - z_ref)
        out = depth.measure_from_line(cal.p["cx"] + k * X, v, _Fixed(z))
        if curvature and out["surface"] is not None:
            curvature_fix(out, geo, tyre_R)
        for g in out["grooves"]:                            # report the groove edges as real image columns again
            for e in ("u0", "u1"):
                g[e] = float(cal.project((g[e] - cal.p["cx"]) / k, cal.plane_y(z_ref), z_ref)[0])
        return out
    if mm_space and len(u) >= 12:
        z = cal.height(u, v)
        out = depth.measure_from_line(surface_columns(u, z, geo), v, _Fixed(z))
    else:
        out = depth.measure_from_line(u, v, cal)
    if curvature and out["surface"] is not None:
        curvature_fix(out, geo, tyre_R)
    return out


def make_measure(model, **flags):
    """model None = use the Calib the harness passes in. flags: see measure_line."""
    def measure(gray, calib=None, off=None):
        p = baseline_steger.extract_line(gray, off)
        u, v = np.asarray(p["u"], float), np.asarray(p["v"], float)
        if calib is None:                                   # real photos: no calibration exists, stay in px
            out = depth.measure_from_line(u, v, None)
        else:
            out = measure_line(u, v, model or calib, **flags)
        out["line"] = p
        return out
    return measure
