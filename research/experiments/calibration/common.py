"""common.py - shared helpers for the calibration experiments. Run scripts from research/:

    OMP_NUM_THREADS=1 python -m experiments.calibration.a_plates

  plate_points(z, ...)   render a flat plate, extract the line with the baseline (what the students get)
  true_line(z, dev)      the exact line of a flat plate at height z for any device (no image, no noise)
  depth_errors(model)    calibration-only depth error: surface plate at +3.2 mm vs floor plate d below
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, RESEARCH)
from algos import baseline_steger  # noqa: E402
from sim import holder_sim as hs   # noqa: E402

FIGS = os.path.join(RESEARCH, "notes", "figs", "calibration")
OUT = os.path.join(HERE, "out")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

ALL15 = [6, 4, 2, 0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20, -22]
SURF_Z = 3.2                                   # the tread surface sits ~3.2 mm ABOVE the feet plane (crown)
DEPTHS = (1.6, 4.0, 8.0, 12.0, 16.0)
UCOLS = np.arange(20, 301, 4.0)

# The simulator ignores tilt for flat plates. Wrap surface_z (no edit of sim/) so a plate can be tilted:
# scene keys plate_tilt_x_deg (slope along Y, the triangulation direction) and plate_tilt_y_deg (along X).
_surface_z = hs.surface_z


def _surface_z_tilt(X, Y, s, dev=hs.DEVICE):
    z = _surface_z(X, Y, s, dev)
    if "flat_z" in s:
        z = z + np.tan(np.radians(s.get("plate_tilt_x_deg", 0.0))) * Y + np.tan(np.radians(s.get("plate_tilt_y_deg", 0.0))) * X
    return z


hs.surface_z = _surface_z_tilt


def device(**changes):
    d = copy.deepcopy(hs.DEVICE)
    d.update(changes)
    return d


def plate_points(z, dev=None, seed=0, tilt_x=0.0, tilt_y=0.0):
    """Render a plate at height z and extract its line exactly as bench/calib.py does. -> (u, v)."""
    rng = np.random.default_rng([seed, int(round((z + 100) * 100))])
    s = hs.flat_scene(z, rng)
    s.update({"plate_tilt_x_deg": tilt_x, "plate_tilt_y_deg": tilt_y})
    on, off, _ = hs.render(s, dev=dev or hs.DEVICE)
    p = baseline_steger.extract_line(on, off)
    u, v = np.asarray(p["u"], float), np.asarray(p["v"], float)
    keep = np.abs(v - np.median(v)) < 6
    return u[keep], v[keep]


def plate_set(heights, dev=None, seed=0, labels=None, cache=None):
    """-> u, v, z arrays for a list of plates. labels: the heights the student WRITES DOWN (default: true)."""
    U, V, Z = [], [], []
    for i, z in enumerate(heights):
        key = (z, seed)
        if cache is not None and key in cache:
            u, v = cache[key]
        else:
            u, v = plate_points(z, dev, seed)
            if cache is not None:
                cache[key] = (u, v)
        U.append(u); V.append(v); Z.append(np.full(len(u), float(z if labels is None else labels[i])))
    return np.concatenate(U), np.concatenate(V), np.concatenate(Z)


_LINES = {}


def true_line(z, dev=None, ucols=UCOLS, tilt_x=0.0, tilt_y=0.0):
    """Exact row of the laser line at the columns ucols for a flat plate at height z."""
    dev = dev or hs.DEVICE
    key = (float(z), tilt_x, tilt_y, repr(sorted(dev.items())))
    if key not in _LINES:
        Hh, _, _, _ = hs.trace_laser({"flat_z": float(z), "grooves": [], "plate_tilt_x_deg": tilt_x,
                                      "plate_tilt_y_deg": tilt_y}, dev)
        u, v = hs.project(Hh, 320, 240, dev)
        o = np.argsort(u)
        _LINES[key] = (u[o], v[o])
    return np.interp(ucols, *_LINES[key])


def depth_errors(model, dev=None, surf_z=SURF_Z, depths=DEPTHS, ucols=UCOLS):
    """(len(depths), len(ucols)) array: model depth minus true depth for a flat surface at surf_z and a
    flat floor d below it, on the exact line of device `dev`. Isolates the calibration from line noise."""
    hs_ = model.height(ucols, true_line(surf_z, dev, ucols))
    return np.stack([hs_ - model.height(ucols, true_line(surf_z - d, dev, ucols)) - d for d in depths])


def height_errors(model, dev=None, zs=np.arange(4.0, -14.01, -1.5), ucols=UCOLS):
    """Model height minus true height over the working volume (+4 .. -14 mm)."""
    return np.stack([model.height(ucols, true_line(z, dev, ucols)) - z for z in zs])


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def table(rows, header):
    """Markdown table as a string (rows: lists of str / float)."""
    f = lambda x: x if isinstance(x, str) else f"{x:.3f}"
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(f(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
