"""variants.py - every (difference mode, detector) combination as a harness-style measure()."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from algos import baseline_steger as base  # noqa: E402
from algos import onoff_common as oc  # noqa: E402
from bench import depth  # noqa: E402


def make(mode, det, **kw):
    """mode: single | plain | gain | reg | cons | robust ; det: steger | cog"""
    def extract(gray, off=None):
        if mode == "single":
            if det == "steger":
                return base.extract_line(gray)
            return oc.cog_line(oc.sf.tophat(np.ascontiguousarray(gray), 15))     # control: COG without an off frame
        d = oc.difference(gray, off, mode, **kw)
        return oc.steger_line(d) if det == "steger" else oc.cog_line(d)

    def measure(gray, calib=None, off=None):
        p = extract(gray, off)
        out = depth.measure_from_line(p["u"], p["v"], calib)
        out["line"] = p
        return out
    return measure


KEYS = ["line_f1", "line_rmse_px", "depth_mae_mm", "depth_p95_mm", "depth_bias_mm", "miss_rate", "false_per_frame",
        "verdict_acc", "false_pass_rate"]


def row(r):
    return " ".join(f"{r[k]:8.3f}" if r[k] is not None else "       -" for k in KEYS)
