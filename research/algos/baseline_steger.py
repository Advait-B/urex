"""baseline_steger.py - the 12 Nov pipeline as the reference algorithm.

Line: top-hat + Steger sub-pixel ridge points + laser-chain selection (pipeline/single_frame.py).
Depth: the default stage in bench/depth.py. Uses only the laser-on frame.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pipeline")))
import single_frame as sf  # noqa: E402

NAME = "baseline_steger"
USES_OFF = False


def extract_line(gray, off=None):
    """single_frame expects the line top-to-bottom, so work on the transpose."""
    g = np.ascontiguousarray(gray.T)
    c = sf.DEFAULTS
    pts = sf.steger_points(g, c["sigma"], c["tophat"], c["min_contrast"], c["rel_strength"])
    keep, _ = sf.select_chain(pts, g.shape, c["chain_gap"], c["keep_within_px"], c["min_piece"], c["min_rel_contrast"])
    return {"u": pts["y"][keep], "v": pts["x"][keep], "strength": pts["strength"][keep]}
