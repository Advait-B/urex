"""pk_steger_s1.py - the baseline's Steger ridge detector at sigma = 1.0 instead of 1.5.

The simulated line has a Gaussian sigma of 0.8-1.3 px on the ideal set; Steger's own rule of
thumb is sigma >= w / sqrt(3) for a line of half-width w, so 1.5 over-smooths the thin lines.
Everything else (top-hat, thresholds, chain selection) is the baseline's.
"""
import numpy as np

from .baseline_steger import sf

NAME = "pk_steger_s1"
USES_OFF = False
SIGMA = 1.0


def extract_line(gray, off=None):
    g = np.ascontiguousarray(gray.T)
    c = sf.DEFAULTS
    pts = sf.steger_points(g, SIGMA, c["tophat"], c["min_contrast"], c["rel_strength"])
    keep, _ = sf.select_chain(pts, g.shape, c["chain_gap"], c["keep_within_px"], c["min_piece"], c["min_rel_contrast"])
    return {"u": pts["y"][keep], "v": pts["x"][keep], "strength": pts["strength"][keep]}
