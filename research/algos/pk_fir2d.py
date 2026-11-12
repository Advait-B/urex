"""pk_fir2d.py - matched filter as pk_fir, plus a Gaussian (sigma 1.5 px) ALONG the line before the parabola. RECOMMENDED.

Pipeline (algos/peakcommon.py): background removal -> up to 3 candidate peaks per column ->
sub-pixel refinement with this detector -> the baseline's laser-chain selection.
Parameters were chosen on slices of the train set (experiments/peak-detectors/tune.py).
"""
from . import peakcommon as pc

NAME = "pk_fir2d"
USES_OFF = False
CFG = {"detector": "fir", "sigma_v": 1.5, "sigma_u": 1.5}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
