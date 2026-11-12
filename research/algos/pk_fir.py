"""pk_fir.py - matched filter: each column convolved with a Gaussian (sigma 1.5 px), parabola through the top 3 samples.

Pipeline (algos/peakcommon.py): background removal -> up to 3 candidate peaks per column ->
sub-pixel refinement with this detector -> the baseline's laser-chain selection.
Parameters were chosen on slices of the train set (experiments/peak-detectors/tune.py).
"""
from . import peakcommon as pc

NAME = "pk_fir"
USES_OFF = False
CFG = {"detector": "fir", "sigma_v": 1.5}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
