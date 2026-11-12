"""pk_gauss3.py - 3-point Gaussian (log-parabola) fit on the raw background-removed pixels.

Pipeline (algos/peakcommon.py): background removal -> up to 3 candidate peaks per column ->
sub-pixel refinement with this detector -> the baseline's laser-chain selection.
Parameters were chosen on slices of the train set (experiments/peak-detectors/tune.py).
"""
from . import peakcommon as pc

NAME = "pk_gauss3"
USES_OFF = False
CFG = {"detector": "gauss3"}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
