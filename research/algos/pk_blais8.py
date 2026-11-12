"""pk_blais8.py - Blais-Rioux 8th-order derivative filter, zero crossing by linear interpolation.

Pipeline (algos/peakcommon.py): background removal -> up to 3 candidate peaks per column ->
sub-pixel refinement with this detector -> the baseline's laser-chain selection.
Parameters were chosen on slices of the train set (experiments/peak-detectors/tune.py).
"""
from . import peakcommon as pc

NAME = "pk_blais8"
USES_OFF = False
CFG = {"detector": "blais8"}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
