"""pk_blais4.py - Blais-Rioux 4th-order derivative filter, zero crossing by linear interpolation.

Pipeline (algos/peakcommon.py): background removal -> up to 3 candidate peaks per column ->
sub-pixel refinement with this detector -> the baseline's laser-chain selection.
Parameters were chosen on slices of the train set (experiments/peak-detectors/tune.py).
"""
from . import peakcommon as pc

NAME = "pk_blais4"
USES_OFF = False
CFG = {"detector": "blais4"}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
