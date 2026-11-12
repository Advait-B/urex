"""pk_cog.py - centre of gravity in a +-4 px window down each column (window minimum subtracted).

Pipeline (algos/peakcommon.py): background removal -> up to 3 candidate peaks per column ->
sub-pixel refinement with this detector -> the baseline's laser-chain selection.
Parameters were chosen on slices of the train set (experiments/peak-detectors/tune.py).
"""
from . import peakcommon as pc

NAME = "pk_cog"
USES_OFF = False
CFG = {"detector": "cog", "half": 4}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
