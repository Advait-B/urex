"""pk_fir2d_nobg.py - pk_fir2d with NO background removal (ablation).

Pipeline (algos/peakcommon.py): background removal -> up to 3 candidate peaks per column ->
sub-pixel refinement with this detector -> the baseline's laser-chain selection.
Parameters were chosen on slices of the train set (experiments/peak-detectors/tune.py).
"""
from . import peakcommon as pc

NAME = "pk_fir2d_nobg"
USES_OFF = False
CFG = {"detector": "fir", "sigma_v": 1.5, "sigma_u": 1.5, "bg": "none"}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
