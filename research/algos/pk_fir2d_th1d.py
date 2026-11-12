"""pk_fir2d_th1d.py - pk_fir2d with a vertical-only 15x1 top-hat (cheaper on an MCU) instead of the 15 px ellipse.

Pipeline (algos/peakcommon.py): background removal -> up to 3 candidate peaks per column ->
sub-pixel refinement with this detector -> the baseline's laser-chain selection.
Parameters were chosen on slices of the train set (experiments/peak-detectors/tune.py).
"""
from . import peakcommon as pc

NAME = "pk_fir2d_th1d"
USES_OFF = False
CFG = {"detector": "fir", "sigma_v": 1.5, "sigma_u": 1.5, "bg": "tophat1d"}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
