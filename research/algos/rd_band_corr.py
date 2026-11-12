"""rd_band_corr.py - baseline line extractor + the bench default stage (RANSAC, default detection, anchored band) plus only the geometric bias correction.
(see algos/robust_depth.py and notes/robust-depth.md)."""
from algos import baseline_steger, robust_depth

NAME = "rd_band_corr"
USES_OFF = False
CONFIG = {'floor': 'band'}
DROP_LOWER_BOUNDS = False
extract_line = baseline_steger.extract_line


def measure(gray, calib=None, off=None):
    line = extract_line(gray, off)
    out = robust_depth.stage(line["u"], line["v"], calib, **CONFIG)
    if DROP_LOWER_BOUNDS:                         # report only grooves whose flat floor was seen
        out["grooves"] = [g for g in out["grooves"] if not g["lower_bound"]]
    out["line"] = line
    return out
