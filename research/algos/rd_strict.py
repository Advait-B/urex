"""rd_strict.py - baseline line extractor + the recommended depth stage, but grooves whose floor was not seen (lower bounds) are not reported.
(see algos/robust_depth.py and notes/robust-depth.md)."""
from algos import baseline_steger, robust_depth

NAME = "rd_strict"
USES_OFF = False
CONFIG = {'floor': 'dbscan', 'detect': 'shape', 'lo_k': 3.5}
DROP_LOWER_BOUNDS = True
extract_line = baseline_steger.extract_line


def measure(gray, calib=None, off=None):
    line = extract_line(gray, off)
    out = robust_depth.stage(line["u"], line["v"], calib, **CONFIG)
    if DROP_LOWER_BOUNDS:                         # report only grooves whose flat floor was seen
        out["grooves"] = [g for g in out["grooves"] if not g["lower_bound"]]
    out["line"] = line
    return out
