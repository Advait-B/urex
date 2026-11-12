"""calib_default_geo.py - baseline line + the harness's DEFAULT calibration + the two geometry fixes of the depth stage (CAD numbers only)."""
from . import calib_fit

NAME = "calib_default_geo"
USES_OFF = False
measure = calib_fit.make_measure(None, curvature=True, mm_space=True)
