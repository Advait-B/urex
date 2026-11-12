"""calib_phys5_geo.py - calib_phys5 + the two geometry fixes of the depth stage (surface fitted in mm across the tread, tyre curvature)."""
from . import calib_fit

NAME = "calib_phys5_geo"
USES_OFF = False
measure = calib_fit.make_measure(calib_fit.load("phys5_15plates", "phys5"), curvature=True, mm_space=True)
