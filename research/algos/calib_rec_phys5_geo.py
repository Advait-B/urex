"""calib_rec_phys5_geo.py - the recommended procedure: 8 plates (+4, +2, 0, -3, -6, -10, -14, -18), physical model, both geometry fixes."""
from . import calib_fit

NAME = "calib_rec_phys5_geo"
USES_OFF = False
measure = calib_fit.make_measure(calib_fit.load("phys5_recommended", "phys5", calib_fit.RECOMMENDED), curvature=True, mm_space=True)
