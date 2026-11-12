"""calib_shims_poly23.py - pipeline/calibrate.py as written: poly(2,3) on 10 shim plates 0 .. -20 mm (nothing above the feet plane)."""
from . import calib_fit

NAME = "calib_shims_poly23"
USES_OFF = False
measure = calib_fit.make_measure(calib_fit.load("poly23_shims", "poly23", calib_fit.SHIMS))
