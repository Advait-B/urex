"""calib_shims_phys5.py - physical model on the same 10 shim plates 0 .. -20 mm as calib_shims_poly23."""
from . import calib_fit

NAME = "calib_shims_phys5"
USES_OFF = False
measure = calib_fit.make_measure(calib_fit.load("phys5_shims", "phys5", calib_fit.SHIMS))
