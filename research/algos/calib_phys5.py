"""calib_phys5.py - baseline line + PHYSICAL calibration (pinhole + k1 + laser plane, 5 numbers) fitted to the 15 'calib' plates."""
from . import calib_fit

NAME = "calib_phys5"
USES_OFF = False
measure = calib_fit.make_measure(calib_fit.load("phys5_15plates", "phys5"))
