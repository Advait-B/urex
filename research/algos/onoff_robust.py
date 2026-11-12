"""onoff_robust.py - laser-on / laser-off: (3) gain match + phase-correlation registration + conservative (dilated-off) subtraction, then the baseline detector.

Falls back to the single-frame baseline when no off frame is given (real photos).
"""
from . import baseline_steger as base
from . import onoff_common as oc

NAME = "onoff_robust"
USES_OFF = True
MODE = "robust"


def extract_line(gray, off=None):
    if off is None:
        return base.extract_line(gray)
    d = oc.difference(gray, off, MODE)
    return oc.steger_line(d)
