"""onoff_steger.py - laser-on / laser-off: (1) plain on - off, then the unchanged baseline detector.

Falls back to the single-frame baseline when no off frame is given (real photos).
"""
from . import baseline_steger as base
from . import onoff_common as oc

NAME = "onoff_steger"
USES_OFF = True
MODE = "plain"


def extract_line(gray, off=None):
    if off is None:
        return base.extract_line(gray)
    d = oc.difference(gray, off, MODE)
    return oc.steger_line(d)
