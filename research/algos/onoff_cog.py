"""onoff_cog.py - laser-on / laser-off: (2) plain on - off, then a per-column centre of gravity (no top-hat, no Steger).

Falls back to the single-frame baseline when no off frame is given (real photos).
"""
from . import baseline_steger as base
from . import onoff_common as oc

NAME = "onoff_cog"
USES_OFF = True
MODE = "plain"


def extract_line(gray, off=None):
    if off is None:
        return base.extract_line(gray)
    d = oc.difference(gray, off, MODE)
    return oc.cog_line(d)
