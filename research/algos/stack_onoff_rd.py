"""stack_onoff_rd.py - laser on/off subtraction line (onoff_cog) + rd_final + cap + lower-bound rule (needs a laser-off frame) (thin composition, see algos/stackcommon.py and notes/best-stack.md)."""
from algos import stackcommon as sc
from algos.onoff_cog import extract_line  # noqa: F401

NAME = "stack_onoff_rd"
USES_OFF = True


def measure(gray, calib=None, off=None):
    return sc.measure(gray, calib, off, line_fn=extract_line, depth="rd", cap=True, lb=True)
