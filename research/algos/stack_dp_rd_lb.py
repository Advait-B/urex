"""stack_dp_rd_lb.py - dp_dense + rd_final + cap + lower-bound verdict rule (>= x mm: PASS if x >= 1.6, else retake) (thin composition, see algos/stackcommon.py and notes/best-stack.md)."""
from algos import stackcommon as sc
from algos.dp_dense import extract_line  # noqa: F401

NAME = "stack_dp_rd_lb"
USES_OFF = False


def measure(gray, calib=None, off=None):
    return sc.measure(gray, calib, off, line_fn=extract_line, depth="rd", cap=True, lb=True)
