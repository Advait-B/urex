"""stack_dp_rd_cap.py - dp_dense line + rd_final depth stage + 25 mm plausibility cap (thin composition, see algos/stackcommon.py and notes/best-stack.md)."""
from algos import stackcommon as sc
from algos.dp_dense import extract_line  # noqa: F401

NAME = "stack_dp_rd_cap"
USES_OFF = False


def measure(gray, calib=None, off=None):
    return sc.measure(gray, calib, off, line_fn=extract_line, depth="rd", cap=True)
