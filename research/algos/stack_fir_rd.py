"""stack_fir_rd.py - pk_fir2d_rows line (fast matched filter) + rd_final depth stage (thin composition, see algos/stackcommon.py and notes/best-stack.md)."""
from algos import stackcommon as sc
from algos.pk_fir2d_rows import extract_line  # noqa: F401

NAME = "stack_fir_rd"
USES_OFF = False


def measure(gray, calib=None, off=None):
    return sc.measure(gray, calib, off, line_fn=extract_line, depth="rd")
