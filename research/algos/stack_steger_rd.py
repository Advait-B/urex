"""stack_steger_rd.py - baseline Steger line + rd_final depth stage (same as rd_final, for the ablation) (thin composition, see algos/stackcommon.py and notes/best-stack.md)."""
from algos import stackcommon as sc
from algos.baseline_steger import extract_line  # noqa: F401

NAME = "stack_steger_rd"
USES_OFF = False


def measure(gray, calib=None, off=None):
    return sc.measure(gray, calib, off, line_fn=extract_line, depth="rd")
