"""stack_dp_rd_risk.py - dp_dense line + rd_final depth stage + 25 mm cap
+ lower-bound verdict rule + RISK gate only (re-fitted on sim `train` for this stack, models/stack_risk_lr.json; no
frame gate, so no second Steger pass). "usable" < 0.5 means "retake". See notes/best-stack.md."""
from algos import stackcommon as sc
from algos.dp_dense import extract_line  # noqa: F401

NAME = "stack_dp_rd_risk"
USES_OFF = False


def measure(gray, calib=None, off=None):
    return sc.measure(gray, calib, off, line_fn=extract_line, depth="rd", cap=True, lb=True,
                      risk_model="stack_risk_lr")
