"""stack_dp_rd_gate.py - THE RECOMMENDED SINGLE-FRAME STACK: dp_dense line + rd_final depth stage + 25 mm cap
+ lower-bound verdict rule + quality gate (frame gate of gated_baseline, unchanged; risk gate re-fitted on sim
`train` for this stack, models/stack_risk_lr.json). "usable" < 0.5 means "retake". See notes/best-stack.md."""
from algos import stackcommon as sc
from algos.dp_dense import extract_line  # noqa: F401

NAME = "stack_dp_rd_gate"
USES_OFF = False


def measure(gray, calib=None, off=None):
    return sc.measure(gray, calib, off, line_fn=extract_line, depth="rd", cap=True, lb=True,
                      frame_gate=True, risk_model="stack_risk_lr")
