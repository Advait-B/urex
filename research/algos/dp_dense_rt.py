"""dp_dense_rt.py - dp_dense + the run test: after the Viterbi path is found, a short isolated run of
line that is tilted against the rest of the line is dropped (a glint picked up where the laser is hidden).
Helps on the simulator, where groove floors are flat; hurts on real photos, where the line hooks."""
from . import dp_dense

NAME = "dp_dense_rt"
USES_OFF = False
P = {**dp_dense.P, "run_test": True}


def extract_line(gray, off=None):
    return dp_dense.extract_line(gray, off, P)
