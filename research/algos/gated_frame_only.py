"""gated_frame_only.py - gated_baseline with stage 1 only (the frame gate), to show what each stage does."""
from algos import gated_baseline as gb

NAME = "gated_frame_only"
USES_OFF = False


def measure(gray, calib=None, off=None):
    return gb.measure(gray, calib, off, stage2=False)
