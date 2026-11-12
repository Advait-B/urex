"""cnn_line_steger.py - algos/cnn_line.py with readout "steger" and weights models/cnn_line.pt."""
import os

from . import cnn_line

NAME = "cnn_line_steger"
USES_OFF = False
_W = os.path.join(os.path.dirname(cnn_line.WEIGHTS), "cnn_line.pt")


def extract_line(gray, off=None):
    return cnn_line.extract_line(gray, readout="steger", path=_W)
