"""cnn_line_noaug.py - algos/cnn_line.py with readout "gated" and weights models/cnn_line_noaug.pt."""
import os

from . import cnn_line

NAME = "cnn_line_noaug"
USES_OFF = False
_W = os.path.join(os.path.dirname(cnn_line.WEIGHTS), "cnn_line_noaug.pt")


def extract_line(gray, off=None):
    return cnn_line.extract_line(gray, readout="gated", path=_W)
