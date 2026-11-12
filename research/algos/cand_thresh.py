"""cand_thresh.py - cand_clf.py with MODEL = "thresh" (see that file)."""
from . import cand_clf as _c

NAME = "cand_thresh"
USES_OFF = False


def extract_line(gray, off=None):
    _c.MODEL = "thresh"
    return _c.extract_line(gray)
