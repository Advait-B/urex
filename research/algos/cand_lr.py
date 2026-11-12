"""cand_lr.py - cand_clf.py with MODEL = "lr" (see that file)."""
from . import cand_clf as _c

NAME = "cand_lr"
USES_OFF = False


def extract_line(gray, off=None):
    _c.MODEL = "lr"
    return _c.extract_line(gray)
