"""cand_hgb_local.py - cand_clf.py with MODEL = "hgb_local" (see that file)."""
from . import cand_clf as _c

NAME = "cand_hgb_local"
USES_OFF = False


def extract_line(gray, off=None):
    _c.MODEL = "hgb_local"
    return _c.extract_line(gray)
