"""pk_fir2d_rows.py - pk_fir2d plus a second pass along the ROWS for near-vertical stretches.

Same matched-filter response; a row peak is kept only where the response is at least twice as
curved across u as across v and is not on the shoulder of a level line (peakcommon.extract).
Aim: groove walls / lug steps on the real photos, where one point per column is too sparse for
the chain selection. rows_flank was picked with the 12 real reference frames in view (leak, see notes).
"""
from . import peakcommon as pc

NAME = "pk_fir2d_rows"
USES_OFF = False
CFG = {"detector": "fir", "sigma_v": 1.5, "sigma_u": 1.5, "rows": True, "rows_ratio": 2.0, "rows_flank": 1.0}


def extract_line(gray, off=None):
    return pc.extract(gray, CFG)
