"""dp_smooth.py - dp_dense + an edge-preserving smoother on the path points.

Model: within one piece of line the height changes smoothly, v[c+1] - 2 v[c] + v[c-1] = small noise, and
each measured point is the true height + noise. The best estimate under that model is what a Kalman
filter + RTS smoother computes; here the same answer is obtained in one go by solving the equivalent
least-squares problem   minimise  sum (z - v)^2 + LAM * sum (second difference of z)^2   (a banded system).
"Edge-preserving": the smoothness term is switched off across every break (a missing column or a step of
more than BREAK_PX), so groove edges are not rounded. BREAK_PX = inf gives the plain smoother, to measure
how much that smears. Wall points are passed through untouched.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import dp_dense

NAME = "dp_smooth"
USES_OFF = False
P = {**dp_dense.P, "lam": 4.0, "break_px": 1.0}


def smooth(u, v, lam, break_px):
    n = len(v)
    if n < 5:
        return v
    same = (np.abs(np.diff(v)) <= break_px) & (np.diff(u) < 1.6)       # neighbours belong to the same piece
    link = (same[:-1] & same[1:]).astype(float)                         # triples (i-1, i, i+1) inside one piece
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    A = sparse.identity(n) + lam * (D.T @ sparse.diags(link) @ D)
    return spsolve(A.tocsc(), v)


def extract_line(gray, off=None, p=P):
    pts = dp_dense.extract_line(gray, off, p)
    n = pts["n_path"]
    v = pts["v"].copy()
    v[:n] = smooth(pts["u"][:n], v[:n], p["lam"], p["break_px"])
    return {**pts, "v": v}
