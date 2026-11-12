"""dp_graph.py - the same idea as a shortest path on the pixel grid (skimage.graph.MCP_Geometric).

Every pixel costs (1 - reward): walking along the laser is free, walking through the dark costs 1 per
column, and a vertical step costs only V_STEP of that (so dropping into a groove is affordable). The
cheapest route from the left edge to the right edge is the line. No hand-written recursion, and the path
may run straight down, or even backwards, so it can follow a near-vertical wall or a hook; the price is
that there is no separate "smooth" and "jump" cost to tune, and it is slower.
"""
import numpy as np
from skimage.graph import MCP_Geometric

from . import dp_dense

NAME = "dp_graph"
USES_OFF = False
P = {**dp_dense.P, "v_step": 0.08, "floor": 0.02}


def extract_line(gray, off=None, p=P):
    strength, contrast, du, dv = dp_dense.ridge_maps(gray, p["sigma"], p["tophat"])
    reward, scale = dp_dense.reward_map(strength, contrast, p)
    H, W = reward.shape
    cost = np.full((H, W + 2), 1e-6)                    # a free column on each side: start and end at any row
    cost[:, 1:-1] = 1 - reward + p["floor"]
    g = MCP_Geometric(cost, fully_connected=True, sampling=(p["v_step"], 1.0))
    g.find_costs([(0, 0)], [(0, W + 1)])
    path = np.array(g.traceback((0, W + 1)))
    v, u = path[:, 0], path[:, 1] - 1
    k = (u >= 0) & (u < W)
    v, u = v[k], u[k]
    k = (reward[v, u] >= p["keep_rel"]) & (np.abs(du[v, u]) <= 0.6) & (np.abs(dv[v, u]) <= 0.6)
    v, u = v[k], u[k]
    mid, width = dp_dense.band_centre(contrast, v, u)   # wide (saturated) line: centre of the band, as in dp_dense
    wide = width > p["wide_px"]
    return {"u": np.where(wide, u, u + du[v, u]), "v": np.where(wide, mid, v + dv[v, u]), "strength": strength[v, u]}
