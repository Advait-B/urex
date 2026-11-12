"""dp_cand.py - the textbook version: Viterbi over a few CANDIDATE peaks per column + a "no line" state.

Same ridge map, reward and jump penalty as dp_dense, but each column offers only its K strongest ridge
peaks (local maxima of the reward along the column) and one gap state. The gap state has no row, so it
cannot remember where the line was: leaving the line through a gap and coming back anywhere costs only
GAP_IN + GAP_OUT, whatever the distance. dp_dense keeps the row through a gap; this file measures what
that is worth. Sub-pixel points are produced exactly as in dp_dense.
"""
import numpy as np

from . import dp_dense

NAME = "dp_cand"
USES_OFF = False
P = {**dp_dense.P, "K": 5, "gap_in": 1.0, "gap_out": 0.5, "gap_stay": 0.05}


def candidates(reward, K, min_r=0.05):
    """-> rows (K x W, int), value (K x W; 0 where a column has fewer than K peaks)"""
    r = np.pad(reward, ((1, 1), (0, 0)))
    peak = np.where((r[1:-1] >= r[:-2]) & (r[1:-1] > r[2:]) & (reward > min_r), reward, 0)
    rows = np.argsort(-peak, axis=0)[:K]
    return rows, np.take_along_axis(peak, rows, axis=0)


def viterbi(rows, val, p, H):
    K, W = rows.shape
    cost = np.r_[-val[:, 0], 0.0]                       # K candidates + the gap state (index K)
    back = np.zeros((W, K + 1), np.int32)
    for c in range(1, W):
        d = np.abs(rows[:, c][:, None] - rows[:, c - 1][None, :])              # to (rows) x from (cols)
        pen = np.where(d <= p["smooth_px"], p["A"] * d, p["J"] + p["B"] * d)
        pen[:, val[:, c - 1] <= 0] = 1e9                                       # empty candidate slots
        trans = np.empty((K + 1, K + 1))
        trans[:K, :K] = cost[None, :K] + pen
        trans[:K, K] = cost[K] + p["gap_out"]                                  # gap -> line
        trans[K, :K] = cost[:K] + p["gap_in"]                                  # line -> gap
        trans[K, K] = cost[K] + p["gap_stay"]
        back[c] = trans.argmin(axis=1)
        cost = trans[np.arange(K + 1), back[c]] - np.r_[np.where(val[:, c] > 0, val[:, c], -1e9), 0.0]
    state = np.empty(W, np.int32)
    state[-1] = cost.argmin()
    for c in range(W - 1, 0, -1):
        state[c - 1] = back[c, state[c]]
    return state


def track(gray, p=P):
    strength, contrast, du, dv = dp_dense.ridge_maps(gray, p["sigma"], p["tophat"])
    reward, scale = dp_dense.reward_map(strength, contrast, p)
    H, W = reward.shape
    rows, val = candidates(reward, p["K"])
    state = viterbi(rows, val, p, H)
    on = state < p["K"]
    path = rows[np.minimum(state, p["K"] - 1), np.arange(W)]
    if on.any():                                         # in a gap, carry the last row forward (for the run logic only)
        last = np.maximum.accumulate(np.where(on, np.arange(W), -1))
        path = np.where(last >= 0, path[np.maximum(last, 0)], path[np.argmax(on)])
    on &= reward[path, np.arange(W)] >= p["keep_rel"]
    on = dp_dense.filter_runs(on, path, p)
    return {"path": path, "on": on, "reward": reward, "du": du, "dv": dv, "scale": scale, "strength": strength,
            "contrast": contrast}


def extract_line(gray, off=None):
    return dp_dense.points_near_path(track(gray), P)
