"""gate_features.py - the numbers the quality gate looks at (shared by training scripts and gated_baseline).

One Steger pass (pipeline/single_frame.analyse on the transposed image, exactly the baseline line),
then two groups of features:
  FRAME  the 12 hand-made features of pipeline/frame_classifier.py ("is there a clean laser line?")
  FIT    what the default depth stage made of that line, always in px so that it means the same with or
         without a calibration ("does the fit look like a tread with grooves?")
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pipeline")))
import frame_classifier as fc  # noqa: E402
import single_frame as sf  # noqa: E402

from bench import depth  # noqa: E402

FRAME = fc.FEATURES
FIT = ["n_grooves", "max_depth_px", "depth_spread_px", "min_groove_pts", "min_floor_pts", "max_floor_std_px",
       "fit_inlier_frac", "resid_std_px", "above_frac", "limbo_frac", "u_coverage", "max_gap_frac", "multi_frac",
       "abs_curv", "abs_slope", "edge_groove", "img_sat_frac", "bg_grad"]
ALL = FRAME + FIT


def analyse(gray):
    """-> (line dict, sf.analyse result). Same points as algos.baseline_steger.extract_line."""
    res = sf.analyse(np.ascontiguousarray(gray.T))
    keep = res["_keep"]
    return {"u": res["_y"], "v": res["_x"], "strength": res["_pts"]["strength"][keep]}, res


def features(gray, line=None, res=None):
    if res is None:
        line, res = analyse(gray)
    f = fc.features(np.ascontiguousarray(gray.T), res)
    h, w = gray.shape
    u, v = np.asarray(line["u"], float), np.asarray(line["v"], float)
    out = depth.measure_from_line(u, v, None)
    g = out["grooves"]
    z = dict.fromkeys(FIT, 0.0)
    z["img_sat_frac"] = float((gray >= 250).mean())
    small = gray[::8, ::8].astype(float)
    z["bg_grad"] = float(np.hypot(np.ptp(small.mean(0)), np.ptp(small.mean(1))))
    if out["surface"] is not None:
        coef = np.array(out["surface"])
        r = v - np.polyval(coef, u)
        inl = np.abs(r) < depth.PX["ransac"]
        cols = np.round(u).astype(int)
        uc = np.unique(cols)
        floor_n, floor_sd = [], []
        for gr in g:
            rr = r[(u >= gr["u0"]) & (u <= gr["u1"]) & (r > depth.PX["min_depth"])]
            band = rr[np.abs(rr - gr["depth"]) <= 2 * depth.PX["band"]]
            floor_n.append(len(band)); floor_sd.append(float(band.std()) if len(band) > 1 else 3.0)
        vr = np.array([np.ptp(v[cols == c]) for c in uc])
        z.update({
            "n_grooves": float(len(g)),
            "max_depth_px": max([x["depth"] for x in g], default=0.0),
            "depth_spread_px": float(np.ptp([x["depth"] for x in g])) if g else 0.0,
            "min_groove_pts": float(min([x["n"] for x in g], default=0)),
            "min_floor_pts": float(min(floor_n, default=0)),
            "max_floor_std_px": float(max(floor_sd, default=0.0)),
            "fit_inlier_frac": float(inl.mean()),
            "resid_std_px": float(r[inl].std()) if inl.sum() > 2 else 3.0,
            "above_frac": float((r < -depth.PX["min_depth"]).mean()),
            "limbo_frac": float(((np.abs(r) >= depth.PX["ransac"]) & (r <= depth.PX["min_depth"]) & (r >= -depth.PX["min_depth"])).mean()),
            "u_coverage": float(len(uc) / w),
            "max_gap_frac": float(np.max(np.diff(np.r_[0, uc, w - 1])) / w),
            "multi_frac": float((vr > 3).mean()),
            "abs_curv": float(abs(coef[0]) * 1e3), "abs_slope": float(abs(2 * coef[0] * w / 2 + coef[1])),
            "edge_groove": float(any(x["u0"] < 6 or x["u1"] > w - 7 for x in g)),
        })
    f.update(z)
    return f, out
