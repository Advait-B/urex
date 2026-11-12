"""gated_baseline.py - the baseline pipeline behind a two-stage quality gate ("give a depth or say retake").

  stage 1  FRAME gate: "is there a clean laser line on a tyre at all?"  Logistic regression on the 12 frame
           features, trained on the 86 labelled real photos (models/frame_lr.json).
  stage 2  RISK gate:  "is one of these depths probably wrong by > 0.5 mm?"  Logistic regression on 30 line / fit
           features, trained on the simulated `train` set only (models/risk_lr.json). It describes the HOLDER
           geometry, so it is applied only when a calibration is given (= the device); on the uncalibrated
           hand-held photos it rejects nearly everything (see notes) and is skipped.
When the gate says no, measure() returns usable < 0.5 and NO grooves: no number is better than a wrong one.
Variant: algos.gated_frame_only (stage 1 only).
"""
import json
import os

import numpy as np

from algos import gate_features as gf
from bench import depth

NAME = "gated_baseline"
USES_OFF = False
STAGE2 = True
_M = {k: json.load(open(os.path.join(os.path.dirname(__file__), "..", "models", k + ".json"))) for k in ("frame_lr", "risk_lr")}


def _prob(model, f):
    x = (np.array([f[k] for k in model["features"]]) - model["mean"]) / np.array(model["scale"])
    return float(1.0 / (1.0 + np.exp(-(x @ np.array(model["coef"]) + model["intercept"]))))


def gate(gray, line=None, res=None, calibrated=True, stage2=True):
    """-> (usable 0..1, p_frame, p_bad). usable >= 0.5 means: show the number."""
    f, _ = gf.features(gray, line, res)
    p_frame, p_bad = _prob(_M["frame_lr"], f), _prob(_M["risk_lr"], f)
    usable = p_frame
    if calibrated and stage2:
        t = _M["risk_lr"]["threshold"]                      # risk <= t  <=>  score >= 0.5
        trust = 0.5 + 0.5 * (t - p_bad) / t if p_bad <= t else 0.5 * (1 - p_bad) / (1 - t)
        usable = min(p_frame, trust)
    return usable, p_frame, p_bad


def measure(gray, calib=None, off=None, stage2=None):
    line, res = gf.analyse(gray)                            # the baseline line, one Steger pass
    out = depth.measure_from_line(line["u"], line["v"], calib)
    usable, p_frame, p_bad = gate(gray, line, res, calib is not None, STAGE2 if stage2 is None else stage2)
    out.update({"line": line, "usable": usable, "p_frame": p_frame, "p_bad": p_bad})
    if usable < 0.5 or not out["grooves"]:
        out["withheld"], out["grooves"] = out["grooves"], []
        out["usable"] = min(usable, 0.49)                   # no groove found is also a "retake"
    return out
