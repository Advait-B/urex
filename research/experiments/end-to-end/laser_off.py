"""laser_off.py - what is the CNN looking at? Score it on the laser-OFF frame of every test scene.
With no laser line there is no triangulation signal at all, so a trustworthy instrument must report nothing.

    python experiments/end-to-end/laser_off.py      -> experiments/end-to-end/laser_off_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from algos import baseline_steger, e2e_cnn  # noqa: E402
from bench import calib, datasets, run  # noqa: E402

cal = calib.default_calib()
_, base = run.resolve(baseline_steger)
res = {}
for name in ("ideal", "stress", "legal"):
    items = datasets.load_sim(name)
    for algo, fn in (("e2e_cnn", e2e_cnn.measure), ("baseline", base)):
        r = run.eval_sim(lambda g, c=None, off=None: fn(off, c, None), items, cal)     # swap in the laser-off frame
        t, p = np.array(r["_true_mm"]), np.array(r["_pred_mm"])
        res[f"{name}/{algo}"] = {k: r[k] for k in ("depth_mae_mm", "depth_bias_mm", "miss_rate", "false_per_frame", "verdict_acc",
                                                  "false_pass_rate")}
        res[f"{name}/{algo}"]["corr_pred_true"] = float(np.corrcoef(t, p)[0, 1]) if len(t) > 2 else None
        print(name, algo, {k: None if v is None else round(v, 3) for k, v in res[f"{name}/{algo}"].items()})
json.dump(res, open("experiments/end-to-end/laser_off_results.json", "w"), indent=1)
