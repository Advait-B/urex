"""real_check.py - what do the learners say about the real hand-held photos? (no mm truth exists)

    python experiments/end-to-end/real_check.py      -> experiments/end-to-end/real_results.json
  * 6 truck-tyre frames with a reference depth in px: is the predicted mm at least monotonic (Spearman)?
  * 67 frames a person would not measure (people, benches, ceilings): does the model still report a depth?
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from algos import baseline_steger, e2e_cnn, e2e_gbm  # noqa: E402
from bench import datasets, run  # noqa: E402

_, base = run.resolve(baseline_steger)
rows = []
for it in datasets.load_real():
    c = e2e_cnn.measure(it["gray"])["grooves"]
    b = base(it["gray"], None, None)["grooves"]
    rows.append({"name": it["name"], "usable": it["usable"], "scene": it["scene"], "feature": it["feature"],
                 "ref_px": it["ref"]["depth_px"] if it["ref"] and it["ref"]["mode"] == "groove" else None,
                 "cnn_n": len(c), "cnn_max_mm": max((g["depth"] for g in c), default=None),
                 "cnn_conf": max((g["conf"] for g in c), default=None),
                 "gbm_mm": e2e_gbm.predict_min_depth(it["gray"]),
                 "base_n": len(b), "base_max_px": max((g["depth"] for g in b), default=None)})
ref = [r for r in rows if r["ref_px"] is not None]
bad = [r for r in rows if not r["usable"]]
good = [r for r in rows if r["usable"]]
out = {"rows": rows}
print("6 reference frames (ref px | cnn mm | gbm mm | baseline px):")
for r in ref:
    print(f"  {r['name']}  {r['ref_px']:6.2f} | {r['cnn_max_mm']} | {r['gbm_mm']:.2f} | {r['base_max_px']}")
x = [r["ref_px"] for r in ref]
for k in ("cnn_max_mm", "gbm_mm", "base_max_px"):
    y = [r[k] if r[k] is not None else 0.0 for r in ref]
    rho, p = spearmanr(x, y)
    out["spearman_" + k] = [float(rho), float(p)]
    print(f"Spearman {k}: rho {rho:.3f} (p {p:.2f}, n 6)")
for name, grp in (("unusable", bad), ("usable", good)):
    cn = [r["cnn_max_mm"] for r in grp if r["cnn_n"]]
    gb = [r["gbm_mm"] for r in grp]
    s = {"n": len(grp), "cnn_reports_depth": len(cn) / len(grp),
         "cnn_depth_pcts_10_50_90": np.percentile(cn, [10, 50, 90]).round(2).tolist() if cn else None,
         "cnn_conf_median": float(np.median([r["cnn_conf"] for r in grp if r["cnn_n"]])) if cn else None,
         "cnn_says_fail": float(np.mean([r["cnn_n"] > 0 and r["cnn_max_mm"] < 1.6 for r in grp])),
         "gbm_reports_depth": 1.0, "gbm_depth_pcts_10_50_90": np.percentile(gb, [10, 50, 90]).round(2).tolist(),
         "gbm_says_pass": float(np.mean(np.array(gb) >= 1.6)),
         "baseline_reports_depth": float(np.mean([r["base_n"] > 0 for r in grp]))}
    out[name] = s
    print(name, json.dumps(s))
print("unusable by scene (cnn reports / n):")
for sc in sorted({r["scene"] for r in bad}):
    g = [r for r in bad if r["scene"] == sc]
    print(f"  {sc:20s} {sum(r['cnn_n'] > 0 for r in g)}/{len(g)}")
json.dump(out, open("experiments/end-to-end/real_results.json", "w"), indent=1)
