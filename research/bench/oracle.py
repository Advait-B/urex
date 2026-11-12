"""oracle.py - feed the TRUE line into the default depth stage. This is the best any line
extractor can do with the shared depth stage, so it separates 'line errors' from 'depth-stage errors'.
    python -m bench.oracle       -> results/oracle_line.json
"""
import json

from . import calib, datasets, depth, run

cal = calib.default_calib()
res = {"name": "oracle_line", "module": "bench.oracle", "uses_off": False, "limit": None, "sets": {}}
for name in ("ideal", "stress", "legal"):
    items = datasets.load_sim(name)
    truth = {id(it["on"]): it["truth"] for it in items}

    def measure(gray, c=None, off=None):
        t = truth[id(gray)]
        line = {"u": t["u"][::3], "v": t["v"][::3]}
        out = depth.measure_from_line(line["u"], line["v"], c)
        out["line"] = line
        return out
    res["sets"][name] = run.eval_sim(measure, items, cal)
    print(name, {k: round(v, 3) for k, v in res["sets"][name].items() if isinstance(v, float)})
with open("results/oracle_line.json", "w") as fh:
    json.dump(res, fh, indent=1)
