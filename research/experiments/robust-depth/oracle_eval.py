"""oracle_eval.py - like bench/oracle.py: feed the TRUE line (every 3rd point) into each rd_* variant's depth
stage and write results/oracle_<variant>.json in the bench.run format.   python experiments/robust-depth/oracle_eval.py"""
import importlib, json, sys
sys.path.insert(0, "experiments/robust-depth")
import common as C
from algos import robust_depth as rd

for mod in sys.argv[1:] or ["rd_final", "rd_strict", "rd_band_corr", "rd_dbscan"]:
    m = importlib.import_module(f"algos.{mod}")

    def stage(u, v, c, m=m):
        out = rd.stage(u, v, c, **m.CONFIG)
        if m.DROP_LOWER_BOUNDS:
            out["grooves"] = [g for g in out["grooves"] if not g["lower_bound"]]
        return out
    res = {"name": f"oracle_{mod}", "module": "experiments/robust-depth/oracle_eval.py", "uses_off": False, "limit": None, "sets": {}}
    for name in ("ideal", "stress", "legal"):
        r = C.score(stage, name, "oracle")
        r.pop("_outs")
        res["sets"][name] = r
        print(mod, name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    with open(f"results/oracle_{mod}.json", "w") as fh:
        json.dump(res, fh, indent=1)
