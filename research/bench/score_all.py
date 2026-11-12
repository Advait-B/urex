"""score_all.py - re-score a list of algorithm modules with scorer v2 (bench/run2.py).

    python -m bench.score_all algos.a algos.b ...        -> results2/<NAME>.json each
Skips a module whose JSON already exists (delete it to re-score). Algorithms that carry their
own calibration (calib_*) cannot use the per-unit calibration of `shifted`: pass --no-shifted.
"""
import argparse
import json
import os
import sys
import traceback

from . import run2

ap = argparse.ArgumentParser()
ap.add_argument("modules", nargs="+")
ap.add_argument("--no-shifted", action="store_true")
ap.add_argument("--out", default="results2")
args = ap.parse_args()
sys.path.insert(0, os.getcwd())
os.makedirs(args.out, exist_ok=True)
sets = ["ideal", "stress", "legal", "shifted", "handheld", "real"]
if args.no_shifted:
    sets.remove("shifted")
for m in args.modules:
    path = os.path.join(args.out, m.split(".")[-1] + ".json")
    if os.path.exists(path):
        print("skip", m)
        continue
    try:
        res = run2.score(m, sets)
        run2.show(res)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
    except Exception:
        print("FAILED", m)
        traceback.print_exc()
