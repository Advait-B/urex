"""dev_train.py - compare all variants on a slice of `train` (the only set used for choices)."""
import sys
from variants import make, KEYS, row
from bench import datasets, calib as calib_mod
from bench.run import eval_sim

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
items = datasets.load_sim("train")[:n]
cal = calib_mod.default_calib()
print(f"{'variant':16s} " + " ".join(f"{k[:8]:>8s}" for k in KEYS))
for mode in ["single", "plain", "gain", "reg", "cons", "robust"]:
    for det in ["steger", "cog"]:
        print(f"{mode + '+' + det:16s} " + row(eval_sim(make(mode, det), items, cal)), flush=True)
