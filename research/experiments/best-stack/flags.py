"""flags.py - what the lower_bound flag is worth on the test sets (scoring only; nothing is fitted here).
    python experiments/best-stack/flags.py     (after build.py ideal stress legal shifted)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import fit_gate as fg  # noqa: E402

for s in ("ideal", "stress", "legal", "shifted"):
    rows = fg.load(s)
    fg.flag_table(rows, s)
    for r in rows:
        if r["pred_pass"] and r["has_view"] and not r["true_pass"]:
            print(f"    unsafe PASS, {s} frame {r['i']}: (true, pred, lower_bound, floor_visible_px) =",
                  [(round(g["true"], 2), g["pred"] and round(g["pred"], 2), g["lower_bound"], round(g["floor_visible_px"])) for g in r["grooves"]],
                  "phantoms", [round(x, 2) for x in r["phantom"]])
