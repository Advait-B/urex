"""make_tables.py - turn results2/*.json (scorer v2) into the LaTeX tables of the paper.

    python paper/make_tables.py            (run from research/)  -> paper/tables/*.tex

Every number in the paper's comparison tables comes from here, so nothing is typed by hand.
"""
import json
import os

R2 = "results2"
OUT = "paper/tables"

# (module file, label in the paper, group)
ROWS = [
    ("baseline_steger", "Baseline (Steger ridge)", "Reference"),
    ("pk_cog", "Centre of gravity", "Per-column peak"),
    ("pk_fir2d_rows", "2-D Gaussian + parabola", "Per-column peak"),
    ("dp_dense", "Viterbi line tracking", "Tracking"),
    ("cnn_line", "U-Net, augmented", "Learned line"),
    ("cnn_line_noaug", "U-Net, no augmentation", "Learned line"),
    ("cand_clf", "Candidate classifier (GBM)", "Learned line"),
    ("rd_final", "Baseline line + robust depth", "Depth stage"),
    ("gated_baseline", "Baseline + quality gate", "Gate"),
    ("onoff_cog", "Laser on/off subtraction", "Two frames"),
    ("e2e_cnn", "End-to-end CNN", "No geometry"),
    ("stack_best", "Recommended stack", "Combined"),
]


def load(name):
    p = os.path.join(R2, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, nd=3):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def best_marks(vals, lower=True):
    nums = [v for v in vals if isinstance(v, (int, float))]
    if not nums:
        return [False] * len(vals)
    b = min(nums) if lower else max(nums)
    return [isinstance(v, (int, float)) and abs(v - b) < 5e-4 for v in vals]


def table(fname, set_name, cols, rows=ROWS, skip_best=("baseline_steger",)):
    """cols: list of (key, header, digits, lower_is_better or None)."""
    data = []
    for mod, label, group in rows:
        r = load(mod)
        if r is None or set_name not in r["sets"]:
            continue
        data.append((label, group, r["sets"][set_name]))
    lines = ["\\begin{tabular}{@{}ll" + "r" * len(cols) + "@{}}", "\\toprule",
             "Group & Algorithm & " + " & ".join(h for _, h, _, _ in cols) + " \\\\", "\\midrule"]
    marks = {}
    for key, _, _, lower in cols:
        vals = [d[2].get(key) for d in data]
        marks[key] = best_marks(vals, lower) if lower is not None else [False] * len(vals)
    last = None
    for i, (label, group, s) in enumerate(data):
        if last is not None and group != last:
            lines.append("\\addlinespace[2pt]")
        cells = []
        for key, _, nd, _ in cols:
            t = f(s.get(key), nd)
            cells.append("\\textbf{" + t + "}" if marks[key][i] else t)
        lines.append(f"{group if group != last else ''} & {label} & " + " & ".join(cells) + " \\\\")
        last = group
    lines += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")
    print(fname, len(data), "rows")


MM = [("coverage", "Cov.", 2, None), ("depth_mae_mm", "MAE", 3, True), ("depth_med_mm", "Median", 3, True),
      ("depth_p95_mm", "P95", 3, True), ("gross_rate", "Gross", 3, True), ("miss_rate", "Miss", 3, True),
      ("false_per_frame", "False", 3, True), ("verdict_acc", "Verdict", 3, False),
      ("false_pass_rate", "F.pass", 3, True), ("cpu_ms", "CPU ms", 1, None)]
PX = [("coverage", "Cov.", 2, None), ("line_f1", "Line F1", 3, False), ("depth_mae_px", "MAE px", 2, True),
      ("depth_med_px", "Median px", 2, True), ("gross_rate", "Gross", 3, True), ("miss_rate", "Miss", 3, True),
      ("false_per_frame", "False", 3, True)]
REAL = [("line_f1_truck", "Line F1", 3, False), ("depth_mae_px", "$\\Delta$ px", 2, None), ("found_ref", "Found", 0, None),
        ("ok_ref", "Within 3 px", 0, None), ("usable_recall", "Usable kept", 3, False),
        ("false_measure_nolaser", "F.meas (no laser)", 3, True), ("false_measure_laser", "F.meas (laser)", 3, True)]

if __name__ == "__main__":
    for s in ("ideal", "stress", "legal", "shifted"):
        table(f"tab_{s}.tex", s, MM)
    table("tab_handheld.tex", "handheld", PX)
    table("tab_real.tex", "real", REAL)
