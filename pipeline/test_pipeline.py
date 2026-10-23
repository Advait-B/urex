"""test_pipeline.py - does laser_profile.py get the right answer on fake data?

    python test_pipeline.py            (prints a table, exits non-zero if any case fails)
    python test_pipeline.py --plots out/   (also saves a diagnostic PNG per case)

Each case makes a synthetic A/B pair with known groove depths, runs the
pipeline with the default calib values, and compares. "err" is the worst
absolute depth error over all grooves. The target from the design sheet is
0.1 mm; the legal decision needs to be right at 1.6 mm.
"""
import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np

import laser_profile as lp
import synth_tyre as st

CASES = [
    # name,              grooves (depth,width) mm,         blur, noise, peak, tilt, expected verdict
    # (expected None = decide from the groove depths: FAIL if any < 1.6 mm else PASS)
    ("sharp_clean",      [(7, 12), (6, 9), (2.5, 8)],        0.0,  3.0, 170, 0.0,  None),
    ("typical",          [(7, 12), (6, 9), (2.5, 8)],        1.0,  6.0, 170, 4.0,  None),
    ("below_legal_1.2",  [(7, 12), (1.2, 8), (3, 9)],        1.0,  6.0, 170, 4.0,  None),
    ("borderline_1.7",   [(1.7, 9), (1.7, 6), (8, 12)],      1.0,  6.0, 170, 4.0,  None),
    ("defocused_2px",    [(7, 12), (6, 9), (2.5, 8)],        2.0,  6.0, 170, 4.0,  None),
    # badly out of focus or a dim laser: contrast collapses, the quality gate must say RETAKE
    ("defocused_4px",    [(7, 12), (6, 9), (2.5, 8)],        4.0,  6.0, 170, 4.0,  "RETAKE"),
    ("dim_laser_60",     [(7, 12), (6, 9), (2.5, 8)],        1.0,  6.0,  60, 4.0,  "RETAKE"),
    ("saturated_300",    [(7, 12), (6, 9), (2.5, 8)],        1.0,  6.0, 300, 4.0,  None),
    ("noisy_15",         [(7, 12), (6, 9), (2.5, 8)],        1.0, 15.0, 170, 4.0,  None),
    ("tilted_20px",      [(7, 12), (6, 9), (2.5, 8)],        1.0,  6.0, 170, 20.0, None),
    ("narrow_4mm",       [(5, 4.5), (5, 12)],                1.0,  6.0, 170, 4.0,  None),
]


def run_case(name, grooves, blur, noise, peak, tilt, expected=None, plots=None, seed=1):
    A, B, truth = st.make_pair(grooves=grooves, blur=blur, noise=noise, line_peak=peak, tilt_px=tilt, seed=seed)
    A, B = cv2.rotate(A, cv2.ROTATE_180), cv2.rotate(B, cv2.ROTATE_180)   # as the inverted camera sees it
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "A.png"), os.path.join(d, "B.png")
        cv2.imwrite(pa, A)
        cv2.imwrite(pb, B)
        result, inter = lp.analyse(pb, pa, calib=None)
        if plots:
            os.makedirs(plots, exist_ok=True)
            lp.plot(result, inter, os.path.join(plots, f"{name}.png"))

    # match each true groove to the detected groove that overlaps it most
    errs, missed = [], 0
    for tg in truth["grooves"]:
        best, best_ov = None, 0
        for dg in result["grooves"]:
            ov = min(tg["col_end"], dg["col_end"]) - max(tg["col_start"], dg["col_start"])
            if ov > best_ov:
                best, best_ov = dg, ov
        if best is None:
            missed += 1
        else:
            errs.append(abs(best["depth_mm"] - tg["depth_mm"]))
    extra = len(result["grooves"]) - (len(truth["grooves"]) - missed)
    worst = max(errs) if errs else float("nan")
    true_verdict = expected or ("FAIL" if min(g[0] for g in grooves) < 1.6 else "PASS")
    if true_verdict == "RETAKE":
        ok = result["verdict"] == "RETAKE"          # only the gate matters here
    else:
        ok = missed == 0 and extra == 0 and (worst < 0.3) and result["verdict"] == true_verdict
    return {"name": name, "err_mm": worst, "missed": missed, "extra": extra, "valid": result["valid_columns_frac"],
            "contrast": result["mean_contrast"], "verdict": result["verdict"], "expected": true_verdict,
            "warn": "; ".join(result["warnings"]), "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", help="folder for diagnostic PNGs")
    args = ap.parse_args()

    print(f"{'case':18s} {'err mm':>7s} {'miss':>4s} {'extra':>5s} {'valid':>6s} {'contr':>6s}  {'verdict':8s} {'expect':6s} ok")
    all_ok = True
    for c in CASES:
        r = run_case(*c, plots=args.plots)
        all_ok &= r["ok"]
        print(f"{r['name']:18s} {r['err_mm']:7.3f} {r['missed']:4d} {r['extra']:5d} {r['valid']*100:5.0f}% {r['contrast']:6.0f}  "
              f"{r['verdict']:8s} {r['expected']:6s} {'yes' if r['ok'] else 'NO'}  {r['warn']}")
    print("\nALL OK" if all_ok else "\nSOME CASES FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
