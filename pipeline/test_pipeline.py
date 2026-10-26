"""test_pipeline.py - does laser_profile.py get the right answer on fake data?

    python test_pipeline.py                (prints a table, exits non-zero if any case fails)
    python test_pipeline.py --plots out/   (also saves a diagnostic PNG per case)

What it does, like a real bench session but with fake photos:
  1. "calibrate": make shim photos at 0..20 mm with synth_tyre.py and run
     calibrate.py's fit on them (same device bow/tilt as the test shots).
  2. "measure": make tread photos with known grooves, run the pipeline twice,
     once with the calibration and once with the old straight-line
     mm_per_px_depth, and compare both to the truth.

err = worst absolute groove-depth error. Target 0.1 mm. A case passes if the
CALIBRATED run finds every groove, errs < 0.15 mm and gives the right verdict.
The straight-line column is there to show why calibration matters.
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

import calibrate as cal
import laser_profile as lp
import synth_tyre as st

SHIMS = [0, 2, 4, 6, 8, 10, 12, 15, 18, 20]

CASES = [
    # name,              grooves (depth,width) mm,       blur noise peak standoff surf_tilt expected
    # (expected None = decide from the groove depths: FAIL if any < 1.6 mm else PASS)
    ("worn_typical",     [(7, 12), (6, 9), (2.5, 8)],    1.0, 6.0, 170, 0.0, 0.0, None),
    ("new_bus_tyre_16",  [(16, 12), (15, 10), (16, 9)],  1.0, 6.0, 170, 0.0, 0.0, None),
    ("standoff_3mm",     [(16, 12), (8, 10), (3, 9)],    1.0, 6.0, 170, 3.0, 0.0, None),
    ("held_crooked_4mm", [(12, 12), (12, 10), (12, 9)],  1.0, 6.0, 170, 1.0, 4.0, None),
    ("below_legal_1.2",  [(7, 12), (1.2, 8), (3, 9)],    1.0, 6.0, 170, 0.0, 0.0, None),
    ("barely_there_0.8", [(7, 12), (0.8, 9), (6, 9)],    1.0, 6.0, 170, 0.0, 0.0, None),
    ("borderline_1.7",   [(1.7, 9), (1.7, 6), (8, 12)],  1.0, 6.0, 170, 0.0, 0.0, None),
    ("sharp_clean",      [(7, 12), (6, 9), (2.5, 8)],    0.0, 3.0, 170, 0.0, 0.0, None),
    ("defocused_2px",    [(7, 12), (6, 9), (2.5, 8)],    2.0, 6.0, 170, 0.0, 0.0, None),
    ("noisy_15",         [(7, 12), (6, 9), (2.5, 8)],    1.0, 15., 170, 0.0, 0.0, None),
    ("saturated_300",    [(7, 12), (6, 9), (2.5, 8)],    1.0, 6.0, 300, 0.0, 0.0, None),
    ("narrow_4.5mm",     [(5, 4.5), (5, 12)],            1.0, 6.0, 170, 0.0, 0.0, None),
    # badly out of focus or a dim laser: contrast collapses, the quality gate must say RETAKE
    ("defocused_4px",    [(7, 12), (6, 9), (2.5, 8)],    4.0, 6.0, 170, 0.0, 0.0, "RETAKE"),
    ("dim_laser_60",     [(7, 12), (6, 9), (2.5, 8)],    1.0, 6.0,  60, 0.0, 0.0, "RETAKE"),
]


def make_calibration(tmp):
    folder = os.path.join(tmp, "shims")
    st.write_shim_series(folder, SHIMS, ext="jpg")
    calib = dict(lp.DEFAULT_CALIB)
    shots = cal.load_shims(folder, calib)
    return {**calib, "height_poly": cal.fit_height_poly([(mm, y, shape) for mm, y, _, shape in shots])}


def score(result, truth):
    """match each true groove to the detected groove that overlaps it most"""
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
    return (max(errs) if errs else float("nan")), missed, extra


def run_case(case, calib_poly, tmp, plots=None, seed=1):
    name, grooves, blur, noise, peak, standoff, tilt, expected = case
    A, B, truth = st.make_pair(grooves=grooves, blur=blur, noise=noise, line_peak=peak,
                               standoff_mm=standoff, surface_tilt_mm=tilt, seed=seed)
    A, B = cv2.rotate(A, cv2.ROTATE_180), cv2.rotate(B, cv2.ROTATE_180)   # as the inverted camera sees it
    # save as JPEG quality 95, exactly what the camera sends (main.c JPEG_QUALITY)
    pa, pb = os.path.join(tmp, f"{name}_A.jpg"), os.path.join(tmp, f"{name}_B.jpg")
    cv2.imwrite(pa, A, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(pb, B, [cv2.IMWRITE_JPEG_QUALITY, 95])

    res, inter = lp.analyse(pb, pa, calib=calib_poly)
    res_lin, _ = lp.analyse(pb, pa, calib=None)             # old straight-line conversion
    if plots:
        os.makedirs(plots, exist_ok=True)
        lp.plot(res, inter, os.path.join(plots, f"{name}.png"))

    err, missed, extra = score(res, truth)
    err_lin, _, _ = score(res_lin, truth)
    want = expected or ("FAIL" if min(g[0] for g in grooves) < 1.6 else "PASS")
    if want == "RETAKE":
        ok = res["verdict"] == "RETAKE"                      # only the gate matters here
    else:
        ok = missed == 0 and extra == 0 and err < 0.15 and res["verdict"] == want
    return dict(name=name, err=err, err_lin=err_lin, missed=missed, extra=extra,
                valid=res["valid_columns_frac"], contrast=res["mean_contrast"],
                standoff=res["shoulder_line"]["standoff_mm"], true_standoff=standoff,
                verdict=res["verdict"], want=want, ok=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", help="folder for diagnostic PNGs")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        calib_poly = make_calibration(tmp)
        print(f"calibrated on synthetic shims {SHIMS} mm\n")
        print(f"{'case':17s} {'err mm':>7s} {'(straight-line)':>15s} {'miss':>4s} {'extra':>5s} {'valid':>5s} "
              f"{'contr':>5s} {'standoff':>12s}  {'verdict':7s} {'expect':6s} ok")
        all_ok = True
        for c in CASES:
            r = run_case(c, calib_poly, tmp, plots=args.plots)
            all_ok &= r["ok"]
            so = "-" if r["standoff"] is None else f"{r['standoff']:.2f}/{r['true_standoff']:.1f}"
            print(f"{r['name']:17s} {r['err']:7.3f} {r['err_lin']:15.3f} {r['missed']:4d} {r['extra']:5d} "
                  f"{r['valid']*100:4.0f}% {r['contrast']:5.0f} {so:>12s}  {r['verdict']:7s} {r['want']:6s} "
                  f"{'yes' if r['ok'] else 'NO'}")
    print("\nALL OK" if all_ok else "\nSOME CASES FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
