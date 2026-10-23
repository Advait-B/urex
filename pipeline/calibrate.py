"""calibrate.py - measure mm_per_px_depth with the shim-and-flat-plate test.

The design sheet PREDICTS 0.249 mm per pixel of line shift from the optics
formula. Real lenses, mounting angles and the hand-soldered laser bracket all
differ, so we MEASURE it:

  1. Put the device on a flat plate (the "tyre surface"). Take a laser-off and
     a laser-on photo.  This is shim 0.0.
  2. Put a shim of known thickness (e.g. 2.0 mm) under the device's feet so
     the plate is now 2 mm further away = a 2 mm "groove" everywhere.
     Photo pair again. Repeat for 4, 6, 8 mm (feeler gauges, stacked PCBs,
     or 3D-printed blocks measured with calipers).
  3. Name the files like   shim_0.0_A.jpg shim_0.0_B.jpg shim_2.0_A.jpg ...
     and run:

        python calibrate.py photos/calib/ --write calib.json

It finds the laser line in each pair (same code as laser_profile.py), takes the
median line row, fits  row = k * shim_mm + row0  and writes
mm_per_px_depth = 1/k plus the right depth_sign into calib.json. It also
prints the residual of each shim so you can see if the relation is linear
(it should be, to well under 0.1 mm, over the 0-10 mm range we care about).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import laser_profile as lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with shim_<mm>_A.jpg / shim_<mm>_B.jpg pairs")
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(__file__), "calib.json"))
    ap.add_argument("--write", help="calib.json to update with the measured values")
    args = ap.parse_args()

    calib = dict(lp.DEFAULT_CALIB)
    if os.path.exists(args.calib):
        with open(args.calib) as fh:
            calib.update({k: v for k, v in json.load(fh).items() if not k.startswith("_")})

    shims, rows = [], []
    for b in sorted(glob.glob(os.path.join(args.folder, "shim_*_B.*"))):
        m = re.search(r"shim_([0-9.]+)_B", os.path.basename(b))
        a = b.replace("_B.", "_A.")
        on = lp.load_gray(b, calib["rotate180"])
        off = lp.load_gray(a, calib["rotate180"]) if os.path.exists(a) else None
        diff = lp.smooth(lp.subtract(on, off))
        y, contrast = lp.find_line(diff, min_contrast=calib["min_contrast"])
        row = float(np.nanmedian(y))
        shims.append(float(m.group(1)))
        rows.append(row)
        print(f"shim {shims[-1]:5.2f} mm -> line row {row:8.3f} px  (valid {np.mean(~np.isnan(y))*100:.0f} %, contrast {np.nanmean(contrast):.0f})")

    if len(shims) < 2:
        raise SystemExit("need at least two shim thicknesses")
    k, row0 = np.polyfit(shims, rows, 1)              # row = k * mm + row0
    resid = np.array(rows) - (k * np.array(shims) + row0)
    mm_per_px = 1.0 / abs(k)
    sign = 1 if k > 0 else -1
    print(f"\nfit: row = {k:.4f} px/mm * shim + {row0:.2f}")
    print(f"mm_per_px_depth = {mm_per_px:.4f}   depth_sign = {sign}")
    print(f"linearity residual (px): " + " ".join(f"{r:+.2f}" for r in resid) +
          f"   -> worst {np.abs(resid).max()*mm_per_px:.3f} mm")

    if args.write:
        out = {}
        if os.path.exists(args.write):
            with open(args.write) as fh:
                out = json.load(fh)
        out.update({"mm_per_px_depth": round(mm_per_px, 4), "depth_sign": sign,
                    "_calibrated_from": [f"shim {s} mm -> row {r:.2f}" for s, r in zip(shims, rows)]})
        with open(args.write, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
