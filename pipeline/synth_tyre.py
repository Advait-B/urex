"""synth_tyre.py - make fake laser-line photos of a tyre tread so we can test
laser_profile.py before the laser is wired up.

Geometry (matches the CMOS optics sheet):
  * The camera looks straight at the tread. The laser line runs LEFT-RIGHT
    across the image (across the tread width). Image columns = position across
    the tread, image rows = height of the line.
  * The laser comes in at an angle, so where the rubber is LOWER (a groove)
    the line appears SHIFTED by  shift_px = depth_mm / mm_per_px_depth.
  * Frame A = laser off (ambient light, rubber texture).
    Frame B = laser on  (same scene + the bright line + a bit more noise).

    python synth_tyre.py out/case1 --grooves 7,2.0  3,9.5  1.2,5.0 --blur 1.0
      -> out/case1_A.png, out/case1_B.png, out/case1_truth.json

Each --grooves item is  depth_mm,width_mm . A 1.2 mm deep groove is below the
1.6 mm legal limit, so that case should FAIL in laser_profile.py.
"""
import argparse
import json
import os

import cv2
import numpy as np


def make_pair(w=320, h=240, grooves=((7.0, 12.0), (6.0, 9.0), (2.5, 8.0)), mm_per_px_depth=0.249,
              mm_per_px_lateral=0.35, tilt_px=4.0, blur=1.0, line_peak=170, ambient=90,
              noise=6.0, sigma=1.5, speckle=0.15, seed=0, baseline_row=None):
    rng = np.random.default_rng(seed)
    if baseline_row is None:
        baseline_row = h * 0.5

    # ---- ground-truth depth per column (mm): 0 on the tread surface
    depth = np.zeros(w)
    x = np.arange(w)
    n = len(grooves)
    centres = np.linspace(w * 0.15, w * 0.85, n) if n else []
    truth = []
    for (d_mm, w_mm), cx in zip(grooves, centres):
        half = (w_mm / mm_per_px_lateral) / 2
        lo, hi = int(round(cx - half)), int(round(cx + half))
        depth[lo:hi] = d_mm
        truth.append({"depth_mm": d_mm, "width_mm": w_mm, "col_start": lo, "col_end": hi})

    # the laser line row for each column: a slightly tilted straight line on the
    # surface (hand-held = never perfectly square) + the groove shift
    y_line = baseline_row + tilt_px * (x - w / 2) / w + depth / mm_per_px_depth

    # ---- frame A: ambient light on rubber (blotchy texture + vignetting)
    yy, xx = np.mgrid[0:h, 0:w]
    texture = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6) * 25
    vignette = 1 - 0.35 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    A = ambient * vignette + texture
    # grooves look darker under ambient light
    A -= 25 * (depth[None, :] > 0) * np.exp(-((yy - y_line[None, :]) ** 2) / (2 * 6 ** 2))

    # ---- frame B: A + the laser line (Gaussian cross-section) with speckle
    r = yy - y_line[None, :]
    line = line_peak * np.exp(-(r ** 2) / (2 * sigma ** 2))
    speck = 1 + speckle * cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2)
    B = A + line * speck

    # optics blur (defocus) and sensor noise, then quantise like a real camera
    if blur > 0:
        A = cv2.GaussianBlur(A.astype(np.float32), (0, 0), blur)
        B = cv2.GaussianBlur(B.astype(np.float32), (0, 0), blur)
    A = A + rng.normal(0, noise, (h, w))
    B = B + rng.normal(0, noise * 1.2, (h, w))
    A = np.clip(A, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    return A, B, {"y_line": y_line.tolist(), "depth_mm": depth.tolist(), "grooves": truth,
                  "mm_per_px_depth": mm_per_px_depth, "mm_per_px_lateral": mm_per_px_lateral,
                  "tilt_px": tilt_px, "blur": blur}


def parse_grooves(items):
    out = []
    for it in items:
        d, w = it.split(",")
        out.append((float(d), float(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="output prefix, e.g. out/case1")
    ap.add_argument("--grooves", nargs="*", default=["7,12", "6,9", "2.5,8"], help="depth_mm,width_mm ...")
    ap.add_argument("--blur", type=float, default=1.0, help="defocus blur sigma in px (0 = perfect lens)")
    ap.add_argument("--noise", type=float, default=6.0)
    ap.add_argument("--peak", type=float, default=170, help="laser brightness above ambient (255 clips)")
    ap.add_argument("--tilt", type=float, default=4.0, help="line tilt across the frame, px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upright", action="store_true",
                    help="write frames the right way up. Default writes them upside down, the way our "
                         "inverted camera sees the scene, so calib.json's rotate180=true undoes it.")
    args = ap.parse_args()

    A, B, truth = make_pair(grooves=parse_grooves(args.grooves), blur=args.blur, noise=args.noise,
                            line_peak=args.peak, tilt_px=args.tilt, seed=args.seed)
    if not args.upright:
        A, B = cv2.rotate(A, cv2.ROTATE_180), cv2.rotate(B, cv2.ROTATE_180)
        truth["note"] = "frames stored rotated 180 deg (camera mounting); truth arrays are in upright coordinates"
    os.makedirs(os.path.dirname(args.prefix) or ".", exist_ok=True)
    cv2.imwrite(args.prefix + "_A.png", A)
    cv2.imwrite(args.prefix + "_B.png", B)
    with open(args.prefix + "_truth.json", "w") as fh:
        json.dump(truth, fh)
    print(f"wrote {args.prefix}_A.png, _B.png, _truth.json  grooves={truth['grooves']}")


if __name__ == "__main__":
    main()
