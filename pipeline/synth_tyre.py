"""synth_tyre.py - make fake laser-line photos of a tyre tread so we can test
laser_profile.py before the laser is wired up.

Geometry (matches the CMOS optics sheet):
  * The camera looks straight at the tread. The laser line runs LEFT-RIGHT
    across the image (across the tread width). Image columns = position across
    the tread, image rows = height of the line.
  * The laser sheet comes in at angle theta to the camera axis. If the surface is
    z mm further away than the reference plane (z = 0 = a flat plate touching
    the device's feet), the line lands z*tan(theta) mm further along, and the
    camera sees that shifted by

        shift_px(z) = z * tan(theta) * f / ((L + z - f) * pitch)

    Note the z in the bottom line: further away = smaller magnification, so
    each extra mm of depth gives FEWER pixels. That makes the relation curved
    (about 4.0 px/mm near z = 0, only 3.2 px/mm near z = 20 mm). Last week's
    version of this file used a straight line. That hid a 2.5 mm error on a
    16 mm groove. See calibrate.py for how the real device is calibrated.
  * Frame A = laser off (ambient light, rubber texture).
    Frame B = laser on  (same scene + the bright line + a bit more noise).

    python synth_tyre.py out/case1 --grooves 7,12 3,9.5 1.2,5 --standoff 2
      -> out/case1_A.png, out/case1_B.png, out/case1_truth.json
    python synth_tyre.py out/shims --shims 0,2,4,6,8,10,12,14,16,18,20
      -> out/shims/shim_0.0_A.png ... (a fake calibration run for calibrate.py)

Each --grooves item is  depth_mm,width_mm . A 1.2 mm deep groove is below the
1.6 mm legal limit, so that case should FAIL in laser_profile.py.
"""
import argparse
import json
import os

import cv2
import numpy as np

# Device optics (13 Sep optics sheet). pitch 5.6 um = OV5640 binned to 640x480.
OPTICS = dict(theta_deg=35.0, f_mm=2.8, L_mm=90.0, pitch_mm=0.0056)


def shift_px(z_mm, theta_deg=35.0, f_mm=2.8, L_mm=90.0, pitch_mm=0.0056):
    """How far (px) the laser line moves when the surface is z_mm beyond the reference plane."""
    z = np.asarray(z_mm, dtype=float)
    return z * np.tan(np.radians(theta_deg)) * f_mm / ((L_mm + z - f_mm) * pitch_mm)


def make_pair(w=320, h=240, grooves=((7.0, 12.0), (6.0, 9.0), (2.5, 8.0)), standoff_mm=0.0,
              surface_tilt_mm=0.0, mm_per_px_lateral=0.35, tilt_px=2.0, bow_px=3.0, blur=1.0,
              line_peak=170, ambient=90, noise=6.0, sigma=1.5, speckle=0.15, seed=0,
              baseline_row=None, optics=None, wall_mm=0.0, render="column"):
    """standoff_mm      tread surface distance beyond the reference plane (rubber feet sinking
                        in, tread crown curvature...). 0 = touching the feet.
    render              "column": the line's brightness is a Gaussian in each column (fast;
                        what the older tests use). "distance": brightness depends on the true
                        distance to the line's centre curve, so steep walls stay as bright as
                        the rest, like a real laser line (use with wall_mm).
    wall_mm             groove walls slope over this width (mm) on each side. 0 = vertical walls
                        (the line jumps). Real grooves slope, so the line runs down the wall.
    surface_tilt_mm     device not square to the tyre: surface height changes by this much
                        from the left edge of the frame to the right edge.
    tilt_px, bow_px     the line on the reference plane is not perfectly straight and level
                        in the image (laser rotated slightly vs the sensor, lens distortion).
                        These belong to the DEVICE, so keep them the same for calibration
                        shots and measurement shots.
    """
    optics = {**OPTICS, **(optics or {})}
    rng = np.random.default_rng(seed)
    if baseline_row is None:
        baseline_row = h * 0.3          # leaves room for the line to move down ~20 mm

    # ---- ground-truth depth per column (mm): 0 on the tread surface
    depth = np.zeros(w)
    x = np.arange(w)
    n = len(grooves)
    centres = np.linspace(w * 0.15, w * 0.85, n) if n else []
    truth = []
    for (d_mm, w_mm), cx in zip(grooves, centres):
        half = (w_mm / mm_per_px_lateral) / 2
        lo, hi = int(round(cx - half)), int(round(cx + half))
        if wall_mm > 0:
            ramp = wall_mm / mm_per_px_lateral
            prof = np.clip(np.minimum(x - (cx - half), (cx + half) - x) / ramp, 0, 1) * d_mm
            depth = np.maximum(depth, prof)
        else:
            depth[lo:hi] = d_mm
        truth.append({"depth_mm": d_mm, "width_mm": w_mm, "col_start": lo, "col_end": hi})

    # absolute distance of each column's surface beyond the reference plane
    u = (x - w / 2) / (w / 2)                                    # -1 .. +1 across the frame
    z = standoff_mm + surface_tilt_mm * u / 2 + depth
    y_ref = baseline_row + tilt_px * u / 2 + bow_px * u ** 2      # line on the reference plane
    y_line = y_ref + shift_px(z, **optics)

    # ---- frame A: ambient light on rubber (blotchy texture + vignetting)
    yy, xx = np.mgrid[0:h, 0:w]
    texture = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6) * 25
    vignette = 1 - 0.35 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    A = ambient * vignette + texture
    # grooves look darker under ambient light
    A -= 25 * (depth[None, :] > 0) * np.exp(-((yy - y_line[None, :]) ** 2) / (2 * 6 ** 2))

    # ---- frame B: A + the laser line (Gaussian cross-section) with speckle
    if render == "distance":
        S = 4                                              # draw the centre curve 4x finer
        canvas = np.full((h * S, w * S), 255, np.uint8)
        pts = np.round(np.c_[(x + 0.5) * S - 0.5, (y_line + 0.5) * S - 0.5]).astype(np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 0, 1)
        dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5) / S
        dist = cv2.resize(dist, (w, h), interpolation=cv2.INTER_AREA)
        line = line_peak * np.exp(-(dist ** 2) / (2 * sigma ** 2))
    else:
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

    return A, B, {"y_line": y_line.tolist(), "z_mm": z.tolist(), "depth_mm": depth.tolist(),
                  "grooves": truth, "standoff_mm": standoff_mm, "surface_tilt_mm": surface_tilt_mm,
                  "mm_per_px_lateral": mm_per_px_lateral, "optics": optics, "blur": blur}


def write_shim_series(folder, shims, rotate180=True, ext="png", **kw):
    """Fake calibration run: flat plate (no grooves) at each shim height."""
    os.makedirs(folder, exist_ok=True)
    for i, s in enumerate(shims):
        A, B, _ = make_pair(grooves=(), standoff_mm=s, seed=100 + i, **kw)
        if rotate180:
            A, B = cv2.rotate(A, cv2.ROTATE_180), cv2.rotate(B, cv2.ROTATE_180)
        cv2.imwrite(os.path.join(folder, f"shim_{s:.1f}_A.{ext}"), A)
        cv2.imwrite(os.path.join(folder, f"shim_{s:.1f}_B.{ext}"), B)


def parse_grooves(items):
    out = []
    for it in items:
        d, w = it.split(",")
        out.append((float(d), float(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="output prefix, e.g. out/case1 (or a folder with --shims)")
    ap.add_argument("--grooves", nargs="*", default=["7,12", "6,9", "2.5,8"], help="depth_mm,width_mm ...")
    ap.add_argument("--shims", help="comma list of plate heights in mm: write a fake calibration run instead")
    ap.add_argument("--standoff", type=float, default=0.0, help="tread surface beyond the reference plane, mm")
    ap.add_argument("--surface-tilt", type=float, default=0.0, help="surface height change across the frame, mm")
    ap.add_argument("--blur", type=float, default=1.0, help="defocus blur sigma in px (0 = perfect lens)")
    ap.add_argument("--noise", type=float, default=6.0)
    ap.add_argument("--peak", type=float, default=170, help="laser brightness above ambient (255 clips)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upright", action="store_true",
                    help="write frames the right way up. Default writes them upside down, the way our "
                         "inverted camera sees the scene, so calib.json's rotate180=true undoes it.")
    args = ap.parse_args()

    if args.shims:
        shims = [float(s) for s in args.shims.split(",")]
        write_shim_series(args.prefix, shims, rotate180=not args.upright, blur=args.blur, noise=args.noise,
                          line_peak=args.peak)
        print(f"wrote {len(shims)} shim pairs into {args.prefix}/")
        return

    A, B, truth = make_pair(grooves=parse_grooves(args.grooves), standoff_mm=args.standoff,
                            surface_tilt_mm=args.surface_tilt, blur=args.blur, noise=args.noise,
                            line_peak=args.peak, seed=args.seed)
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
