"""focus_check.py - score saved photos for sharpness and brightness.

    python focus_check.py ../urex-camera-photo/urex-camera-photo/pc/photos
    python focus_check.py photos --sheet sheet.jpg       (also writes a contact sheet)

For each .jpg it prints:
  sharp   variance of the Laplacian over the centre third (bigger = sharper).
          Same metric as focus_assist_openmv.py. Below ~30 is mush.
  mean    average grey level. Aim for 80-160: not clipped dark, not blown out.
  clip%   percentage of pixels at 255. Above 2 % the laser peak will saturate and
          the sub-pixel centroid becomes biased (a flat top has no centre).
Use it after a B1-press session to see which shots are usable without opening
each one, and to compare lens settings.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def score(gray: np.ndarray) -> dict:
    h, w = gray.shape
    roi = gray[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    lap = cv2.Laplacian(roi, cv2.CV_64F, ksize=3)
    return {
        "sharp": float(lap.var()),
        "mean": float(gray.mean()),
        "clip_pct": float((gray >= 255).mean() * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--sheet", help="write a labelled contact sheet to this file")
    ap.add_argument("--rotate180", action="store_true", help="our camera is mounted upside down")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.jpg")))
    if not files:
        raise SystemExit(f"no .jpg files in {args.folder}")

    print(f"{'file':34s} {'sharp':>8s} {'mean':>6s} {'clip%':>6s}  verdict")
    tiles = []
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if args.rotate180:
            g = cv2.rotate(g, cv2.ROTATE_180)
        s = score(g)
        verdict = []
        if s["sharp"] < 30:
            verdict.append("BLURRY")
        if s["clip_pct"] > 2:
            verdict.append("saturated")
        if s["mean"] < 60:
            verdict.append("dark")
        print(f"{os.path.basename(f):34s} {s['sharp']:8.0f} {s['mean']:6.0f} {s['clip_pct']:6.1f}  {' '.join(verdict) or 'ok'}")
        if args.sheet:
            t = cv2.cvtColor(cv2.resize(g, (320, 240)), cv2.COLOR_GRAY2BGR)
            cv2.putText(t, os.path.basename(f)[6:21], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(t, f"sharp={s['sharp']:.0f} mean={s['mean']:.0f}", (4, 232),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            tiles.append(t)

    if args.sheet:
        cols = 6
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(args.sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"\ncontact sheet -> {args.sheet}")


if __name__ == "__main__":
    main()
