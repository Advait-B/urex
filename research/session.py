"""session.py - take photos and measure them as you go, then write one clean table.

    python research/session.py                  # start the camera logger and measure every new photo
    python research/session.py --name bus123    # same, with a label in the folder name
    python research/session.py --folder some/folder/of/jpgs     # no camera: measure photos you already have

Press the blue B1 button on the Nucleo for each photo. Press Ctrl+C when you are done.
Everything goes into  sessions/<date>_<time>[_name]/ :
    photos/        every photo, exactly as received
    results.csv    one row per photo (open in Excel)
    results.md     the same table, formatted

Each photo is measured with the recommended pipeline (algos/stack_dp_rd_gate.py):
line tracking -> groove depth in pixels -> quality gate (OK or RETAKE).
The mm column uses the DESIGN geometry (laser 35 deg, 90 mm stand-off). It is an ESTIMATE until the
shim calibration is done; the low-high range shows what a 70-110 mm stand-off would give.
"""
import argparse
import csv
import datetime
import glob
import os
import subprocess
import sys
import time
import warnings

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # urex-main/
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
warnings.filterwarnings("ignore")

import single_frame as sf                                      # noqa: E402  (design-geometry px -> mm)
from algos import stack_dp_rd_gate as stack                    # noqa: E402  (the recommended pipeline)
from bench.datasets import to_canonical                        # noqa: E402

LEGAL_MM = 1.6
MIN_PX = 2.0          # dips smaller than this are noise in an uncalibrated photo, not grooves
PC_DIR = os.path.join(ROOT, "urex-camera-photo", "urex-camera-photo", "pc")
COLS = ["#", "photo", "status", "grooves", "depth_px", "approx_mm", "mm_range", "verdict"]


def measure(path):
    """-> one table row (dict) for one photo."""
    row = dict.fromkeys(COLS, "")
    row["photo"] = os.path.basename(path)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        row["status"] = "UNREADABLE"
        return row
    try:
        out = stack.measure(to_canonical(gray), None, None)
    except Exception as e:                                     # one bad photo must not stop the session
        row["status"] = f"ERROR ({type(e).__name__})"
        return row
    grooves = sorted((g for g in out["grooves"] if g["depth"] >= MIN_PX), key=lambda g: g["u0"])
    if out.get("usable", 0) < 0.5 or not grooves:
        row["status"], row["grooves"] = "RETAKE", 0
        return row
    px = [g["depth"] for g in grooves]
    width = gray.shape[1]
    mm = [sf.fixed_scale_mm(p, width, 90.0) for p in px]
    lo = [sf.fixed_scale_mm(p, width, 70.0) for p in px]
    hi = [sf.fixed_scale_mm(p, width, 110.0) for p in px]
    row.update(status="OK", grooves=len(px),
               depth_px=" / ".join(f"{p:.1f}" for p in px),
               approx_mm=" / ".join(f"{m:.1f}" for m in mm),
               mm_range=" / ".join(f"{a:.1f}-{b:.1f}" for a, b in zip(lo, hi)),
               verdict="PASS" if min(mm) >= LEGAL_MM else "FAIL")
    row["_mm"] = mm
    return row


def find_port():
    """The Nucleo's ST-LINK serial port: COMx on Windows, /dev/tty.usbmodem* on macOS, /dev/ttyACM* on Linux."""
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    stlink = [p.device for p in ports if p.vid == 0x0483 or "STLink" in (p.description or "") or "STMicro" in (p.description or "")]
    other = [p.device for p in ports if "usbmodem" in p.device or "ttyACM" in p.device]
    return next(iter(stlink + other), None)


def write_tables(rows, folder):
    with open(os.path.join(folder, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    wid = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLS} if rows else {c: len(c) for c in COLS}
    line = lambda vals: "| " + " | ".join(str(v).ljust(wid[c]) for c, v in zip(COLS, vals)) + " |"
    table = [line(COLS), "|" + "|".join("-" * (wid[c] + 2) for c in COLS) + "|"] + [line([r[c] for c in COLS]) for r in rows]
    ok = [r for r in rows if r["status"] == "OK"]
    allmm = [m for r in ok for m in r["_mm"]]
    summary = [f"Photos: {len(rows)}   measured: {len(ok)}   retake: {len(rows) - len(ok)}"]
    if allmm:
        summary.append(f"Groove depth over all measured photos: median {np.median(allmm):.1f} mm, "
                       f"min {min(allmm):.1f} mm, max {max(allmm):.1f} mm (design-geometry estimate)")
    note = ("approx_mm = design geometry at 90 mm stand-off, NOT calibrated. mm_range = the same reading at 70-110 mm. "
            "Several values in one cell = several grooves, left to right.")
    text = "\n".join([f"# UREX session {os.path.basename(folder)}", "", *table, "", *summary, "", note, ""])
    with open(os.path.join(folder, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", help="measure the .jpg files already in this folder (no camera)")
    ap.add_argument("--name", default="", help="label added to the session folder name")
    ap.add_argument("--port", help="Nucleo serial port (default: found automatically)")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M") + (f"_{args.name}" if args.name else "")
    session = os.path.join(ROOT, "sessions", stamp)
    photos = os.path.abspath(args.folder) if args.folder else os.path.join(session, "photos")
    os.makedirs(photos, exist_ok=True)
    os.makedirs(session, exist_ok=True)

    cam = None
    if not args.folder:
        port = args.port or find_port()
        if port is None:
            sys.exit("No Nucleo found. Plug in the ST-LINK USB cable, or pass --port, or use --folder.")
        cam = subprocess.Popen([sys.executable, "get_photo.py", port, "--out", photos], cwd=PC_DIR)
        print(f"Saving photos to {photos}\nPress B1 on the Nucleo for each photo. Ctrl+C when you are done.\n")

    rows, seen = [], set()
    try:
        while True:
            new = [p for p in sorted(glob.glob(os.path.join(photos, "*.jpg"))) if p not in seen]
            for p in new:
                seen.add(p)
                if p.endswith("_A.jpg"):                       # pair mode: A = laser off, nothing to measure
                    continue
                time.sleep(0.3)                                # let the file finish writing
                row = measure(p)
                row["#"] = len(rows) + 1
                rows.append(row)
                shown = f"{row['approx_mm']} mm  ({row['depth_px']} px)  {row['verdict']}" if row["status"] == "OK" else ""
                print(f"  #{row['#']:<3} {row['photo']:<32} {row['status']:<8} {shown}")
                write_tables(rows, session)                    # keep the files up to date after every photo
            if args.folder or (cam and cam.poll() is not None):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if cam and cam.poll() is None:
            cam.terminate()
    print("\n" + write_tables(rows, session))
    print(f"Saved: {os.path.join(session, 'results.csv')}\n       {os.path.join(session, 'results.md')}")


if __name__ == "__main__":
    main()
