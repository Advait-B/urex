"""get_photo.py - save the photos the NUCLEO-C542RC sends over its USB cable.

    pip install pyserial
    python get_photo.py COM5              (Windows)
    python get_photo.py /dev/tty.usbmodem* (macOS)   python get_photo.py /dev/ttyACM0 (Linux)

Leave it running, press the blue B1 button on the Nucleo, and each photo is
written to ./photos/photo_YYYYmmdd_HHMMSS.jpg. Close the serial terminal in
STM32CubeIDE first: only one program can open the port at a time.

In pair mode (PAIR_MODE 1 in main.c) each press gives two files,
pair_YYYYmmdd_HHMMSS_A.jpg (laser off) and ..._B.jpg (laser on).

    python get_photo.py PORT --analyse     also runs pipeline/laser_profile.py on
                                           each pair and prints the groove depths
    python get_photo.py PORT --out photos/calib --shim 2.0
                                           calibration run: names the next pair
                                           shim_2.0_A/B.jpg (then change --shim)
"""
import argparse
import datetime
import json
import os
import sys
import zlib

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", help="ST-LINK virtual COM port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="photos", help="folder for the .jpg files")
    ap.add_argument("--analyse", action="store_true", help="run the laser pipeline on each A/B pair")
    ap.add_argument("--shim", type=float, help="calibration: name pairs shim_<mm>_A/B (asks for the next value)")
    args = ap.parse_args()

    analyse = None
    if args.analyse:
        # pipeline/ lives at urex-main/pipeline, three folders up from pc/
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "pipeline")))
        try:
            import laser_profile
            calib_path = os.path.join(os.path.dirname(laser_profile.__file__), "calib.json")
            with open(calib_path) as fh:
                calib = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
            analyse = lambda a, b: laser_profile.analyse(b, a, calib)[0]
        except ImportError as e:
            print(f"--analyse needs numpy + opencv-python in this Python ({e}); continuing without it")
    tag, stamp, last_a = None, None, None

    os.makedirs(args.out, exist_ok=True)
    port = serial.Serial(args.port, args.baud, timeout=10)
    print(f"listening on {args.port} - press B1 on the Nucleo (Ctrl+C to stop)")

    buf, size = None, 0
    while True:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode(errors="replace").strip()
        word = line.split(" ")

        if word[0] == "PHOTO" and len(word) == 4:
            size = int(word[1])
            buf = bytearray(size)
            got = 0
            print(f"receiving {word[2]}x{word[3]} photo, {size} bytes ", end="", flush=True)

        elif word[0] == "DATA" and len(word) == 3 and buf is not None:
            off, n = int(word[1]), int(word[2])
            chunk = port.read(n)
            if len(chunk) != n:
                print("\ntimed out in the middle of a piece")
                buf = None
                continue
            buf[off:off + n] = chunk
            got += n
            print(".", end="", flush=True)

        elif word[0] == "DONE" and len(word) == 2 and buf is not None:
            crc_ok = f"{zlib.crc32(buf) & 0xFFFFFFFF:08x}" == word[1].lower()
            jpeg_ok = buf[:2] == b"\xff\xd8" and got == size
            if tag is None:
                name = datetime.datetime.now().strftime("photo_%Y%m%d_%H%M%S.jpg")
            elif args.shim is not None:
                name = f"shim_{args.shim:.1f}_{tag}.jpg"
            else:
                name = f"pair_{stamp}_{tag}.jpg"
            path = os.path.join(args.out, name)
            with open(path, "wb") as fh:
                fh.write(buf)
            status = "OK" if (crc_ok and jpeg_ok) else "CHECK FAILED - photo may be damaged"
            print(f"\nsaved {path}  [{status}]")
            buf = None
            if tag == "A":
                last_a = path if (crc_ok and jpeg_ok) else None
            elif tag == "B" and last_a and crc_ok and jpeg_ok:
                if analyse:
                    try:
                        r = analyse(last_a, path)
                        depths = ", ".join(f"{g['depth_mm']:.2f}" for g in r["grooves"]) or "none"
                        print(f"  -> {r['verdict']}   grooves (mm): {depths}   "
                              f"valid {r['valid_columns_frac']*100:.0f} %  contrast {r['mean_contrast']:.0f}")
                        for w in r["warnings"]:
                            print(f"     note: {w}")
                    except Exception as e:      # a bad shot must not stop the logger
                        print(f"  -> pipeline could not read this pair: {e}")
                if args.shim is not None:
                    nxt = input(f"  shim {args.shim} mm done. Next shim thickness in mm (Enter = finish): ").strip()
                    if not nxt:
                        return
                    args.shim = float(nxt)
                    print(f"  put the {args.shim} mm spacers under the feet, then press B1")
            if tag == "B":
                tag = None

        elif word[0] == "SHOT" and len(word) == 2:
            tag = word[1]
            if tag == "A":
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        elif word[0] == "FAIL":
            print(f"\nNucleo reports: {line}")
            buf = None

        else:
            print(f"[nucleo] {line}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
