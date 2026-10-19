"""get_photo.py - save the photos the NUCLEO-C542RC sends over its USB cable.

    pip install pyserial
    python get_photo.py COM5              (Windows)
    python get_photo.py /dev/tty.usbmodem* (macOS)   python get_photo.py /dev/ttyACM0 (Linux)

Leave it running, press the blue B1 button on the Nucleo, and each photo is
written to ./photos/photo_YYYYmmdd_HHMMSS.jpg. Close the serial terminal in
STM32CubeIDE first: only one program can open the port at a time.
"""
import argparse
import datetime
import os
import sys
import zlib

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", help="ST-LINK virtual COM port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="photos", help="folder for the .jpg files")
    args = ap.parse_args()

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
            name = datetime.datetime.now().strftime("photo_%Y%m%d_%H%M%S.jpg")
            path = os.path.join(args.out, name)
            with open(path, "wb") as fh:
                fh.write(buf)
            status = "OK" if (crc_ok and jpeg_ok) else "CHECK FAILED - photo may be damaged"
            print(f"\nsaved {path}  [{status}]")
            buf = None

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
