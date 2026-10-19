# main.py - OpenMV Cam H7 Plus: take a photo when the Nucleo asks for one.
#
# Commands handled (see stm32/proto.h):
#   0x01 PING   -> 0x81 "OMV1"
#   0x02 SNAP   -> 0x82 w, h, mean, 80x60 thumbnail      (Stage 1 of the manual)
#   0x06 PHOTO  -> 0x86 u32 size, u16 w, u16 h, u16 shot  (captures + JPEG)
#   0x07 READ   -> 0x87 u32 offset, bytes                 (one piece of the JPEG)
#
# Save this file and link.py onto the camera's drive (or SD card) so it runs
# by itself at power-up, without the OpenMV IDE.
import os
import struct
from link import Link

PIXFORMAT = "GRAYSCALE"   # "GRAYSCALE" (what the laser pipeline uses) or "RGB565" (colour)
FRAMESIZE = "QVGA"        # "QVGA" 320x240 (~1-2 s to transfer) or "VGA" 640x480 (slower)
CHUNK_MAX = 512           # must match PROTO_CHUNK_MAX in proto.h

# Newer firmware has the `csi` module; older firmware only has `sensor`.
try:
    import csi
    cam = csi.CSI()
    cam.reset()
    cam.pixformat(getattr(csi, PIXFORMAT))
    cam.framesize(getattr(csi, FRAMESIZE))
    cam.snapshot(time=2000)              # let auto-exposure settle
    snapshot = cam.snapshot
except ImportError:
    import sensor
    sensor.reset()
    sensor.set_pixformat(getattr(sensor, PIXFORMAT))
    sensor.set_framesize(getattr(sensor, FRAMESIZE))
    sensor.skip_frames(time=2000)
    snapshot = sensor.snapshot

# Keep a copy of every photo on the SD card if one is fitted.
try:
    os.stat("/sdcard")
    SAVE_DIR = "/sdcard"
except OSError:
    SAVE_DIR = None

link = Link()
jpeg = b""        # last photo, waiting to be read by the Nucleo
shot = 0


def take_photo(quality):
    global jpeg, shot
    img = snapshot()
    w, h = img.width(), img.height()
    jpg = img.to_jpeg(quality=quality, copy=True)   # copy: frame buffer is reused
    jpeg = bytes(jpg.bytearray()[:jpg.size()])
    saved = 0xFFFF
    if SAVE_DIR:
        try:
            jpg.save("%s/photo_%04d.jpg" % (SAVE_DIR, shot))
            saved = shot
        except OSError:
            pass                                    # card full / missing: still answer
    shot += 1
    return struct.pack("<IHHH", len(jpeg), w, h, saved)


while True:
    msg = link.recv()
    if msg is None:
        continue
    cmd, arg = msg

    if cmd == 0x01:                                 # PING
        link.send(0x81, b"OMV1")

    elif cmd == 0x02:                               # SNAP (Stage 1 thumbnail, 4800 B in GRAYSCALE)
        img = snapshot()
        mean = img.get_statistics().mean()
        thumb = img.copy(x_scale=0.25, y_scale=0.25)
        link.send(0x82, struct.pack("<HHB", img.width(), img.height(), mean)
                  + bytes(thumb.bytearray()))

    elif cmd == 0x06:                               # PHOTO
        q = arg[0] if len(arg) >= 1 and 1 <= arg[0] <= 100 else 80
        link.send(0x86, take_photo(q))

    elif cmd == 0x07 and len(arg) == 6:             # READ offset, n
        off, n = struct.unpack("<IH", arg)
        n = min(n, CHUNK_MAX, max(0, len(jpeg) - off))
        link.send(0x87, struct.pack("<I", off) + jpeg[off:off + n])
