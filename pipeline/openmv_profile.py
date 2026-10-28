# openmv_profile.py - the heart of laser_profile.py, written for the OpenMV
# board (MicroPython + ulab). This is what will answer the PROFILE command
# (0x04) in the link protocol: the camera does the pixel work and sends the
# Nucleo one number per column, not an image.
#
# Steps 2-4 of laser_profile.py (subtract, smooth, per-column sub-pixel line).
# Steps 5-7 (shoulder line, depth, grooves) are cheap and can live on the
# Nucleo in depth.c, so the CCD branch can share them.
#
# NOT YET TESTED ON HARDWARE. Test plan: run it in OpenMV IDE with the laser
# on a flat plate, print profile[::20], compare with laser_profile.py on the
# same scene photographed through PHOTO. The numbers should agree to ~0.1 px.
#
# ulab notes: `from ulab import numpy as np` gives a small numpy. No NaN-aware
# functions, so invalid columns are marked with -1 instead of NaN. No argmax
# along an axis on older builds, so we work row-by-row with running maxima.

from ulab import numpy as np

WIN = 3            # +-rows around the peak for the centroid
MIN_CONTRAST = 40  # same as calib.json


def frame_to_array(img):
    """GRAYSCALE image -> uint8 array (h, w). The bytearray is a view, so this
    is cheap; the sum/argmax below convert to float a column at a time."""
    return np.frombuffer(img.bytearray(), dtype=np.uint8).reshape((img.height(), img.width()))


def profile(img_on, img_off):
    """Return (y, contrast): two float arrays of length w.
    y[c] = sub-pixel row of the laser line in column c, or -1 if invalid."""
    on = frame_to_array(img_on)
    off = frame_to_array(img_off)
    h, w = on.shape

    # 2. subtract, clipped at 0, as int16 so we don't wrap around
    d = np.array(on, dtype=np.int16) - np.array(off, dtype=np.int16)
    d = np.where(d < 0, 0, d)

    # 3. smooth vertically with a 3-tap [1 2 1]/4 kernel (close enough to the
    #    Gaussian the PC uses; a wider one costs RAM we don't have)
    s = np.zeros(d.shape, dtype=np.int16)
    s[1:-1, :] = (d[:-2, :] + 2 * d[1:-1, :] + d[2:, :]) // 4

    # 4a. brightest row per column
    peak_row = np.argmax(s, axis=0)
    cols = np.arange(w)
    contrast = np.array([s[peak_row[c], c] for c in range(w)], dtype=np.float)

    # 4b. centroid in a +-WIN window with half the peak subtracted
    y = np.full(w, -1.0, dtype=np.float)
    for c in range(w):
        pk = int(peak_row[c])
        if contrast[c] < MIN_CONTRAST:
            continue
        lo = max(0, pk - WIN)
        hi = min(h, pk + WIN + 1)
        seg = np.array(s[lo:hi, c], dtype=np.float) - 0.5 * contrast[c]
        seg = np.where(seg < 0, 0, seg)
        tot = np.sum(seg)
        if tot <= 0:
            continue
        rows = np.arange(lo, hi, dtype=np.float)
        y[c] = np.sum(rows * seg) / tot
    return y, contrast


def pack_profile(y, contrast):
    """Payload for response 0x84: u16 w, then per column u16 y*16 (fixed point,
    0xFFFF = invalid) and u8 contrast. 320 columns -> 962 bytes."""
    import struct
    w = len(y)
    out = bytearray(struct.pack("<H", w))
    for c in range(w):
        v = 0xFFFF if y[c] < 0 else int(y[c] * 16 + 0.5)
        out += struct.pack("<HB", v, int(min(255, contrast[c])))
    return bytes(out)


# ---- how it plugs into main.py (sketch):
#
#   elif cmd == 0x04:                       # PROFILE
#       laser.off(); img_off = snapshot().copy()
#       laser.on();  img_on  = snapshot(); laser.off()
#       y, con = profile(img_on, img_off)
#       link.send(0x84, pack_profile(y, con))
#
# `laser` is a Pin on the OpenMV side only if the laser MOSFET is driven from
# the camera. In the current design the Nucleo drives it (PA8 -> MOSFET), so
# the Nucleo must toggle the laser between two GRAB commands instead, then
# ask for PROFILE on slots 0 and 1. Either way, keep the two exposures as
# close together in time as possible so the hand hasn't moved.
