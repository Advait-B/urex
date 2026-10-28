# focus_assist_openmv.py - run this in OpenMV IDE while you twist the lens.
#
# WHY: the OpenMV H7 Plus has a manual-focus M12 lens. Ours is set for "far"
# (the ceiling photos are sharp, the 5 cm breadboard photo is mush). Software
# cannot undo defocus, so we set the lens ONCE for our working distance and
# then lock it. This script shows a live sharpness number so you know when
# the lens is right instead of guessing by eye on a small screen.
#
# HOW:
#   1. Plug the camera into the laptop by USB, open OpenMV IDE, Connect, Run this.
#   2. Put a printed target (text, a ruler, a chequerboard) at the working
#      distance L = 90 mm from the front of the lens (see cmos-branch-visuals).
#   3. Twist the lens slowly. Watch SHARP in the frame buffer / serial terminal.
#      It rises, peaks, falls. Go back to the peak. BEST remembers the highest
#      value seen so you know how close you are.
#   4. When at the peak, hold the lens and tighten the set screw (or a dab of
#      nail varnish / threadlocker on the thread). Re-check the number.
#
# The number is the variance of a Laplacian (edge) filter over the centre of
# the image: sharp edges -> big variance. It is the same metric the PC-side
# focus_check.py uses on saved photos, so you can compare.

import time

ROTATE_180 = True       # our camera is mounted upside down (see photos); set False if not

# Newer firmware has `csi`, older only `sensor`.
try:
    import csi
    cam = csi.CSI()
    cam.reset()
    cam.pixformat(csi.GRAYSCALE)
    cam.framesize(csi.QVGA)
    if ROTATE_180:
        cam.vflip(True)
        cam.hmirror(True)
    cam.snapshot(time=1500)
    snapshot = cam.snapshot
except ImportError:
    import sensor
    sensor.reset()
    sensor.set_pixformat(sensor.GRAYSCALE)
    sensor.set_framesize(sensor.QVGA)
    if ROTATE_180:
        sensor.set_vflip(True)
        sensor.set_hmirror(True)
    sensor.skip_frames(time=1500)
    snapshot = sensor.snapshot

best = 0.0
clock = time.clock()

while True:
    clock.tick()
    img = snapshot()
    w, h = img.width(), img.height()

    # Score only the central third: that is where the laser line will be.
    roi = (w // 3, h // 3, w // 3, h // 3)
    patch = img.copy(roi=roi)          # work on a copy so the display stays clean
    patch.laplacian(1)                 # 3x3 edge filter, in place
    sharp = patch.get_statistics().stdev() ** 2
    best = max(best, sharp)

    img.draw_rectangle(roi, color=255)
    img.draw_string(4, 4, "SHARP %5.0f" % sharp, color=255, scale=2, mono_space=False)
    img.draw_string(4, 26, "BEST  %5.0f" % best, color=255, scale=2, mono_space=False)
    # a bar: full width = BEST, so it fills up as you approach the peak
    bar = int((w - 8) * sharp / best) if best > 0 else 0
    img.draw_rectangle(4, h - 12, bar, 8, color=255, fill=True)
    print("SHARP %.0f  BEST %.0f  fps %.1f" % (sharp, best, clock.fps()))
