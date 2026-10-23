# UREx image pipeline: from laser-line photo to tread depth

Status 23 Oct 2026. The camera chain works (B1 → OpenMV → Nucleo → laptop, 32 photos in
`../urex-camera-photo/urex-camera-photo/pc/photos`). This folder is the next layer: the maths
that turns a photo of the laser line into "groove 2 is 2.5 mm deep → PASS".

```
 laser-off frame A ─┐
                    ├─ subtract ─ smooth ─ per-column sub-pixel line ─ RANSAC shoulder ─ depth mm ─ grooves ─ PASS/FAIL
 laser-on  frame B ─┘        (steps 2-4 will run on the OpenMV)        (steps 5-7 can run on the Nucleo)
```

| File | What it does | Runs on |
|---|---|---|
| `focus_assist_openmv.py` | Live sharpness number while you twist the lens. **Do this first.** | OpenMV IDE |
| `focus_check.py` | Scores saved photos: sharp / dark / saturated. Contact sheet option. | laptop |
| `synth_tyre.py` | Makes fake laser-line A/B photos of a tread with known groove depths. | laptop |
| `laser_profile.py` | **The pipeline.** Photo pair → JSON result + diagnostic plot. | laptop |
| `test_pipeline.py` | 11 synthetic cases (blur, noise, dim, saturated, tilted, 1.2 mm illegal …). All pass. | laptop |
| `calibrate.py` | Shim test → measured `mm_per_px_depth` written into `calib.json`. | laptop |
| `calib.json` | Optics constants. Starts with the design-sheet estimate, replaced by the shim test. | both |
| `openmv_profile.py` | ulab port of steps 2-4 for the camera (PROFILE command). Not yet run on hardware. | OpenMV |

Setup on the laptop (one time): `pip install numpy opencv-python matplotlib` into the project venv
(`../.venv/bin/pip install ...`). Then:

```bash
cd pipeline
python test_pipeline.py                        # should print ALL OK
python synth_tyre.py out/demo --grooves 7,12 2.5,9 1.2,8
python laser_profile.py out/demo_B.png --off out/demo_A.png --plot out/demo.png
python focus_check.py ../urex-camera-photo/urex-camera-photo/pc/photos --sheet out/sheet.jpg
```

## Can software "focus" a blurry photo? No — and here is why it matters anyway

The breadboard photo from 19 Oct (`photo_20261019_163437.jpg`, sharpness 7 vs 290 for the ceiling)
is defocused because the lens is set for far objects. No filter recovers that: deconvolution amplifies
noise and would bias the sub-pixel centroid we depend on. **Focus is a lens twist** (see
`focus_assist_openmv.py`). The pipeline itself is quite blur-tolerant — a blurred Gaussian line is
still a Gaussian with the same centre — but blur spreads the laser's light over more pixels, so
contrast drops and the centroid gets noisy:

| synthetic case | blur σ | contrast | depth error | verdict |
|---|---|---|---|---|
| sharp | 0 px | 144 | 0.011 mm | PASS |
| typical | 1 px | 123 | 0.006 mm | PASS |
| defocused | 2 px | 93 | 0.012 mm | PASS |
| defocused | 4 px | 58 | 0.093 mm | **RETAKE** (quality gate) |
| dim laser | 1 px | 45 | — | **RETAKE** |

So: set the focus once at the 90 mm working distance, lock it, and let the quality gate
(valid columns ≥ 90 %, contrast > 60) refuse bad shots instead of guessing.

## Reading the result

```json
{ "valid_columns_frac": 1.0, "mean_contrast": 123.5,
  "shoulder_line": {"slope_px_per_col": 0.012, "intercept_px": 117.0, "inlier_frac": 0.74},
  "grooves": [{"col_start": 31, "col_end": 65, "width_mm": 11.9, "depth_mm": 7.01, "depth_max_mm": 7.08}, ...],
  "min_groove_depth_mm": 1.21, "quality_ok": true, "warnings": [], "verdict": "FAIL" }
```
`verdict` is PASS, FAIL (a groove under 1.6 mm), NO_GROOVE (flat surface or the line missed the
grooves) or RETAKE (shot too poor to judge). `depth_max_mm` is the deepest point; `depth_mm` is the
median of the groove's central half, which is what the tread-depth gauge would read.

## Things the pipeline assumes (check on the bench)

- Laser line runs **left-right** across the image; grooves shift it **vertically**. If the laser is
  mounted so the line is vertical, transpose the images first (one line in `load_gray`).
- `rotate180: true` because the camera is mounted upside down (every 19 Oct photo is inverted).
  Either keep this, or set `vflip/hmirror` on the camera and set it false — not both.
- `depth_sign`: grooves must come out positive. The pipeline warns if they don't; `calibrate.py` sets it.
- Tread surface occupies more than half the line (true for a bus tyre: grooves ≈ 30 % of width).
  RANSAC finds the line most columns agree on, so the grooves can't fool it.
- Working distance 90 mm, 0.249 mm depth per px of shift, 0.35 mm per column across the tread.
  All from the 13 Sep optics sheet, all to be replaced by measurement.

## What's next, in order

1. **Focus and lock the lens** at 90 mm with `focus_assist_openmv.py`. Fix the upside-down mounting
   (either physically or with `vflip/hmirror` in `main.py` — then set `rotate180: false`).
2. **Wire the laser** (9 V rail, AO3400 low-side, PA8 → gate; see cmos-stm32-bringup) and add a
   `GRAB` that takes laser-off then laser-on. Even before that: shine any line laser / laser pointer
   through a glass rod by hand and take two B1 photos — that's enough to run `laser_profile.py` on
   real pixels for the first time.
3. **Shim calibration** on a flat plate with 0/2/4/6/8 mm shims → `calibrate.py --write calib.json`.
   Check the linearity residual is < 0.1 mm.
4. **First real tyre** (any car tyre will do): photo pairs at three spots, compare `depth_mm` to a
   cheap tread-depth gauge. This is the number for the next meeting.
5. **Port to the camera**: run `openmv_profile.py` in OpenMV IDE on the same scene, check its `y`
   matches `laser_profile.find_line` to ~0.1 px, then wire it to the PROFILE (0x04) command so the
   Nucleo gets 962 bytes per press instead of a 6 kB JPEG.
6. The same `depth_from_line → find_grooves` logic becomes `depth.c` on the Nucleo so the CCD branch
   (one row of 2048 pixels = one "column" scan) feeds the same decision code.
