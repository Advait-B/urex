# UREx image pipeline: from laser-line photo to tread depth

Status 26 Oct 2026. The camera chain works (B1 → OpenMV → Nucleo → laptop). One press now takes a
**laser-off / laser-on pair** (`main.c` PAIR_MODE), with exposure locked on the camera. This folder is
the maths that turns that pair into "groove 2 is 2.5 mm deep → PASS".

```
 laser-off A ─┐
              ├ subtract ─ smooth ─ sub-pixel line per column ─ TRIANGULATE (calibrated) ─ RANSAC shoulder ─ grooves ─ verdict
 laser-on  B ─┘   (steps 2-4 will run on the OpenMV)             row → mm, per column       depth = z − shoulder
```

| File | What it does | Runs on |
|---|---|---|
| `focus_assist_openmv.py` | Live sharpness number while you twist the lens. **Do this first.** | OpenMV IDE |
| `focus_check.py` | Scores saved photos: sharp / dark / saturated. Contact sheet option. | laptop |
| `calibrate.py` | **Shim test → height calibration** (`height_poly` in `calib.json`). | laptop |
| `laser_profile.py` | **The pipeline.** Photo pair → JSON result + diagnostic plot. | laptop |
| `synth_tyre.py` | Fake A/B photos using the real triangulation physics, or a fake shim run. | laptop |
| `test_pipeline.py` | Calibrates on fake shims, then 14 fake tyres (new 16 mm, worn 0.8 mm, crooked, blurry…). All pass. | laptop |
| `calib.json` | Settings + calibration. Until you run `calibrate.py` it has only the design-sheet guess. | both |
| `openmv_profile.py` | ulab port of steps 2-4 for the camera (PROFILE command). Not yet run on hardware. | OpenMV |

Setup on the laptop (one time): `bash ../setup_mac.sh` (installs numpy, opencv, matplotlib, pyserial).

```bash
cd pipeline
../.venv/bin/python test_pipeline.py              # should print ALL OK
cd .. && bash take_photo.sh --analyse             # press B1: saves pair_..._A/B.jpg and prints the depths
```

---

## How the triangulation works

### The idea in one picture

```
            camera (looks straight down)
               |
               |        laser (tilted at θ ≈ 35°)
               |       /
               |      /
   tread  ─────●─────/───────          ← line hits the tread here
               |    /
   groove      |   ●                   ← in a groove the surface is z mm further away,
   bottom      |                         so the sheet hits it z·tanθ further along
```

The camera looks straight at the tyre. The laser sheet comes in at an angle. Where the rubber is further
away (a groove), the line lands further along, and the camera sees it **shifted by some pixels**. Measure
the shift, convert to millimetres, done. The whole problem is that conversion.

### Why "mm per pixel" is not good enough

The optics sheet's formula is `shift_px = z·tanθ·f / ((L + z − f)·pitch)`. The `z` in the bottom line
means a deeper surface is also *further from the lens*, so it looks smaller: each extra millimetre gives
fewer pixels. With our numbers:

| depth z | shift | px per mm here |
|---|---|---|
| 1 mm | 4.0 px | 4.0 |
| 8 mm | 29.4 px | 3.4 |
| 16 mm | 54.3 px | 3.1 |

A single constant (0.249 mm/px, the slope at the surface) reads **a 16 mm groove as 13.5 mm**. It's
fine for a worn tyre and badly wrong for a new one (bus tyres start at ~16 mm). Using the formula
directly doesn't help either: 1° of error in θ is 4 % (0.6 mm at 16 mm), the lens focal length is
±5 %, and the lens bends lines near the image edges.

### What we do instead: calibrate, then look it up

1. **Calibrate once.** Photograph a flat plate at known distances: shims under the device's feet
   (0, 2, 4 … 20 mm). For every column, `calibrate.py` now knows which row the line sits in at each
   height. It fits one smooth surface, `z = poly(column, row)` (cubic in row, quadratic across), through
   ~3000 points. That captures the angle, lens, distortion and laser tilt together. Nothing needs a
   protractor.
2. **Measure.** For each column, find the line row (sub-pixel) and look up its **absolute height** z.
3. **Subtract the shoulder.** The tread surface is the straight line (in mm) most columns agree on
   (RANSAC). `depth = z − shoulder`. Because both are absolute heights, it doesn't matter if the
   device sits 3 mm off the tread or is held crooked. `standoff_mm` in the result tells you where the
   surface was. Expect 0–3 mm; a bigger number means the feet weren't on the tread.

Synthetic result (`test_pipeline.py`, JPEG q95 like the camera sends):

| case | calibrated error | straight-line error |
|---|---|---|
| worn tyre 7 / 6 / 2.5 mm | 0.02 mm | 0.54 mm |
| new bus tyre 16 / 15 / 16 mm | 0.01 mm | 2.49 mm |
| device 3 mm off the tread | 0.02 mm | 2.88 mm |
| held crooked (4 mm tilt across) | 0.02 mm | 1.94 mm |
| 1.2 mm illegal groove | 0.005 mm → FAIL | 0.16 mm |

### The shim calibration, step by step (≈15 minutes)

You need: a flat **matt** plate (a ceramic tile, painted MDF; not shiny metal, which makes laser glare),
spacers you can stack under the three feet (washers, stacked PCBs, 3D-printed blocks), calipers.

```bash
bash take_photo.sh --out photos/calib --shim 0
```
1. Device flat on the plate, feet touching. Press B1. The script saves `shim_0.0_A/B.jpg` and asks for
   the next thickness.
2. Put spacers under **all three** feet. Measure them with calipers (don't trust the label). Type the
   measured value, press B1. Repeat: about 0, 2, 4, 6, 8, 10, 12, 15, 18, 20 mm. At least 6 heights.
3. Press Enter to finish, then:
```bash
cd pipeline && ../.venv/bin/python calibrate.py ../urex-camera-photo/urex-camera-photo/pc/photos/calib --write calib.json
```
4. Read the table. **bias** (systematic error per shim) should be < 0.05 mm. **noise** (single-column
   scatter, averaged away inside a groove) should be < 0.15 mm. The last column shows what the old
   straight-line method would have said.

After calibrating: **don't move the lens, the laser, or the bracket**, and don't change FRAMESIZE.
The pipeline refuses to use a calibration made at a different image size.

---

## Can software "focus" a blurry photo? No

Defocus can't be undone by a filter without biasing the sub-pixel centre. Focus is a lens twist, set once
at 90 mm (`focus_assist_openmv.py`) and locked. The pipeline tolerates ~2 px of blur. Beyond that,
contrast collapses and the quality gate (valid columns ≥ 90 %, contrast > 60) says RETAKE instead of
guessing.

## Reading the result

```json
{ "valid_columns_frac": 1.0, "mean_contrast": 123.5,
  "shoulder_line": {"slope_mm_per_col": 0.0001, "intercept_mm": 2.98, "inlier_frac": 0.74, "standoff_mm": 3.0},
  "grooves": [{"col_start": 31, "col_end": 65, "width_mm": 11.9, "depth_mm": 16.01, "depth_max_mm": 16.06}, ...],
  "min_groove_depth_mm": 1.21, "quality_ok": true, "warnings": [], "verdict": "FAIL" }
```

| verdict | meaning |
|---|---|
| PASS | every groove found is ≥ 1.6 mm |
| FAIL | at least one groove < 1.6 mm |
| CHECK | no grooves found, or fewer than `expected_grooves`. Worn flat, or the line missed the tread. **Not a pass.** |
| RETAKE | photo too poor to judge (blur, dim laser, line off the edge) |

Grooves are detected at **0.5 mm**, far below the 1.6 mm limit, on purpose. An earlier 1.0 mm
threshold let a 1.2 mm groove that measured slightly shallow fall apart into fragments and vanish,
giving PASS. A worn groove must be *found* so it can fail. Set `expected_grooves` in `calib.json`
(e.g. 4 for your test tyre) and a missing groove becomes CHECK.

## Things the pipeline assumes (check on the bench)

- The laser line runs **left-right** across the image and grooves shift it **vertically**. If the line
  is vertical, transpose the images first (one line in `load_gray`).
- `rotate180: true` because the camera is mounted upside down. Alternatively set `ROTATE_180 = True`
  in the camera's `main.py` and `rotate180: false` here. Not both. **Recalibrate after changing it.**
- The tread surface covers more than half the line. That's true for a bus tyre, where grooves are
  ≈ 30 % of the width.
- `mm_per_px_lateral` (0.35, groove *width* only) is still the design-sheet value. Check it by
  photographing a ruler at 90 mm.

## What's next, in order

1. **Focus and lock the lens** at 90 mm (`focus_assist_openmv.py`).
2. **Flash the new `main.c` + copy the new `main.py`** to the camera. Then check a pair: in
   `pair_..._A.jpg` the line must be **absent**, and in `_B.jpg` present. If A shows the line,
   increase `DISCARD_FRAMES` in `main.c`. If B is saturated (`focus_check.py` clip% > 2), set
   `EXPOSURE_US` in `main.py` to something shorter (try 4000).
3. **Shim calibration** (above).
4. **First real tyre** against a tread-depth gauge, 3 spots × 3 presses. Report mean and spread.
   That's the number for the next meeting.
5. **Port to the camera** (`openmv_profile.py`, PROFILE 0x04). Then `height_mm → RANSAC → find_grooves`
   becomes `depth.c` on the Nucleo, shared with the CCD branch.
