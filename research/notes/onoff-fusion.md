# onoff-fusion: what do extra frames buy us?

Branch `research/onoff-fusion`. Two questions: (A) is the laser-OFF frame worth capturing, and how much
device motion between the two frames can it take; (B) do several presses per tyre position help.

Everything here is **simulation**. The 86 real photos have **no laser-off frame**, so nothing in this
note is validated on real data: on real photos every `onoff_*` module falls back to the single-frame
baseline, and the `real` block in `results/onoff_*.json` is just the baseline's numbers again.

## What was built
| file | what |
|---|---|
| `algos/onoff_common.py` | difference modes (plain / gain / registered / conservative / robust), the baseline detector on any image, a per-column centre-of-gravity (COG) detector |
| `algos/onoff_steger.py` | (1) `on - off`, then the unchanged baseline detector (top-hat + Steger + chain) |
| `algos/onoff_cog.py` | (2) `on - off`, then per-column peak + centre of gravity: no top-hat, no Hessian |
| `algos/onoff_robust.py` | (3) gain match + phase-correlation registration + subtract the off frame dilated by 1 px, then COG |
| `experiments/onoff-fusion/perturb.py` | moves / rotates / re-exposes the off frame |
| `experiments/onoff-fusion/{dev_train,motion_sweep,factors,presses,fuse,figs}.py` | the experiments below |

Choices made on `train` only (first 200 frames, `dev_train.py`): COG instead of Steger after the
subtraction (depth MAE 0.121 vs 0.665 mm), and COG as the detector of the robust variant. The COG
parameters were written down once and not tuned. One leak to declare: before that I ran the three
modules once with `--limit 60` on ideal/stress/legal as a smoke test; no parameter was changed afterwards.

## Part A: laser-off subtraction

### Table A1. Harness results (`python -m bench.run algos.<m> --out results/<m>.json`)
| algorithm | set | line F1 | RMSE px | depth MAE mm | P95 mm | bias mm | miss | false/frame | verdict acc | false pass |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_steger (ON only) | ideal | 0.931 | 0.162 | 0.147 | 0.474 | 0.068 | 0.090 | 0.093 | 0.980 | 0.013 |
| baseline_steger | stress | 0.821 | 0.235 | 1.516 | 6.678 | 1.017 | 0.099 | 0.280 | 0.915 | 0.015 |
| baseline_steger | legal | 0.977 | 0.155 | 0.053 | 0.210 | -0.000 | 0.054 | 0.127 | 0.993 | 0.000 |
| onoff_steger | ideal | 0.930 | 0.161 | 0.181 | 0.480 | 0.025 | 0.086 | 0.073 | 0.953 | 0.013 |
| onoff_steger | stress | 0.853 | 0.227 | 0.967 | 2.451 | 0.476 | 0.096 | 0.135 | 0.900 | 0.010 |
| onoff_steger | legal | 0.978 | 0.153 | 0.057 | 0.132 | 0.002 | 0.057 | 0.127 | 0.993 | 0.000 |
| **onoff_cog** | ideal | 0.980 | 0.101 | 0.062 | 0.192 | 0.049 | 0.015 | 0.020 | 0.987 | 0.013 |
| **onoff_cog** | stress | 0.968 | 0.196 | 0.147 | 0.396 | 0.008 | 0.003 | 0.030 | 0.980 | 0.000 |
| **onoff_cog** | legal | 0.997 | 0.108 | 0.026 | 0.066 | -0.016 | 0.054 | 0.000 | 0.987 | 0.007 |
| onoff_robust | ideal | 0.979 | 0.100 | 0.062 | 0.191 | 0.050 | 0.011 | 0.020 | 0.987 | 0.013 |
| onoff_robust | stress | 0.966 | 0.193 | 0.143 | 0.377 | 0.016 | 0.012 | 0.025 | 0.970 | 0.000 |
| onoff_robust | legal | 0.997 | 0.108 | 0.026 | 0.066 | -0.016 | 0.054 | 0.000 | 0.987 | 0.007 |
| oracle_line (ceiling) | stress | 0.998 | 0 | 0.108 | 0.344 | 0.069 | 0.006 | 0.005 | 1.000 | 0.000 |

Real photos: not applicable (no off frame). The modules fall back to the baseline there.

`onoff_cog` on stress is close to the oracle (0.147 vs 0.108 mm). On `legal` it has one false pass in
150 frames (0.007) where the baseline has none; on that set the two differ by a single frame in verdict.

### Table A2. Motion sweep on `stress` (200 frames, off frame perturbed by `perturb.py`)
Cells: depth MAE mm / P95 mm / false grooves per frame. Single-frame baseline for reference:
**1.516 / 6.678 / 0.280** (it does not use the off frame, so it does not change).

| off frame | plain + Steger | plain + COG | registered + COG | conservative (2 px guard) + COG | robust + COG |
|---|---|---|---|---|---|
| no motion | 0.967 / 2.451 / 0.135 | 0.147 / 0.396 / 0.030 | 0.148 / 0.395 / 0.030 | 0.167 / 0.378 / 0.050 | 0.143 / 0.377 / 0.025 |
| 0.5 px | 0.970 / 2.439 / 0.135 | 0.146 / 0.407 / 0.030 | 0.147 / 0.398 / 0.030 | 0.142 / 0.374 / 0.050 | 0.159 / 0.393 / 0.040 |
| 1 px | 1.067 / 2.650 / 0.170 | 0.747 / 0.476 / 0.060 | 0.147 / 0.406 / 0.025 | 0.144 / 0.375 / 0.035 | 0.143 / 0.377 / 0.025 |
| 2 px | 1.316 / 2.742 / 0.200 | **3.459 / 7.413 / 0.150** | 0.147 / 0.401 / 0.040 | 0.159 / 0.380 / 0.030 | 0.143 / 0.376 / 0.025 |
| 3 px | 1.269 / 2.991 / 0.235 | 4.388 / 23.503 / 0.270 | 0.566 / 0.397 / 0.035 | 0.283 / 0.420 / 0.065 | 0.158 / 0.376 / 0.045 |
| 5 px | 1.444 / 6.527 / 0.335 | 6.775 / 51.757 / 0.345 | 1.051 / 0.440 / 0.050 | 3.593 / 13.958 / 0.150 | 0.341 / 0.384 / 0.030 |
| 1 px + 0.3 deg | 0.932 / 2.475 / 0.155 | 1.050 / 0.473 / 0.085 | 0.189 / 0.411 / 0.055 | 0.144 / 0.376 / 0.035 | 0.160 / 0.375 / 0.035 |
| exposure +-20 % | 1.034 / 2.544 / 0.145 | 0.479 / 0.426 / 0.060 | 0.131 / 0.395 / 0.040 | 0.221 / 0.387 / 0.095 | 0.137 / 0.390 / 0.040 |
| 2 px + 0.3 deg + exp +-20 % | 1.051 / 2.474 / 0.210 | 2.359 / 3.885 / 0.155 | 0.132 / 0.388 / 0.075 | 0.242 / 0.428 / 0.095 | 0.126 / 0.387 / 0.060 |

"registered" and "robust" include the gain match; "plain" does not. Full metrics (and the 97 sunniest
frames separately) are in `experiments/onoff-fusion/motion_{a,b}.json`. Figure: `figs/onoff-fusion/error_vs_motion.png`.

### Table A3. One stress factor at a time (40 ideal scenes + one factor, `factors.py`)
Cells: line F1 / depth MAE mm.

| factor | ON only, Steger (baseline) | ON only, top-hat + COG | ON-OFF, Steger | ON-OFF, COG |
|---|---|---|---|---|
| none | 0.928 / 0.180 | 0.980 / 0.060 | 0.926 / 0.183 | 0.979 / 0.061 |
| sunlight (ambient 1.0) | 0.943 / 0.283 | 0.979 / 0.063 | 0.944 / 0.122 | 0.980 / 0.204 |
| sunlight + gradient | 0.940 / 0.282 | 0.979 / 0.061 | 0.942 / 0.117 | 0.980 / 0.062 |
| 6 wet glints near the line | 0.606 / 12.296 | 0.518 / 7.509 | 0.919 / 0.195 | 0.965 / 0.085 |
| paint stripe, 80 DN | 0.917 / 0.136 | 0.978 / 0.079 | 0.938 / 0.109 | 0.979 / 0.062 |
| paint stripe, saturating the sensor | 0.868 / 4.223 | 0.896 / 2.810 | 0.897 / 4.118 | 0.924 / 3.978 |
| saturated laser (600 DN) | 0.942 / 0.567 | 0.987 / 0.062 | 0.941 / 0.511 | 0.987 / 0.062 |
| dim laser (40 DN) | 0.797 / 0.045 | 0.975 / 0.207 | 0.727 / 0.039 | 0.960 / 0.334 |
| blur 2.4 px + wide line | 0.939 / 0.866 | 0.971 / 0.065 | 0.941 / 0.848 | 0.969 / 0.061 |
| 5 laser spikes | 0.920 / 0.469 | 0.977 / 0.225 | 0.924 / 0.538 | 0.976 / 0.071 |
| rocking 1.5 deg | 0.933 / 0.298 | 0.980 / 0.220 | 0.926 / 0.321 | 0.979 / 0.220 |

With 40 scenes the MAE column moves by 0.1-0.2 mm when one groove goes wrong (e.g. the 0.204 under plain
sunlight, where the gradient row next to it is 0.062): read differences of that size as noise.

### Findings, part A
1. **Measured: the off frame alone does almost nothing for the baseline detector.** `onoff_steger` vs
   baseline: stress F1 0.821 -> 0.853, MAE 1.516 -> 0.967, ideal slightly worse (0.147 -> 0.181), verdict
   accuracy slightly worse on ideal and stress. The baseline's top-hat already removes the smooth background;
   what limits it is the detector itself.
2. **Measured: subtraction makes a much simpler detector possible, and that combination is the win.**
   `onoff_cog`: stress MAE 0.147 mm (oracle 0.108), miss 0.003, false grooves 0.030 per frame (baseline 0.280),
   verdict accuracy 0.980 (baseline 0.915).
3. **Measured (A3): the off frame is needed for exactly one thing: thin bright background features (wet glints).**
   On glints, both single-frame detectors collapse (F1 0.61 / 0.52) and both on/off detectors recover
   (0.92 / 0.97). Sunlight, a light gradient, and a non-saturating paint stripe are *not* a problem for a
   single frame once the detector is the COG (top-hat + COG, ON only, F1 0.98 in all three). So most of the
   baseline -> `onoff_cog` gain on ideal/legal comes from the detector, not from the second frame.
4. **Measured: what subtraction does not fix.** A background that saturates the sensor (bright paint in sun):
   255 - 255 = 0, the laser information is gone, all four variants fail (MAE 2.8-4.2 mm). Rocking: a geometry
   error of the depth stage, identical with and without the off frame (0.220). Dim laser: more columns are
   found but MAE rises (0.207 / 0.334) because subtracting two noisy frames raises the noise by sqrt(2).
   Saturated laser, blur and spikes are handled by the COG with or without the off frame; they hurt Steger
   with or without it. (Spikes: see the simulator caveat below.)
5. **Measured: motion tolerance.** Plain subtraction + COG is unaffected at 0.5 px, starts producing outliers
   at 1 px (MAE 0.747, P95 still 0.476) and is **worse than the single-frame baseline from 2 px** (MAE 3.459
   vs 1.516; false grooves 0.150 vs 0.280 are still fewer). A +-20 % exposure change alone costs 0.147 -> 0.479.
   Registration (phase correlation, translation only) + gain match restores 0.13-0.15 mm up to 2 px and
   under exposure change; it has outlier frames from 3 px (MAE 0.566, P95 unchanged 0.397). The conservative
   2 px dilation needs no registration and holds to 2 px, fails at 5. Registration + a 1 px guard
   (`onoff_robust`) is flat to 3 px (0.158) and 0.341 at 5 px, and costs nothing when there is no motion.
   Plain + Steger degrades slowly and never gets worse than the baseline on MAE, but never gets good either.

## Part B: several presses per tyre position
`presses.py`: 40 scenes per condition x 9 presses (360 renders each). Between presses: new seed
(noise, speckle, spike positions), tilt +-0.3 deg on both axes, x offset +-1 mm, exposure +-10 %. Glints,
paint, debris and stones stay (they belong to the tyre). Each press is measured by the single-frame
baseline and by `onoff_cog` (off frame without motion). `fuse.py` fuses the per-groove depth over N
presses (40 random subsets of the 9 per scene, the same subsets for every rule; N = 9 uses all).
Only grooves whose floor is visible in all 9 presses are scored. `legal_stress` = stress conditions with
0.6-3.5 mm grooves. Verdict = PASS when every fused groove is >= 1.6 mm; with 40 positions one position
is 0.025.

Rules: mean; median; trimmed (drop lowest and highest; equals median at N = 3); median+MAD (drop presses
further than 3 MAD from the median, MAD floored at 0.1 mm, then mean); quality (keep the best two thirds of
presses by surface inlier fraction, then median); consensus (median, but only if at least half the presses
found the groove).

### Table B. Depth MAE mm / P95 mm / false-pass rate versus N
| condition, algorithm | rule | N = 1 | N = 3 | N = 5 | N = 9 |
|---|---|---|---|---|---|
| ideal, on/off COG | mean | 0.068 / 0.224 / 0.000 | 0.066 / 0.234 / 0.000 | 0.065 / 0.232 / 0.000 | 0.065 / 0.222 / 0.000 |
| | median | 0.068 / 0.224 / 0.000 | 0.066 / 0.225 / 0.000 | 0.065 / 0.225 / 0.000 | 0.065 / 0.226 / 0.000 |
| | trimmed | 0.068 / 0.224 / 0.000 | 0.066 / 0.225 / 0.000 | 0.065 / 0.231 / 0.000 | 0.065 / 0.230 / 0.000 |
| | median+MAD | 0.068 / 0.224 / 0.000 | 0.066 / 0.234 / 0.000 | 0.065 / 0.232 / 0.000 | 0.065 / 0.222 / 0.000 |
| | quality | 0.068 / 0.224 / 0.000 | 0.067 / 0.227 / 0.000 | 0.066 / 0.235 / 0.000 | 0.066 / 0.229 / 0.000 |
| stress, on/off COG | mean | 0.224 / 0.442 / 0.025 | 0.207 / 0.518 / 0.025 | 0.188 / 0.572 / 0.025 | 0.186 / 0.581 / 0.025 |
| | median | 0.224 / 0.442 / 0.025 | 0.184 / 0.435 / 0.025 | 0.174 / 0.429 / 0.025 | 0.122 / 0.309 / 0.025 |
| | trimmed | 0.224 / 0.442 / 0.025 | 0.184 / 0.435 / 0.025 | 0.171 / 0.429 / 0.025 | 0.171 / 0.418 / 0.025 |
| | median+MAD | 0.224 / 0.442 / 0.025 | 0.186 / 0.429 / 0.025 | 0.179 / 0.412 / 0.025 | 0.126 / 0.329 / 0.025 |
| | quality | 0.224 / 0.442 / 0.025 | 0.242 / 0.478 / 0.025 | 0.197 / 0.423 / 0.025 | 0.240 / 0.404 / 0.025 |
| legal, on/off COG | mean | 0.023 / 0.053 / 0.000 | 0.021 / 0.048 / 0.000 | 0.021 / 0.048 / 0.000 | 0.020 / 0.047 / 0.000 |
| | median | 0.023 / 0.053 / 0.000 | 0.022 / 0.049 / 0.000 | 0.021 / 0.049 / 0.000 | 0.021 / 0.046 / 0.000 |
| legal_stress, on/off COG | mean | 0.044 / 0.105 / 0.019 | 0.034 / 0.087 / 0.019 | 0.033 / 0.085 / 0.017 | 0.031 / 0.086 / 0.025 |
| | median | 0.044 / 0.105 / 0.019 | 0.035 / 0.089 / 0.016 | 0.033 / 0.083 / 0.013 | 0.030 / 0.081 / 0.025 |
| | median+MAD | 0.044 / 0.105 / 0.019 | 0.034 / 0.087 / 0.019 | 0.033 / 0.085 / 0.017 | 0.031 / 0.086 / 0.025 |
| | quality | 0.044 / 0.105 / 0.019 | 0.037 / 0.090 / 0.023 | 0.034 / 0.085 / 0.021 | 0.034 / 0.085 / 0.025 |
| ideal, single-frame baseline | mean | 0.101 / 0.292 / 0.000 | 0.124 / 0.270 / 0.000 | 0.117 / 0.274 / 0.000 | 0.120 / 0.249 / 0.000 |
| | median | 0.101 / 0.292 / 0.000 | 0.114 / 0.255 / 0.000 | 0.097 / 0.212 / 0.000 | 0.072 / 0.176 / 0.000 |
| | consensus | 0.101 / 0.292 / 0.000 | 0.087 / 0.220 / 0.000 | 0.080 / 0.192 / 0.000 | 0.065 / 0.171 / 0.000 |
| stress, single-frame baseline | mean | 2.587 / 14.356 / 0.028 | 2.555 / 19.970 / 0.036 | 2.595 / 20.105 / 0.029 | 2.610 / 13.010 / 0.025 |
| | median | 2.587 / 14.356 / 0.028 | 2.625 / 13.560 / 0.028 | 2.595 / 13.010 / 0.025 | 2.744 / 13.010 / 0.025 |
| | median+MAD | 2.587 / 14.356 / 0.028 | 2.630 / 13.958 / 0.029 | 2.587 / 13.010 / 0.025 | 2.724 / 13.010 / 0.025 |
| | quality | 2.587 / 14.356 / 0.028 | 2.482 / 13.958 / 0.033 | 2.478 / 13.010 / 0.026 | 2.518 / 13.010 / 0.025 |
| legal, single-frame baseline | mean | 0.053 / 0.285 / 0.000 | 0.049 / 0.225 / 0.000 | 0.046 / 0.167 / 0.000 | 0.046 / 0.136 / 0.000 |
| | median | 0.053 / 0.285 / 0.000 | 0.041 / 0.087 / 0.000 | 0.037 / 0.071 / 0.000 | 0.034 / 0.063 / 0.000 |
| legal_stress, single-frame baseline | mean | 1.831 / 5.212 / 0.058 | 1.761 / 5.154 / 0.092 | 1.752 / 5.136 / 0.098 | 1.737 / 5.099 / 0.100 |
| | median | 1.831 / 5.212 / 0.058 | 1.758 / 5.198 / 0.047 | 1.751 / 5.207 / 0.043 | 1.749 / 5.113 / 0.050 |
| | median+MAD | 1.831 / 5.212 / 0.058 | 1.753 / 5.154 / 0.051 | 1.741 / 5.136 / 0.049 | 1.717 / 5.114 / 0.050 |

All rules, miss rates, median absolute errors and verdict accuracy: `experiments/onoff-fusion/fusion.json`.
Figure: `figs/onoff-fusion/error_vs_presses.png`.

Noise versus bias, per groove over its 9 presses (median over grooves):

| condition, algorithm | repeatable error of the groove (abs, mm) | spread between presses (SD, mm) |
|---|---|---|
| ideal, on/off COG | 0.033 | 0.021 |
| stress, on/off COG | 0.093 | 0.047 |
| legal, on/off COG | 0.020 | 0.014 |
| legal_stress, on/off COG | 0.027 | 0.033 |
| ideal, baseline | 0.046 | 0.032 |
| stress, baseline | 0.339 | 0.381 |
| legal, baseline | 0.026 | 0.020 |
| legal_stress, baseline | 0.057 | 0.325 |

### Findings, part B
1. **Measured: the error is dominated by bias, not by random noise.** With the on/off COG a single press already
   repeats to 0.014-0.047 mm, while the part of the error that is the same on every press is as large or larger.
   Pure noise would take ideal MAE from 0.068 to 0.023 at N = 9; measured 0.065. Legal_stress: 0.044 -> 0.030
   measured, 0.015 predicted. Known repeatable parts: the ~0.07 mm "reads too deep" of the depth stage (README),
   tilt, and whatever the tyre has on it.
2. **Measured: more presses do not rescue the single-frame baseline under stress** (MAE 2.59 -> 2.74, P95 stays
   13 mm, any rule). Its big errors are caused by glints that are there on every press, so they are repeatable.
   Taking the *mean* makes the verdict worse (legal_stress false pass 0.058 -> 0.100): one wild press poisons it.
3. **Measured: use the median, never the mean.** Median, trimmed and median+MAD are within noise of each other;
   the quality rule (surface inlier fraction) is no better than the plain median and sometimes worse, so the
   inlier fraction is a poor per-press score for this purpose. The consensus rule lowers the baseline's error
   by refusing grooves that were found in fewer than half the presses (miss rate 0.12 instead of 0.05-0.08 on ideal).
4. **Measured, small sample: the false-pass rate does not improve with N.** It is 0-1 position out of 40 in
   every on/off cell; this experiment cannot resolve anything below 0.025.
5. Where N helps the on/off pipeline: the outlier tail under stress (median: MAE 0.224 -> 0.184 at N = 3,
   0.122 at N = 9; P95 0.442 -> 0.435 -> 0.309).

## Recommendation
- **Capture the off frame, and change the detector with it.** The off frame is cheap (one more frame) and it
  is the only thing here that removes wet glints; with it a 30-line per-column COG beats the Steger pipeline
  on every simulated set. Do not ship plain subtraction with today's 1-3 s gap: beyond about 1 px of
  apparent motion it is worse than no subtraction. Either take the pair ~70 ms apart, or use
  `onoff_robust` (gain match + phase correlation + 1 px guard), which is flat to 3 px in this test.
  Inference, not measured: at 2.9 px/mm, 1 px is 0.35 mm on the tyre, so a 70 ms pair tolerates roughly
  5 mm/s of hand drift; a 1-3 s pair does not unless the holder is truly locked.
- **Presses: N = 1 is enough for the number; N = 3 with a median is cheap insurance.** One press gives
  0.07 mm (ideal) / 0.02 mm (legal). Three presses with a median trim the stress tail a little and give a
  free repeatability check. N = 5 or 9 buys almost nothing more because the remaining error is bias.
  Technician time (inference): one press is a button press plus 0.1-3 s of capture; three presses with a
  re-seat between them is of the order of 5-10 s per tyre position, nine would be 20-30 s.
- To reduce the error further, work on the bias (depth stage, tilt correction, calibration), not on N.

## Limitations and caveats
- Simulation only; no real on/off pair exists. The first thing to do with the device is to record real
  on/off pairs (dry, wet, sun) at both 70 ms and ~2 s spacing.
- Tables A3 and B use 40 scenes per condition; mean errors there move visibly with a single bad groove.
- `perturb.py` warps the off frame, noise included (interpolation smooths the noise slightly), with a rigid
  shift/rotation. Real hand motion also changes perspective and where glints appear; a glint that moves or
  changes brightness between frames is not removed by any registration. Registration is translation only;
  ECC / rotation-aware registration was not tried (the 0.3 deg rows did not need it).
- The COG returns one point per column. In the simulator that is enough. On the real hand-held photos the
  line doubles back inside a groove (see `pipeline/single_frame.py`), which a one-per-column detector cannot
  follow; this is not tested here.
- Part B keeps glints fixed between presses and re-draws the rubber texture with each seed (a simulator
  property). If real glints move between presses, the median over presses would help the single-frame
  baseline more than shown.
- In part B the verdict uses only fused grooves matched to a true groove; false grooves are ignored there.

## Things in the harness / simulator to know about
1. `bench/run.py` `eval_real` takes `uses_off` but never uses it: for a `USES_OFF = True` module it calls
   `measure(gray, None, None)` and stores the result under the on/off algorithm's name. The docstring says
   such modules are "skipped on real photos' depth". Evidence: the `real` blocks of `results/onoff_*.json`
   are identical to the baseline's (line_f1_ref 1.000, depth_mae_px 0.132, false_measure 0.701).
2. `depth_mae_mm` is a plain mean and is set by a handful of catastrophic grooves. Example (A2, 3 px,
   registered + COG): MAE 0.566 while P95 is 0.397, i.e. fewer than 5 % of grooves carry most of the mean.
   Rankings by MAE between good algorithms are therefore noisy; a median or trimmed error would be steadier.
3. Simulator `spikes` are discs centred exactly on the true line, so they cannot bias a centre of gravity.
   Real specular flashes need not be centred; the COG's good result on spikes may be optimistic.
4. The simulated off frame has the same exposure, no motion and an identical background, which is the best
   case for subtraction (hence the motion sweep). Sunlight in the simulator also never saturates the rubber;
   only paint/glints do.
5. `ms_per_frame` on this shared machine varied by 2-3x between identical runs; not reported here.

## Code excerpts
The detector (`algos/onoff_common.py`, shortened):
```python
def cog_line(d, sigma=1.2, k_noise=6.0, min_dn=12.0, rel=0.25, win=4, jump_px=3.0):
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) + 0.5       # MAD of the difference
    s = cv2.GaussianBlur(d.astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=sigma)
    p = s.argmax(axis=0)                                             # brightest row per column
    peak = s[p, np.arange(w)]
    thr = max(min_dn, k_noise * noise / 2.0, rel * np.percentile(peak, 60))
    for c in np.nonzero(peak > thr)[0]:
        r = np.clip(p[c] + rows, 0, h - 1)                           # +-win rows around the peak
        wgt = np.clip(s[r, c] - 0.5 * peak[c], 0, None)              # only what is above half the peak
        v[c] = (r * wgt).sum() / wgt.sum()
    # then: keep a column only if a neighbour within 2 columns agrees to jump_px per column
```
The robust subtraction (`algos/onoff_common.py`):
```python
off = off * gain_match(on, off)            # exposure ratio from the darker 60 % of the frame
off, _, _ = register(on, off)              # cv2.phaseCorrelate on background-only, high-passed images
off = cv2.dilate(cv2.GaussianBlur(off, (0, 0), 1.0), np.ones((3, 3), np.uint8))   # 1 px guard
d = np.clip(on - off, 0, None)             # a moved background feature can never leave a residue
```

## Figures
- `figs/onoff-fusion/example_on_off_diff.png`: stress frame 182 (sun, 7 glints): on, off, difference, difference with a 2 px moved off frame, robust difference.
- `figs/onoff-fusion/error_vs_motion.png`: table A2 as curves.
- `figs/onoff-fusion/error_vs_presses.png`: table B as curves with the 1/sqrt(N) line.
- `figs/onoff-fusion/factors.png`: table A3.
