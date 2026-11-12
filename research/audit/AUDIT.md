# AUDIT - can the research/ numbers go into the paper?

Independent review of commit `9e9f4fd` (detached, clean checkout), 12 Nov 2026. Everything below was
run or read by the auditor; nothing in `bench/`, `algos/`, `results*/` was changed. Scratch scripts and
re-run outputs are outside the repo (`/tmp/claude-0/audit_scratch/`, `/tmp/claude-0/audit_out/`).

**Short verdict.** The numbers are reproducible and the simulated test sets are clean (no leakage found).
The problems are in *what the numbers mean*: three scorer quirks, very wide error bars on every MAE,
tiny counts behind the safety metric, and real-photo "accuracy" that is not accuracy at all.

## 1. Reproducibility - PASS
Re-ran `python -m bench.run2 algos.<m>` for 9 modules and compared every non-underscore metric except
`cpu_ms` (and also the `_true`/`_pred`/`_per_frame` lists) with `results2/<m>.json`.

| module | sets re-run | metrics compared | differences > 1e-6 |
|---|---|---|---|
| baseline_steger, dp_dense, pk_fir2d_rows, rd_final, cand_clf, gated_baseline, onoff_cog, e2e_cnn | all 6 | 89 each | **0** |
| cnn_line | stress, real | 25 | **0** |

Reproduced = stored, for the headline rows (depth MAE / median, mm; handheld in px):

| algorithm | ideal | stress | legal | shifted | handheld px | stress coverage |
|---|---|---|---|---|---|---|
| baseline_steger | 0.147 / 0.054 | 1.516 / 0.535 | 0.053 / 0.030 | 1.597 / 0.138 | 8.115 / 2.806 | 1.00 |
| dp_dense | 0.064 / 0.041 | 0.544 / 0.086 | 0.032 / 0.026 | 0.501 / 0.081 | 3.661 / 0.405 | 1.00 |
| pk_fir2d_rows | 0.060 / 0.039 | 1.089 / 0.089 | 0.029 / 0.025 | 1.059 / 0.075 | 2.631 / 0.507 | 1.00 |
| rd_final | 0.081 / 0.022 | 0.464 / 0.093 | 0.019 / 0.016 | 0.479 / 0.056 | 7.057 / 0.608 | 1.00 |
| cand_clf (JSON name `cand_hgb`) | 0.154 / 0.079 | 0.273 / 0.088 | 0.044 / 0.028 | 0.466 / 0.093 | 1.872 / 0.454 | 1.00 |
| gated_baseline | 0.107 / 0.048 | 0.136 / 0.069 | 0.053 / 0.030 | 0.117 / 0.067 | 3.768 / 1.961 | **0.16** |
| onoff_cog | 0.062 / 0.037 | 0.147 / 0.080 | 0.026 / 0.020 | 0.182 / 0.073 | 1.106 / 0.555 | 1.00 |
| e2e_cnn | 0.100 / 0.074 | 0.483 / 0.247 | 0.072 / 0.062 | 5.216 / 3.231 | none (miss 1.0) | 1.00 |

## 2. Data integrity - PASS
- Regenerated the first 12 frames of `ideal`, `stress`, `legal`, `train` and the first 10 of `handheld`
  from their seeds: `on`, `off` and every truth field are bit-identical to `/home/claude/urex_data`.
- Seeds: ideal 101, stress 202, legal 303, train 404, shifted 505, handheld 606, calib 909
  (`bench/datasets.py:28-29,55`, `bench/datasets2.py:91,104`). Across the six sets: 0 shared scene
  dicts, 0 shared images, 0 shared per-scene render seeds, 0 shared groove geometries.
- Not checked: `shifted` was not regenerated (needs 45 calibration renders); only its scenes were compared.

## 3. Leakage - no test-set leakage found; one in-sample real-photo number
| component | trained on | tuned on | verdict |
|---|---|---|---|
| `cand_hgb.pkl`, `cand_lr.json`, `cand_thresh.json` | train 0-899 (`experiments/pixel-classifier/train.py:15`) | train 900-1199 (`tune.py:9`) | clean. `hyst.py` printed real-photo scores as a diagnostic; the notes say nothing was selected on them. |
| `cnn_line.pt` | train minus last 200 (`cnn-segmentation/train.py:150`) | last 200 of train (`evaluate.py:31`) | clean. One seed, 1000 iterations. |
| `e2e_cnn.pt`, `e2e_gbm.joblib` | train 0-999 (`end-to-end/train_cnn.py:58`, `train_gbm.py:24`) | train 1000-1199 | clean. |
| `risk_lr.json` (gate stage 2) | train, threshold on 5-fold out-of-fold train (`quality-gate/part_b.py:86-99`) | - | clean. ideal/stress/legal are only scored. |
| `frame_lr.json` (gate stage 1) | **all 86 real photos** (`quality-gate/fit_frame.py:13-17`) | C = 0.03 picked on leave-one-session-out in October | **in-sample on `real`** (see P3). |
| dp_dense, pk_*, rd_*, onoff_* (no model) | - | slices of train / own extra sets (per notes and scripts) | clean. onoff notes declare one `--limit 60` smoke run on the test sets, no change afterwards. |

`shifted` and `handheld` were created after the first round, so nothing could have been tuned on them.
Limits of this check: I read the scripts; I cannot prove nobody looked at a test table and then changed a constant by hand.

## 4. Scorer sanity (v2) - works on the basics, three real problems
Synthetic algorithms fed to `eval_sim2` (ideal / stress / legal / handheld):

| fake algorithm | result | as it should be? |
|---|---|---|
| oracle (true grooves) | MAE 0, miss 0, false 0, verdict 1.0, false_pass 0 | yes |
| no grooves, answers | miss 1.0, false_pass 0, MAE None, **verdict_acc 0.067 / 0.085 / 0.427** | yes, but note: "no groove" counts as a correct FAIL |
| always abstains | coverage 0, false_pass 0, **miss_rate 0.0, verdict_acc 0.0** | misleading: these should be "None" (`run2.py:112-113`, division guarded by `max(1, ..)`) |
| oracle + 3 junk slivers (20 mm) on each groove | MAE 0, false 0, verdict 1.0, split 5.0/frame | junk next to a real groove is free unless it is < 1.6 mm |
| only a sliver just OUTSIDE each groove | miss 0, false 0 (MAE large) | the +-4 px pad counts it as "found" (`run2.py:44`) |
| oracle that abstains when any groove < 3 mm | coverage 0.85 / 0.80 / 0.17, everything else perfect | abstaining is unpunished except in `coverage` |

`false_pass_rate` is over all frames with a groove in view, including abstained ones (`run2.py:73-75`): confirmed.

## Confirmed problems, most serious first
**P1. e2e_cnn / e2e_gbm real-photo `false_measure = 0.0` is a scoring artefact.** They return grooves with
`"unit": "mm"` whatever the input (`algos/e2e_cnn.py:80,87`). `eval_real2` keeps only px grooves
(`run2.py:127`) so "reports" is always false (`:129`), while `usable_recall` reads the algorithm's own
flag (`:128`). Run directly, e2e_cnn outputs a depth on 17/19 usable, **9/12 laser-but-unusable and 34/55
no-laser photos**: its true false-measure rates are 0.75 and 0.62, not 0.0 and 0.0. Same filter on
`handheld` (`run2.py:69`) gives miss 1.0 with coverage 0.987: the model was not evaluated there at all.

**P2. Stress/shifted MAE is set by a handful of grooves and its error bars are wide.** Bootstrap over
frames (1000 resamples, real frame identity, `stress`, 200 frames):

| | MAE | 95 % interval | median | 95 % interval | worst 5 % of grooves carry | MAE without them |
|---|---|---|---|---|---|---|
| baseline_steger (302 grooves) | 1.516 | 1.13 - 1.93 | 0.535 | 0.32 - 0.67 | 48 % of all error | 0.825 |
| dp_dense (329) | 0.544 | 0.25 - 0.94 | 0.086 | 0.07 - 0.11 | **78 %** (7 grooves > 5 mm) | 0.123 |
| rd_final (297) | 0.464 | 0.34 - 0.63 | 0.093 | 0.07 - 0.11 | 53 % | 0.229 |

Paired differences: baseline - dp_dense MAE 0.95 [0.44, 1.45]; baseline - rd_final 1.03 [0.68, 1.45]:
both real. **dp_dense - rd_final 0.08 [-0.24, +0.48], median -0.004 [-0.02, +0.02]: no difference.**
On `shifted` (per-groove bootstrap) dp_dense is 0.50 [0.16, 1.09] and 80 % of its error is 12 grooves.
Also: each MAE is over the grooves that algorithm *found* (297-329 of 335), so the rows are not the same grooves.

**P3. gated_baseline's real-photo row is in-sample.** `usable_recall 0.895`, `false_measure 0.036 / 0.25`
in `results2/gated_baseline.json` score the frame gate on the 86 photos it was fitted on
(`fit_frame.py`; the JSON itself carries no warning). `notes/quality-gate.md:44-63` says so plainly and
gives cross-validated rows (17/19, 5/67 LOO; 4/67 leave-one-session-out). My own refit of the classifier
alone gave 18/19 with 7/67 (LOO) and 6/67 (LOSO) false accepts; the count moves between 2 and 12 as C
changes. All 19 usable photos come from two sessions (28 and 30 Oct), so "leave one session out" has two
informative folds.

**P4. The safety metric rests on very few frames.** Tyres below 1.6 mm: 10 of 150 in `ideal`, 17 of 200
in `stress`, 64 of 150 in `legal`. `false_pass_rate` divides by ALL frames: baseline stress 0.015 = 3
frames = 3 of the 17 illegal tyres passed. "0.000" means "0 of 17". An algorithm that always says PASS
scores verdict_acc 0.933 / 0.915 on ideal / stress - the baseline's stress verdict_acc is 0.915.
Only `legal` can support a verdict claim.

**P5. Abstaining algorithms are not comparable row-for-row.** gated_baseline answers 16 % of stress, 52 %
of shifted, 34 % of handheld frames; its MAE 0.136 is over 50 grooves, its verdict_acc over 32 frames.

**P6. Real-photo depth numbers cannot rank algorithms** (section 6).

**P7. On/off results assume a perfectly still device.** The simulated `off` frame is the same scene with
no motion; on real photos the family is the baseline (`algos/onoff_cog.py:14-15`; verified: identical
grooves on all 86 photos, identical `real` row). There is no real evidence for on/off.

**P8. Minor.** (a) empty-denominator metrics print 0.0 instead of None (section 4). (b) `shifted`
calibrations were fitted with the baseline extractor (`datasets2.py:97`). (c) e2e_cnn ignores the
calibration it is given, hence 5.2 mm on `shifted` - a true result, but say why. (d)
`results2/cand_clf.json` has `"name": "cand_hgb"`. (e) the run2 docstring says abstaining "is not counted
as a wrong verdict" - true, and it is also not counted as a miss.

## 6. What the real photos can and cannot show
86 photos, 19 usable by eye, 12 unusable with laser, 55 without. Six truck-tyre reference grooves
(depths 10.6-51.4 px), produced by running the baseline pipeline itself (`bench/make_real_refs.py:22`)
and accepted by eye. No ruler or gauge measurement exists.
- `depth_mae_px`, `ok_ref`: agreement with the baseline, not accuracy. Baseline 0.13 px / 6/6 is true by
  construction. "5/6" vs "6/6" is one photo. Cannot say any algorithm is more or less accurate on real tyres.
- `repeat_pair_diff_px`: one pair of photos. Anecdote.
- `usable_recall`: 19 frames, one frame = 5.3 points; 17/19 has a 95 % interval of roughly 0.67-0.99.
  Labels are one person's judgement. Differences of one or two frames between algorithms mean nothing.
- `false_measure_laser`: 12 frames (one frame = 8 points). `false_measure_nolaser`: 55 frames; this is
  the only real-photo number with some weight, e.g. baseline 38/55 vs gated 2/55 (in-sample, P3).
- mm on real photos: not available (laser was not fixed to the camera).

## Claims that are SAFE to print
1. All scorer-v2 results reproduce exactly from a clean checkout; test sets regenerate bit-for-bit from seeds.
2. No learned model was trained or tuned on ideal / stress / legal / shifted / handheld (code-level check).
3. In simulation, on the 200-frame stress set, dp_dense and rd_final cut the baseline's depth MAE from
   1.52 mm to about 0.5 mm (paired 95 % intervals exclude zero) and its median error from 0.53 to 0.09 mm.
4. In simulation, median depth error of the better methods is 0.02-0.09 mm on ideal / legal / stress.
5. Without a quality gate the baseline reports a depth on 38 of 55 real photos that contain no laser line.
6. A simulator-trained candidate classifier (cand_clf) does not transfer: 2/6 reference grooves found,
   usable recall 0.58 on real photos.
7. A calibration-free end-to-end CNN fails on other holder units (5.2 mm MAE on `shifted`).

## Claims that need a caveat (suggested wording)
- Any stress / shifted MAE: "mean error is dominated by a few gross failures (the worst 5 % of grooves
  carry 50-80 % of it); we therefore also report the median and the share of errors above 1 mm. 95 %
  bootstrap intervals over frames are given in Table X."
- Quality gate on real photos: "the frame gate was fitted on the same 86 photos; cross-validated, it keeps
  17 of 19 usable photos and wrongly accepts 4-7 of 67 unusable ones. With 19 positives from two sessions
  this is a feasibility result, not a performance estimate."
- gated_baseline on sim: "at 16 % coverage on stress (52 % shifted); errors are over answered frames only."
- false_pass: give counts - "x of the 17 (stress) / 64 (legal) tyres below 1.6 mm were passed."
- On/off subtraction: "simulation only, with a perfectly registered laser-off frame; not tested on hardware."
- handheld set: "simulated hand-held frames, scored in pixels"; never call it real data.
- Real `depth_mae_px` / `ok_ref`: "agreement with the baseline pipeline on six reference grooves".
- cnn_line: "one training run, one seed".

## Claims that should NOT be made
- "dp_dense is better than rd_final" (or the reverse), or any ranking among dp_dense / rd_final /
  cand_clf / cnn_line / pk_fir2d_rows based on stress or shifted MAE. The intervals overlap.
- "e2e_cnn never measures an unusable photo" (false_measure 0.0) - a scoring artefact (P1). Do not print
  the e2e `real` or `handheld` rows at all until the unit handling is fixed.
- Any accuracy in mm, or "X is as accurate as / more accurate than the baseline", on real tyres.
- "Verdict accuracy 91-99 %" from ideal or stress as evidence of legal-limit performance (P4).
- "Zero unsafe passes" as a safety claim: it is 0 of 10-17 tyres on ideal / stress.
- gated_baseline's `results2` real row as a measure of how the gate behaves on new photos.
- Treating the on/off family's real-photo row as a result (it is the baseline).

## Suggested fixes before the paper (not done here; `bench/` is frozen)
1. `eval_real2`: decide "reports" from `usable` and any groove regardless of unit; mark mm-only
   algorithms "not applicable" on `handheld` instead of miss 1.0.
2. Return None for miss_rate / verdict_acc when nothing was answered; report false passes as counts
   over illegal tyres.
3. Store per-frame errors in the JSON so intervals can be computed without re-running.
4. Score the gate on real photos with the cross-validated predictions, or mark the row "in-sample".
5. Get gauge-measured depths for a few tyres: without them there is no real accuracy number.
