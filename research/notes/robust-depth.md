# robust-depth: line points -> groove depths -> PASS/FAIL verdict

Branch `research/robust-depth`. Code: `algos/robust_depth.py` (the stage, every step swappable), thin wrappers
`algos/rd_*.py` (baseline line extractor + one configuration), experiments in `experiments/robust-depth/`.
The line extractor is NOT changed: every variant gets either the true line ("true line", as `bench/oracle.py`)
or the points of `algos/baseline_steger.py` ("baseline line").

## Recommended stage (`algos/rd_final.py`)
1. **Coordinates**: convert each point to height (calibration) and to its true lateral position X across the
   tread (pinhole + lens distortion, holder constants). Fit and compare in X, not in image column u.
2. **Surface**: RANSAC quadratic (unchanged from the bench).
3. **Detection**: default rule (> 0.5 mm below the surface) minus narrow-and-shallow dips ("sipes"), plus a
   shallow-groove rule: 9 px running median of the profile, runs > max(0.10 mm, 3.5 x noise) at least 10 px wide.
4. **Floor**: DBSCAN on (u, residual); floor = deepest cluster, depth = its median.
5. **Curvature term** removed (formula below). **Sigma** and a **lower_bound** flag attached to every groove.

## Results (harness numbers, full test sets)
| variant (results/*.json) | set | depth MAE mm | P95 mm | bias mm | miss | false/frame | verdict acc | false PASS |
|---|---|---|---|---|---|---|---|---|
| baseline line + bench default (`baseline_steger`) | ideal | 0.147 | 0.474 | +0.068 | 0.090 | 0.093 | 0.980 | 0.013 |
| baseline line + bench default (`baseline_steger`) | stress | 1.516 | 6.678 | +1.017 | 0.099 | 0.280 | 0.915 | 0.015 |
| baseline line + bench default (`baseline_steger`) | legal | 0.053 | 0.210 | -0.000 | 0.054 | 0.127 | 0.993 | 0.000 |
| baseline line + bench stage + bias correction (`rd_band_corr`) | ideal | 0.136 | 0.738 | +0.029 | 0.097 | 0.093 | 0.973 | 0.013 |
| baseline line + bench stage + bias correction (`rd_band_corr`) | stress | 1.387 | 5.508 | +0.894 | 0.104 | 0.245 | 0.905 | 0.015 |
| baseline line + bench stage + bias correction (`rd_band_corr`) | legal | 0.062 | 0.130 | -0.001 | 0.051 | 0.127 | 0.980 | 0.000 |
| baseline line + DBSCAN floor + correction (`rd_dbscan`) | ideal | 0.087 | 0.173 | -0.027 | 0.097 | 0.093 | 0.973 | 0.013 |
| baseline line + DBSCAN floor + correction (`rd_dbscan`) | stress | 0.539 | 2.292 | -0.015 | 0.104 | 0.245 | 0.905 | 0.015 |
| baseline line + DBSCAN floor + correction (`rd_dbscan`) | legal | 0.027 | 0.062 | -0.022 | 0.051 | 0.127 | 0.987 | 0.000 |
| **baseline line + recommended stage** (`rd_final`) | ideal | 0.081 | 0.157 | -0.019 | 0.082 | 0.100 | 0.987 | 0.000 |
| **baseline line + recommended stage** (`rd_final`) | stress | 0.464 | 2.225 | +0.070 | 0.113 | 0.205 | 0.940 | 0.010 |
| **baseline line + recommended stage** (`rd_final`) | legal | 0.019 | 0.047 | -0.010 | 0.000 | 0.113 | 0.993 | 0.000 |
| baseline line + recommended, lower bounds dropped (`rd_strict`) | ideal | 0.028 | 0.077 | -0.022 | 0.127 | 0.047 | 0.893 | 0.000 |
| baseline line + recommended, lower bounds dropped (`rd_strict`) | stress | 0.192 | 0.524 | +0.057 | 0.215 | 0.060 | 0.805 | 0.010 |
| baseline line + recommended, lower bounds dropped (`rd_strict`) | legal | 0.019 | 0.047 | -0.011 | 0.003 | 0.093 | 0.987 | 0.000 |
| true line + bench default (`oracle_line`) | ideal | 0.071 | 0.205 | +0.068 | 0.011 | 0.000 | 0.987 | 0.013 |
| true line + bench default (`oracle_line`) | stress | 0.108 | 0.344 | +0.069 | 0.006 | 0.005 | 1.000 | 0.000 |
| true line + bench default (`oracle_line`) | legal | 0.018 | 0.038 | +0.002 | 0.044 | 0.000 | 1.000 | 0.000 |
| true line + bench stage + bias correction (`oracle_rd_band_corr`) | ideal | 0.020 | 0.061 | -0.017 | 0.011 | 0.000 | 0.987 | 0.013 |
| true line + bench stage + bias correction (`oracle_rd_band_corr`) | stress | 0.083 | 0.258 | -0.014 | 0.006 | 0.005 | 1.000 | 0.000 |
| true line + bench stage + bias correction (`oracle_rd_band_corr`) | legal | 0.011 | 0.030 | -0.006 | 0.051 | 0.000 | 1.000 | 0.000 |
| true line + DBSCAN floor + correction (`oracle_rd_dbscan`) | ideal | 0.018 | 0.052 | -0.013 | 0.011 | 0.000 | 0.987 | 0.013 |
| true line + DBSCAN floor + correction (`oracle_rd_dbscan`) | stress | 0.082 | 0.255 | -0.009 | 0.006 | 0.005 | 1.000 | 0.000 |
| true line + DBSCAN floor + correction (`oracle_rd_dbscan`) | legal | 0.011 | 0.030 | -0.003 | 0.051 | 0.000 | 1.000 | 0.000 |
| **true line + recommended stage** (`oracle_rd_final`) | ideal | 0.017 | 0.051 | -0.012 | 0.000 | 0.000 | 1.000 | 0.000 |
| **true line + recommended stage** (`oracle_rd_final`) | stress | 0.081 | 0.253 | -0.007 | 0.003 | 0.005 | 1.000 | 0.000 |
| **true line + recommended stage** (`oracle_rd_final`) | legal | 0.010 | 0.029 | -0.002 | 0.000 | 0.000 | 1.000 | 0.000 |
| true line + recommended, lower bounds dropped (`oracle_rd_strict`) | ideal | 0.017 | 0.051 | -0.012 | 0.019 | 0.000 | 0.940 | 0.000 |
| true line + recommended, lower bounds dropped (`oracle_rd_strict`) | stress | 0.081 | 0.256 | -0.007 | 0.027 | 0.000 | 0.920 | 0.000 |
| true line + recommended, lower bounds dropped (`oracle_rd_strict`) | legal | 0.010 | 0.029 | -0.002 | 0.000 | 0.000 | 1.000 | 0.000 |

Real photos (px, `calib=None`: no lateral fit, no correction; 6 truck-tyre references in px: 10.63, 51.38, 48.14, 33.94, 36.4, 28.87):

| variant | depth_mae_px | found_ref | repeat_pair_diff_px | usable_recall | false_measure | per-frame px |
|---|---|---|---|---|---|---|
| `baseline_steger` | 0.132 | 6/6 | 2.45 | 0.947 | 0.701 | 10.69, 51.58, 48.3, 34.04, 36.49, 29.05 |
| `rd_band_corr` | 0.319 | 6/6 | 2.45 | 0.947 | 0.701 | 10.69, 52.62, 48.39, 34.04, 36.49, 29.05 |
| `rd_dbscan` | 1.136 | 6/6 | 0.18 | 0.947 | 0.701 | 8.42, 52.62, 48.39, 36.43, 36.61, 29.28 |
| `rd_final` | 1.188 | 6/6 | 0.17 | 0.947 | 0.627 | 8.21, 52.88, 48.31, 36.44, 36.61, 29.2 |
| `rd_strict` | 12.376 | 4/6 | - | 0.737 | 0.418 | 8.21, -, 1.6, -, 36.61, 29.2 |

`ms_per_frame` in the JSONs was measured with two jobs sharing the machine and is not reliable; the stage
costs roughly as much as the bench stage (RANSAC dominates) plus a few ms.

## 1. Surface model: keep RANSAC
`experiments/robust-depth/surface.py` (DBSCAN floor, correction on, lower-bound grooves excluded; depth MAE mm):

| surface | stress, true line | stress, baseline line | wide_test, true line | wide_test, baseline line |
|---|---|---|---|---|
| RANSAC quadratic | 0.082 | 0.280 | 0.097 | 0.100 |
| RANSAC then Cauchy IRLS (scipy) | 0.082 | 0.275 | 0.098 | 0.098 |
| Theil-Sen (sklearn) | 0.082 | 0.531 | 0.952 | 1.232 |
| Huber (sklearn) | 0.087 | 0.709 | 1.811 | 2.038 |

`wide_test` = 150 extra scenes with 3-4 grooves 13-17 mm wide and +-1.5 deg rocking (my own set, see
`common.py`). Grooves are one-sided outliers, so Huber and Theil-Sen are dragged down as soon as grooves fill
much of the view; RANSAC is not. IRLS refinement changes nothing measurable. Rocking does raise the error for
every model alike (stress, true line: 0.029 mm for rock < 0.5 deg, 0.112 mm for rock > 1 deg), but that is the
tilt term of section 3, not the surface fit.

## 2. Floor estimator: DBSCAN (band fails on real line errors, not on stones)
`experiments/robust-depth/floor.py stress` (RANSAC surface, default detection, correction on; all found grooves):

| floor | true line MAE | true line, stone grooves (n=44) MAE / bias | baseline line MAE | median | P95 | baseline, stone grooves (n=40) MAE / bias |
|---|---|---|---|---|---|---|
| anchored band (bench) | 0.083 | 0.099 / -0.026 | 1.387 | 0.740 | 5.508 | 1.090 / +0.390 |
| median of deepest 30 % | 0.092 | 0.094 / -0.016 | 0.745 | 0.118 | 4.242 | 0.523 / -0.210 |
| KDE mode (deepest tall peak) | 0.081 | 0.084 / -0.010 | 0.654 | 0.119 | 2.355 | 0.823 / +0.122 |
| trapezoid template (soft-L1) | 0.096 | 0.195 / -0.122 | 0.812 | 0.120 | 5.015 | 0.842 / -0.322 |
| change-point (ruptures KernelCPD) | 0.106 | 0.271 / -0.197 | 0.675 | 0.098 | 2.706 | 0.812 / -0.426 |
| GMM (BIC, deepest component) | 0.163 | 0.103 / -0.030 | 0.822 | 0.154 | 5.015 | 0.673 / -0.537 |
| **DBSCAN (deepest cluster)** | 0.082 | 0.089 / -0.011 | 0.539 | 0.098 | 2.292 | 0.597 / -0.197 |

* With a perfect line every estimator except GMM is within 0.02 mm of the others: the floor estimator is not
  what limits the oracle. Stones: band, KDE and DBSCAN still find the true floor (the deepest plateau);
  the trapezoid and change-point fits follow the stone (-0.12 / -0.20 mm bias).
* With the baseline line the anchored band is the worst: it anchors on the 3rd-deepest point, and three stray
  deep points are common. DBSCAN needs a connected cluster of >= 4 points, so strays are ignored.
* Figure: `figs/robust-depth/floor_estimators_stress.png`.
* The remaining tail on the baseline line is not a floor-estimator problem: the baseline extractor **loses the
  floor of deep grooves** and returns only the upper wall (seen directly in train frames 289, 381, 108: no
  points between the two walls). See section 4, "hidden floors".

## 3. The bias: two geometric terms, and the README names only the smaller one
Measured with the true line (ideal, bench floor estimator, `sweep.py ideal 0 correct`):

| correction | MAE mm | bias mm |
|---|---|---|
| none (bench default) | 0.071 | +0.068 |
| curvature term only | 0.050 | +0.047 |
| parallax only (shift u, no lens model) | 0.036 | -0.010 |
| surface fitted in lateral X only | 0.022 | +0.004 |
| lateral X + curvature, R = 384 (-20 %) | 0.024 | -0.023 |
| lateral X + curvature, R = 480 | 0.020 | -0.017 |
| lateral X + curvature, R = 576 (+20 %) | 0.019 | -0.014 |

**Curvature (along the tyre).** The laser plane passes through the origin at 35 deg, so a point at height z
is hit at Y = z tan35. The tread (height z_s) is hit at Y_s = z_s tan35, the floor (z_f) at Y_f = z_f tan35.
The tyre surface is z = c - Y^2 / 2R, so above the floor the tread is lower than where we measured it by
(Y_f^2 - Y_s^2) / 2R. Therefore

    d_true = (z_s - z_f) - tan^2(35) * (z_f^2 - z_s^2) / (2R)

With z_s = 3.2 mm, R = 480: -0.005 mm at d = 3 mm, +0.018 at 10 mm, +0.092 at 17 mm. It is quadratic in depth
and negligible below ~6 mm, so it cannot matter for the 1.6 mm verdict. A +-20 % error in R moves the mean by
+-0.005 mm (table): R does not need to be known well.

**Parallax (across the tyre).** The camera sits above the tread centre. A floor point at depth d appears
closer to the image centre than the tread directly above it (u - uc scales with 1 / (Zc - z)). The bench
subtracts the fitted surface at the floor's own image column, i.e. the tread a little nearer the crown, which
is higher (cross radius 900 mm and any sideways tilt) -> reads too deep, growing with depth and with distance
from the centre. In addition the crown is not a parabola in image columns because of the lens distortion
(k1 = -0.12): the quadratic misfits by +0.02 mm at the centre and -0.05 mm near the edges (measured on
`uneven_tune`). Fitting the surface against true lateral X removes both. This term is ~3x the curvature term.

After both corrections -0.017 mm is left. The calibration polynomial itself reads the tread 0.009 mm low and
floors at -4..-12 mm 0.005 mm high (measured against truth heights), which accounts for about -0.014 mm.

**What cannot be corrected: rocking about the tread axis** (`tilt_x`). It adds tan(tilt) * tan35 * d, i.e.
1.2 % of depth per degree (0.12 mm at 10 mm). On train the residual of the bench stage regresses on this term
with slope 1.08 (r = 0.77). One laser line cannot tell this tilt from a height change; it is the floor of
the stress oracle error (0.081 mm) and the main term in the sigma below.

Figure: `figs/robust-depth/bias_vs_depth.png`.

```python
def lateral_px(u, v, z, width=320, height=240):
    f = F_PX * width / 320.0
    xd, yd = (u - (width - 1) / 2) / f, (v - (height - 1) / 2) / f
    x, y = xd.copy(), yd.copy()
    for _ in range(3):                         # undo the radial distortion
        k = 1 + K1 * (x * x + y * y)
        x, y = xd / k, yd / k
    return 160.0 + 2.9 * x * (ZC_MM - z)       # X in mm, expressed as "ideal pixels"

# in stage():   u = lateral_px(px, v, -q);  coef = RANSAC quadratic of q against u;  r = q - polyval(coef, u)
# per groove:   z_s = -polyval(coef, u_floor);  z_f = z_s - d_raw
#               d = (z_s - z_f) - TAN_LASER**2 * (z_f**2 - z_s**2) / (2 * tyre_R)
```

## 4. Shallow grooves and the verdict
`legal` cannot show the problem: all grooves of a `legal` scene are within +-0.25 mm of each other, so when one
is under 0.5 mm (and missed) the others already fail. The bench default has false PASS 0.000 there. The unsafe
case is uneven wear, and it does occur in `ideal` (frames 22 and 58: a 0.40 / 0.46 mm groove next to 2.4 mm
ones -> PASS; that is the 0.013). I therefore rendered `uneven_tune` / `uneven_test` (300 scenes each, different
seeds, legal-style conditions, each groove worn independently, down to 0.15 mm) and tuned only on `uneven_tune`
and `train`. `experiments/robust-depth/detect.py`:

| set, line | rule | verdict acc | false PASS | false FAIL | miss | false/frame |
|---|---|---|---|---|---|---|
| legal, true | bench default | 1.000 | 0.000 | 0.000 | 0.044 | 0.000 |
| legal, true | recommended | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| legal, baseline | bench default | 0.993 | 0.000 | 0.007 | 0.054 | 0.127 |
| legal, baseline | recommended | 0.993 | 0.000 | 0.007 | 0.000 | 0.113 |
| ideal, baseline | bench default | 0.980 | 0.013 | 0.007 | 0.090 | 0.093 |
| ideal, baseline | recommended | 0.987 | 0.000 | 0.013 | 0.082 | 0.100 |
| stress, baseline | bench default | 0.915 | 0.015 | 0.070 | 0.099 | 0.280 |
| stress, baseline | recommended | 0.940 | 0.010 | 0.050 | 0.113 | 0.205 |
| uneven_test, true | bench default | 0.913 | 0.087 | 0.000 | 0.115 | 0.000 |
| uneven_test, true | recommended | 0.997 | 0.000 | 0.003 | 0.000 | 0.000 |
| uneven_test, baseline | bench default | 0.907 | 0.090 | 0.003 | 0.122 | 0.090 |
| uneven_test, baseline | recommended | 0.997 | 0.000 | 0.003 | 0.010 | 0.107 |

The bench default misses every groove under 0.5 mm; the shape rule finds all grooves >= 0.3 mm and 84 % of
those 0.15-0.3 mm on the baseline line (all of them on the true line). Figure: `figs/robust-depth/legal_boundary.png`.

```python
cols, prof = integer columns, median residual per column               # 1 value per px
sm = median_filter(np.interp(cols, cols[have], prof[have]), size=9)    # a groove is 20-40 px wide: it survives
noise = 1.4826 * MAD(r[inliers within +-lo])
lo = max(P["lo"], lo_k * noise)                                        # 0.10 mm or 3.5 x noise
low = (sm > lo) & have                                                 # and not already a default groove
accept every run of `low` that is >= 10 px wide
# default-rule runs narrower than 10 px AND shallower than 2.5 mm are dropped (sipes / blips)
```
After detection the surface is refitted without the groove columns (a 0.2 mm groove is a RANSAC inlier and
would pull the surface down).

What is left: on `stress` with the baseline line 2 of 200 frames are still false PASS and 5 % are false FAIL.
On train the false FAILs were, in this order: no groove reported at all (line extraction failed; "not PASS"
is arguably the right answer for the product, but the harness counts it as a wrong verdict), small false
grooves, and deep grooves read from a wall fragment. These are line-extraction errors.

The image-prior idea (dark groove band in the image) was **not** tried: the simulator darkens a groove by
0.75 g / (g + 2.5), i.e. 10 % at 0.4 mm, which is weakest exactly where it is needed.

**Hidden floors.** `lower_bound = True` when the floor cluster is under 3 px wide and the reading is over 1 mm
(then the reported depth is the deepest point). Measured with the recommended stage:

| set, line | flagged | truth floor < 3 px among flagged | hidden grooves caught | reading <= truth + 0.1 mm | abs err > 0.5 mm: flagged / not flagged |
|---|---|---|---|---|---|
| ideal, true | 43 / 306 | 0.88 | 38 / 38 | 1.00 | 0.42 / 0.000 |
| stress, true | 68 / 398 | 0.88 | 60 / 64 | 0.96 | 0.57 / 0.006 |
| ideal, baseline | 41 / 275 | 0.71 | 29 / 29 | 0.61 | 0.76 / 0.000 |
| stress, baseline | 71 / 339 | 0.51 | 36 / 42 | 0.46 | 0.89 / 0.071 |

Proposal: report such a groove as ">= x mm", never as a depth. For the verdict a lower bound >= 1.6 mm is
enough for PASS; a lower bound below 1.6 mm means "re-measure". With a clean line the flag really is a lower
bound (96-100 %). With the baseline line it is only an "unreliable" flag: 40-54 % of flagged readings are too
deep, because stray points are involved. `rd_strict` (drop flagged grooves) shows the trade in harness terms:
ideal MAE 0.081 -> 0.028 mm and stress 0.464 -> 0.192 mm, but the harness then scores a frame whose only
grooves were dropped as "not PASS" (verdict 0.893 / 0.805), so the flag has to be used by the verdict logic,
not by deleting grooves.

## 5. Uncertainty
sigma^2 = (floor scatter / sqrt(n_eff))^2 + (surface under the groove, from the fit covariance)^2
+ (tan(0.75 deg) * tan35 * depth)^2, times a scale of 2.0 chosen on `train` with the baseline line.
Coverage of depth +- 1.96 sigma for unflagged grooves with a visible floor (`uncertainty.py`):

| set | true line: 68 % / 95 % interval | baseline line: 68 % / 95 % interval | median sigma mm |
|---|---|---|---|
| train (scale tuned here) | n/a | 95 % by construction | - |
| ideal | 0.996 / 0.996 | 1.000 / 1.000 | 0.15 |
| stress | 0.880 / 0.997 | 0.779 / 0.916 | 0.14 |
| legal | 0.993 / 0.997 | 0.939 / 1.000 | 0.05 |
| uneven_test | 0.981 / 0.993 | 0.949 / 0.998 | 0.05 |

Honest reading: the interval is **conservative, not calibrated**. It over-covers on ideal / legal (2-7x wider
than needed) and under-covers on stress with the baseline line (0.916; 5 % of those grooves are off by more
than 5 sigma: gross line errors that no Gaussian sigma describes). One number cannot be sharp for both a locked
and a rocking holder, because the dominant term (rocking) is not observable in a single frame. A bootstrap
was not tried. Figure: `figs/robust-depth/interval_coverage.png`.

## Real photos
All estimators run with `calib=None` and find 6/6. The references are derived from the baseline's own
band-type estimator, so only agreement can be judged. DBSCAN differs from the reference by 2.2-2.5 px on two
photos (8.2 vs 10.63 px; 36.44 vs 33.94 px); on the hooked real profiles there are few floor points and the
"deepest connected cluster" is not the same set as "within a band of the 3rd-deepest". I cannot say which is
right without a gauge reading. One hint, from a single pair only: the two photos of the same groove
(154820 / 154854) differ by 2.45 px with the bench stage and 0.17 px with DBSCAN. 3 of the 6 reference grooves
are flagged `lower_bound`, which fits the hooked profile. Dropping flagged grooves (`rd_strict`) is wrong on
real photos (4/6 found, 12.4 px MAE because a different, smaller dip is then reported).
`false_measure` 0.701 -> 0.627 comes only from the sipe rule; there is still no quality gate.

## Limitations
* The lateral-X fit and the curvature term use holder constants (Zc, focal length, k1, laser angle, R). On the
  real device these need a calibration of their own (the current plate calibration gives height only).
  Sensitivity to a wrong R is small (table); sensitivity to wrong Zc / k1 was not measured.
* The simulator's tread crown is an exact parabola (cross radius 900 mm). A real tread with ribs at slightly
  different heights will not benefit as much from the lateral fit.
* Shape-rule widths (10 px, 9 px filter) assume grooves >= ~4 mm wide at 2.9 px/mm.
* DBSCAN / shallow thresholds in px are the mm values x ~2.1 and were not tuned on real photos.
* `uneven_*` and `wide_*` are my own simulated sets (cached in `experiments/robust-depth/_cache/`, regenerated
  by `common.py`); they are not part of the shared benchmark.
* Needs `pip install ruptures` only for the change-point estimator (not used by the recommended stage).
* Leakage: parameters were set on `train` and `uneven_tune` / `wide_tune`. Estimator ranking was first seen on
  `train`; the test sets were used for reporting only.

## Reproduce (from research/)
    python -m bench.run algos.rd_final --out results/rd_final.json
    python experiments/robust-depth/oracle_eval.py
    python experiments/robust-depth/floor.py stress ; surface.py stress ; detect.py uneven_test 0 3.5
    python experiments/robust-depth/sweep.py ideal 0 correct ; uncertainty.py ideal,stress,legal,uneven_test ; figures.py
