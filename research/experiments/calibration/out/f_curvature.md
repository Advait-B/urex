**Bias vs depth, 'ideal', true line (mm)**

| true depth | grooves | default stage, true heights: mean error | formula along (R = 480) | formula across (crown + tilt_y) | sum of formulas | both fixes + phys5: mean error | both fixes: MAE |
|---|---|---|---|---|---|---|---|
| 0-2 mm | 19 | -0.003 | -0.003 | 0.015 | 0.012 | -0.004 | 0.006 |
| 2-4 mm | 39 | 0.008 | -0.004 | 0.025 | 0.021 | -0.006 | 0.006 |
| 4-8 mm | 63 | 0.044 | 0.002 | 0.055 | 0.057 | -0.006 | 0.009 |
| 8-12 mm | 83 | 0.101 | 0.023 | 0.087 | 0.110 | -0.007 | 0.017 |
| 12-15 mm | 44 | 0.139 | 0.054 | 0.101 | 0.155 | -0.019 | 0.029 |
| 15-19 mm | 17 | 0.196 | 0.083 | 0.108 | 0.191 | -0.004 | 0.027 |

**All sets, true line: MAE / bias (mm)**

| set | default stage, true heights | phys5 + curvature fix only | phys5 + mm_space only | phys5 + both | both, if tilt_x were ALSO known (not available) | regression of the default-stage error on (along, across, rocking) |
|---|---|---|---|---|---|---|
| ideal | 0.081 / +0.079 | 0.058 / +0.055 | 0.023 / +0.013 | 0.015 / -0.008 | 0.007 | 1.50, 0.67, 0.99 |
| stress | 0.113 / +0.079 | 0.099 / +0.053 | 0.085 / +0.019 | 0.082 / -0.005 | 0.011 | 1.60, 0.58, 1.01 |
| legal | 0.019 / +0.004 | 0.020 / +0.006 | 0.012 / -0.007 | 0.011 / -0.004 | 0.005 | -2.20, -0.11, 0.96 |

**Tyre radius (80 ideal scenes re-rendered, true line; the fix always assumes R = 480)**

| true R (mm) | default stage: bias, grooves > 12 mm | both fixes: bias, all | both fixes: bias, grooves > 12 mm | formula along @16 mm | left over @16 mm when 480 is assumed |
|---|---|---|---|---|---|
| 450 | 0.144 | -0.004 | -0.008 | 0.087 | 0.005 |
| 480 | 0.140 | -0.005 | -0.012 | 0.082 | 0.000 |
| 520 | 0.136 | -0.007 | -0.016 | 0.075 | -0.006 |
