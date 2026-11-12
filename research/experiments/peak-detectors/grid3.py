F = {"detector": "fir", "sigma_v": 1.5, "sigma_u": 1.5}
C = {"detector": "cog", "half": 4}
GRID = []
for bg in ("none", "tophat", "tophat1d", "colpct"):
    GRID.append((f"fir2d bg={bg}", {**F, "bg": bg}))
    GRID.append((f"cog4 bg={bg}", {**C, "bg": bg}))
for mc in (6.0, 12.0, 20.0):
    for rel in (0.15, 0.25, 0.4):
        if (mc, rel) != (12.0, 0.25):
            GRID.append((f"fir2d mc={mc} rel={rel}", {**F, "min_contrast": mc, "rel": rel}))
GRID.append(("fir2d max_per_col=1", {**F, "max_per_col": 1}))
GRID.append(("fir2d max_per_col=2", {**F, "max_per_col": 2}))
