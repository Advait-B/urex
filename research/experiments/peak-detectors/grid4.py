F = {"detector": "fir", "sigma_v": 1.5, "sigma_u": 1.5}
GRID = [("fir2d", F)]
for fl in (0.25, 0.5, 1.0):
    for ra in (1.0, 2.0):
        GRID.append((f"fir2d rows flank={fl} ratio={ra}", {**F, "rows": True, "rows_flank": fl, "rows_ratio": ra}))
GRID.append(("cog4 rows", {"detector": "cog", "half": 4, "rows": True}))
