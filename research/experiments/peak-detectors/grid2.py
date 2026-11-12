GRID = []
for sv in (1.0, 1.5, 2.0, 2.5):
    for su in (0.0, 1.5):
        GRID.append((f"fir sv={sv} su={su}", {"detector": "fir", "sigma_v": sv, "sigma_u": su}))
for det, half in (("cog", 4), ("cog", 5), ("blais8", 6), ("gauss3", 1)):
    for sv in (1.0, 1.5):
        for ru in (0.0, 1.5):
            GRID.append((f"{det} half={half} sv={sv} su=ru={ru}", {"detector": det, "half": half, "sigma_v": sv, "sigma_u": ru, "refine_smooth_u": ru}))
