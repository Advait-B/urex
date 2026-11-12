GRID = []
for det, half in [("cog", 2), ("cog", 3), ("cog", 4), ("gauss3", 1), ("blais4", 4), ("blais8", 6), ("fir", 1), ("fir_log", 1), ("none", 1)]:
    GRID.append((f"{det} half={half}", {"detector": det, "half": half}))
