"""report_figs.py - figures for the single-frame analysis: raw vs cleaned data.

frame_figure(gray, res, path)  one photo, four panels:
    1. the photo as the camera saved it
    2. RAW: brightest pixel in every row (what a naive script would measure)
    3. CLEANED: sub-pixel laser points kept by the pipeline, coloured by their
       offset from the fitted tread surface; the surface line; groove apexes
    4. the cleaned depth profile: offset from the surface (px, and mm on the
       right axis with the nominal scale) along the line
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import single_frame as sf  # noqa: E402

NOMINAL_MM_PER_PX_QVGA = None   # filled from the geometry the first time it is needed


def nominal_mm_per_px(width, z_mm=90.0, geom=sf.GEOMETRY):
    """slope of the design triangulation at z_mm: mm of depth per px of shift"""
    pitch = geom["pitch_mm_at_640"] * 640.0 / width
    K = geom["f_mm"] * geom["baseline_mm"] / pitch
    return z_mm ** 2 / K


def frame_figure(gray, res, path, title=""):
    h, w = gray.shape
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
    ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for a in ax[:3]:
        a.imshow(gray, cmap="gray", vmin=0, vmax=255)
        a.set_xticks([]), a.set_yticks([])
    ax[0].set_title("photo")

    rx, ry, rv = sf.raw_profile(gray)
    ax[1].plot(rx, ry, ".", ms=2.5, color="#ff4040")
    ax[1].set_title("RAW: brightest pixel per row")
    ax[1].set_xlim(0, w - 1), ax[1].set_ylim(h - 1, 0)

    x, y, r = res["_x"], res["_y"], res["_r"]
    ax[2].set_title("CLEANED: laser points, surface, apex")
    ax[2].set_xlim(0, w - 1), ax[2].set_ylim(h - 1, 0)
    if len(x):
        lim = max(5.0, float(np.percentile(np.abs(r), 99))) if len(r) else 5
        ax[2].scatter(x, y, c=r, s=3, cmap="coolwarm_r", vmin=-lim, vmax=lim, linewidths=0)
    if res["surface"]:
        yy = np.linspace(0, h - 1, 50)
        ax[2].plot(sf.surface_x(res["surface"]["coef"], yy), yy, "--", color="#40ff40", lw=1)
        groove = res.get("groove")
        if res["mode"] == "steps":
            for st in res["steps"]:
                ax[2].plot([st["x_above"], st["x_below"]], [st["row"], st["row"]], "-", color="yellow", lw=2)
        else:
            for e in res["excursions"]:
                big = e is groove
                ax[2].plot(e["apex_x"], e["apex_y"], "o" if big else "x", mfc="none", mec="yellow",
                           ms=9 if big else 6, mew=1.5)

    # depth profile: offset from the surface vs row, "far" (deeper) plotted downward
    sgn = -1 if sf.GEOMETRY["far_is_left"] else 1
    if len(x):
        o = np.argsort(y)
        ax[3].plot(y[o], sgn * r[o], ".", ms=2.5, color="#3060c0")
    ax[3].axhline(0, color="k", lw=0.8)
    ax[3].set_xlabel("row (along the laser line)")
    ax[3].set_ylabel("offset from surface (px), + = deeper")
    k = nominal_mm_per_px(w)
    sec = ax[3].secondary_yaxis("right", functions=(lambda p: p * k, lambda m: m / k))
    sec.set_ylabel(f"≈ mm (nominal {k:.2f} mm/px)")
    if res["mode"] == "groove":
        ax[3].axhline(res["depth_px"], color="orange", lw=1, ls=":")
        ax[3].set_title(f"groove apex {res['depth_px']:.1f} px ≈ {res['depth_px']*k:.1f} mm")
    elif res["mode"] == "steps":
        for st in res["steps"]:
            ax[3].axvline(st["row"], color="orange", lw=0.8, ls=":")
        ax[3].set_title(f"lug steps: median {res['depth_px']:.1f} px ≈ {res['depth_px']*k:.1f} mm "
                        f"({len(res['steps'])} steps)", fontsize=9)
    else:
        ax[3].set_title("no groove found")
    ax[3].invert_yaxis()
    ax[3].grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)
