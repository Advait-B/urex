"""Figures that belong to the paper itself: geometry, simulator samples, sim vs real."""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import holder_sim as hs  # noqa: E402
from bench import datasets  # noqa: E402

OUT = os.path.join(HERE, "figs")
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.6})
INK, LASER, FOV = "#1b1f24", "#139a43", "#2f6fb5"


def geometry():
    """Side section (Y-Z) of the holder optics on a grooved tyre + the triangulation response."""
    dev = hs.DEVICE
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    y = np.linspace(-45, 75, 400)
    surf = dev["crown_z"] - y ** 2 / (2 * dev["tyre_R"])
    a.fill_between(y, surf, -30, color="#d9dde2", lw=0)
    a.plot(y, surf, color=INK, lw=0.8)
    d = 12.0
    a.fill_between(y, surf - d, surf, where=(y > -30) & (y < 30), color="#f4f5f7", lw=0)   # groove seen side-on
    a.plot(y[(y > -30) & (y < 30)], (surf - d)[(y > -30) & (y < 30)], color=INK, lw=0.6, ls="--")
    E = np.array(dev["laser_exit"][1:])
    a.plot([E[0], 0], [E[1], 0], color=LASER, lw=1.2)
    tz = np.tan(np.radians(dev["theta_deg"]))
    a.plot([0, -d * tz * 1.0], [0, -d + dev["crown_z"]], color=LASER, lw=1.0, ls=(0, (3, 1.5)))
    a.plot(dev["crown_z"] * tz, dev["crown_z"], "o", ms=3.5, color=LASER)
    a.plot((dev["crown_z"] - d) * tz, dev["crown_z"] - d, "o", ms=3.5, mfc="white", mec=LASER)
    C = (0, dev["Zc"])
    for p in ((dev["crown_z"] * tz, dev["crown_z"]), ((dev["crown_z"] - d) * tz, dev["crown_z"] - d)):
        a.plot([C[0], p[0]], [C[1], p[1]], color=FOV, lw=0.6, ls=(0, (5, 2)))
    a.add_patch(plt.Rectangle((-8, dev["Zc"]), 16, 9, fc="#3b4250", ec=INK, lw=0.5))
    a.add_patch(plt.Rectangle((-18, dev["Zc"] + 9), 36, 2, fc="#a9503f", ec=INK, lw=0.5))
    a.text(-10, dev["Zc"] + 13, "camera, 2.7 mm lens", fontsize=7)
    a.add_patch(plt.Polygon([E + np.array([-4.1, 2.9]), E + np.array([4.1, -2.9]), E + np.array([27, 30]), E + np.array([19, 36])],
                            fc="#4a5059", ec=INK, lw=0.5))
    a.text(E[0] - 6, E[1] + 38, "line laser", fontsize=7)
    a.annotate("", (0, 30), (0 + 30 * np.sin(np.radians(35)) * 0, 30))
    a.text(8, 34, "35°", color=LASER, fontsize=8)
    a.annotate("groove floor", ((dev["crown_z"] - d) * tz, dev["crown_z"] - d), (-44, -22), fontsize=7,
               arrowprops=dict(arrowstyle="-", lw=0.4))
    a.annotate("tread surface", (dev["crown_z"] * tz + 20, 2.6), (30, 14), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.4))
    for fy in (-55 + 12, 55):
        pass
    a.plot([55, 55], [0, 100], color="#8a939d", lw=2.5, solid_capstyle="butt")
    a.plot([-42, -42], [0, 100], color="#8a939d", lw=2.5, solid_capstyle="butt")
    a.plot([-45, 75], [100, 100], color="#8a939d", lw=3)
    a.text(57, 45, "printed\nframe", fontsize=6.5, color="#5b6570")
    a.set_xlim(-48, 80), a.set_ylim(-28, 132), a.set_aspect("equal")
    a.set_xlabel("Y, along the tyre (mm)"), a.set_ylabel("Z above the feet plane (mm)")
    a.set_title("(a) Optical section of the holder", fontsize=8)
    b = ax[1]
    dd = np.linspace(0, 18, 100)
    for W, ls in ((640, "-"), (320, "--")):
        f = hs.focal_px(W)
        h0, h1 = dev["crown_z"], dev["crown_z"] - dd
        sh = f * tz * (h0 / (dev["Zc"] - h0) - h1 / (dev["Zc"] - h1))
        b.plot(dd, sh, color=INK, ls=ls, lw=1, label=f"{W} px wide")
    b.axvline(1.6, color="#b3261e", lw=0.7, ls=":")
    b.text(1.9, 52, "legal minimum\n1.6 mm", color="#b3261e", fontsize=6.5)
    b.set_xlabel("groove depth (mm)"), b.set_ylabel("line shift in the image (px)")
    b.set_title("(b) Triangulation response", fontsize=8)
    b.grid(alpha=0.3, lw=0.4), b.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_geometry.pdf"))
    fig.savefig(os.path.join(OUT, "fig_geometry.png"), dpi=200)
    plt.close(fig)


def sim_vs_real():
    ideal, stress, real = datasets.load_sim("ideal"), datasets.load_sim("stress"), datasets.load_real()
    by = {r["name"][6:21]: r for r in real}
    picks_real = ["20261030_154820", "20261030_154510", "20261028_162117", "20261030_154909"]
    cap_real = ["truck tyre, groove", "truck tyre, groove", "rover tyre, lugs", "too close: blurred"]
    fig, ax = plt.subplots(3, 4, figsize=(7.0, 5.6))
    for j, k in enumerate([3, 12, 40, 77]):
        ax[0, j].imshow(ideal[k]["on"], cmap="gray", vmin=0, vmax=255)
        ds = ", ".join(f"{g['depth_mm']:.1f}" for g in ideal[k]["truth"]["grooves"])
        ax[0, j].set_title(f"sim ideal: {ds} mm", fontsize=7)
    for j, k in enumerate([1, 2, 3, 7]):
        ax[1, j].imshow(stress[k]["on"], cmap="gray", vmin=0, vmax=255)
        s = stress[k]["truth"]["scene"]
        tag = [t for t, c in (("glints", s["glints"]), ("paint", s["paint"]), ("gaps", s["gaps"]), ("stone", s["stones"]),
                              ("zig-zag", s["zigzag_mm"])) if c]
        ax[1, j].set_title("sim stress: " + (", ".join(tag) or f"ambient {s['ambient']:.1f}"), fontsize=7)
    for j, (k, c) in enumerate(zip(picks_real, cap_real)):
        ax[2, j].imshow(by[k]["gray"], cmap="gray", vmin=0, vmax=255)
        ax[2, j].set_title(f"real {k[9:]}: {c}", fontsize=7)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, "fig_sim_vs_real.pdf"), dpi=200)
    fig.savefig(os.path.join(OUT, "fig_sim_vs_real.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    geometry()
    sim_vs_real()
    print("ok")
