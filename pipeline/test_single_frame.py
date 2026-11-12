"""test_single_frame.py - does single_frame.py measure the right number of pixels?

Uses synth_tyre.py's laser-on frame (no laser-off frame, like the real photos),
turned so the line runs top-to-bottom and grooves push it LEFT, as in the
30 Oct photos. The true shift in px is known from the synthetic geometry.

    python test_single_frame.py      (exits non-zero on failure)
"""
import sys

import cv2
import numpy as np

import single_frame as sf
import synth_tyre as st


def vertical(B):
    """synth line is horizontal and grooves move it DOWN -> transpose (down becomes right)
    then mirror (right becomes left)"""
    return np.ascontiguousarray(cv2.flip(B.T, 1))


def run():
    ok = True
    print(f"{'case':28s} {'true px':>8s} {'measured':>9s} {'mode':7s} ok")
    # 1. grooves with sloped walls: true shift = shift_px(depth) at the surface (standoff 0)
    for depth, blur, noise in [(4.0, 1.0, 6.0), (8.0, 1.0, 6.0), (8.0, 2.0, 10.0), (15.0, 1.0, 6.0)]:
        _, B, _ = st.make_pair(w=320, h=240, grooves=[(depth, 10.0)] * 2, blur=blur, noise=noise, tilt_px=3,
                               bow_px=2, seed=3, wall_mm=4.0, render="distance")
        g = vertical(B)
        res = sf.analyse(g)
        true = float(st.shift_px(depth))
        got = res["depth_px"]
        # accuracy target: 0.8 px or 3 %, whichever is bigger (the apex of a rounded
        # groove floor, read from a single frame with no laser-off partner)
        good = res["mode"] == "groove" and got is not None and abs(got - true) < max(0.8, 0.03 * true)
        ok &= good
        print(f"{f'groove {depth} mm blur {blur}':28s} {true:8.2f} {got if got is None else round(got, 2)!s:>9s} "
              f"{res['mode']:7s} {'yes' if good else 'NO'}")
    # 2. no groove: must not invent one
    _, B, _ = st.make_pair(w=240, h=320, grooves=[], seed=4)
    res = sf.analyse(vertical(B))
    good = res["mode"] == "none"
    ok &= good
    print(f"{'flat, no groove':28s} {'-':>8s} {str(res['depth_px']):>9s} {res['mode']:7s} {'yes' if good else 'NO'}")
    # 3. no laser at all
    rng = np.random.default_rng(0)
    g = np.clip(cv2.GaussianBlur(rng.normal(90, 25, (240, 320)).astype(np.float32), (0, 0), 3), 0, 255).astype(np.uint8)
    res = sf.analyse(g)
    good = res["depth_px"] is None
    ok &= good
    print(f"{'no laser':28s} {'-':>8s} {str(res['depth_px']):>9s} {res['mode']:7s} {'yes' if good else 'NO'}")
    # 4. design triangulation: a point shifted by shift_px(d) at L0 must come back as d mm.
    #    synth_tyre uses the thin-lens form (L + z - f), single_frame the pinhole form (L + z);
    #    they differ by ~3 %, so allow 5 %.
    for d in (2.0, 8.0, 15.0):
        px = float(st.shift_px(d, pitch_mm=sf.GEOMETRY["pitch_mm_at_640"] * 2))   # pixel is 2x bigger at 320 wide
        back = sf.fixed_scale_mm(px, 320, sf.GEOMETRY["L0_mm"])
        good = abs(back - d) < 0.05 * d
        ok &= good
        print(f"{f'triangulation {d} mm':28s} {px:8.2f} {back:9.3f} {'':7s} {'yes' if good else 'NO'}")
    print("\nALL OK" if ok else "\nSOME CASES FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
