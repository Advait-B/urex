"""perturb.py - make the simulator's laser-off frame look as if the device moved before it was taken.

In the holder the laser is fixed to the camera, so when the device moves the laser line stays
(almost) where it was in the image while the tyre texture, glints and paint slide underneath.
Warping only the OFF frame is therefore the right first-order model.
"""
import cv2
import numpy as np


def perturb_off(off, shift_px=0.0, rot_deg=0.0, exposure=1.0, rng=None):
    """Shift by shift_px in a random direction, rotate by +-rot_deg about the image centre, scale brightness."""
    rng = rng or np.random.default_rng(0)
    h, w = off.shape
    ang = rng.uniform(0, 2 * np.pi)
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), rot_deg * rng.choice([-1, 1]), 1.0)
    M[:, 2] += shift_px * np.array([np.cos(ang), np.sin(ang)])
    out = cv2.warpAffine(off.astype(np.float32), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(np.round(out * exposure), 0, 255).astype(np.uint8)


def perturbed_items(items, shift_px=0.0, rot_deg=0.0, exposure_pm=0.0, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for it in items:
        e = 1.0 + exposure_pm * rng.choice([-1, 1]) if exposure_pm else 1.0
        out.append({**it, "off": perturb_off(it["off"], shift_px, rot_deg, e, rng)})
    return out
