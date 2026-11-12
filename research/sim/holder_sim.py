"""holder_sim.py - simulate what the camera sees with the 3-D printed holder on a tyre.

Geometry is the Rev A holder (urex-holder/cad, project doc "printed holder"):
  * device frame in mm: Z up from the plane of the three feet, camera at (0, 0, 85)
    looking straight down, laser exit at (0, 63.1, 90.1) aimed at the origin (35 deg),
    110 deg fan spread along X (across the tread).
  * lens 2.7 mm on OV5640 (3.67 mm wide sensor): 470.8 px focal length at 640 px wide.
  * tyre: cylinder of radius 480 mm, axis along X; crown ~3.2 mm above the feet plane;
    circumferential grooves (run along Y), so the laser line crosses them.

How an image is made (all vectorised numpy, ~30 ms per 320x240 frame):
  1. height field h(X, Y): crown + tilt + grooves (trapezoid, sloped walls) + stones.
  2. laser: ~2400 rays across the fan are marched to their FIRST hit on h (so groove
     walls shadow the floor exactly as they would for real).
  3. camera: each hit is kept only if the straight path to the lens clears h
     (the wall hides part of the floor), then projected through a pinhole with
     radial distortion and splatted as a power packet.
  4. the line image is blurred (laser line width + defocus), multiplied by speckle,
     added to an ambient image (rubber texture, dark grooves, light gradient, glints),
     then exposure, shot + read noise, 8-bit clipping and a JPEG q95 round trip.

Image convention (the one the whole research/ folder uses):
  the laser line runs LEFT-RIGHT, and a deeper surface moves it DOWN (larger v).

Returned truth: the exact sub-pixel line (u, v, height_mm), per-groove depth and image
span, a line mask, and the laser-off frame of the same scene.

    python holder_sim.py out_dir --mode ideal --n 4     (writes PNGs to look at)
"""
import argparse
import json
import os

import cv2
import numpy as np

DEVICE = {
    "Zc": 85.0,                     # lens front above the feet plane (mm)
    "f_mm": 2.7, "sensor_w_mm": 3.67,
    "k1": -0.12,                    # barrel distortion of the wide lens (normalised coords)
    "laser_exit": (0.0, 63.1, 90.1),
    "theta_deg": 35.0,              # laser angle to the camera axis
    "laser_roll_deg": 0.0,          # line not parallel to the deck's short edge
    "fan_half_deg": 42.0,           # part of the 110 deg fan that can reach the field of view
    "tyre_R": 480.0, "crown_z": 3.2, "cross_R": 900.0,
}


def focal_px(width, dev=DEVICE):
    return dev["f_mm"] / dev["sensor_w_mm"] * width


# ------------------------------------------------------------------ scene
def sample_scene(rng, mode="ideal"):
    """Random scene parameters. 'ideal' = holder locked on a clean tyre in shade.
    'stress' = everything that can go wrong in a depot: rocking, sunlight, wet glints,
    stones, blur, dim or saturated laser. 'train' = a mix of both for ML training."""
    if mode == "train":
        mode = "stress" if rng.random() < 0.6 else "ideal"
    stress = mode == "stress"
    n_g = int(rng.choice([1, 2, 2, 2, 3]))
    while True:                                   # groove centres inside the view, >= 22 mm apart
        cx = np.sort(rng.uniform(-44, 44, n_g))
        if n_g == 1 or np.min(np.diff(cx)) >= 22:
            break
    base_d = rng.uniform(0.6, 17.0)               # a tyre wears fairly evenly across its grooves
    grooves = [{"x": float(c), "w": float(rng.uniform(8, 14)),
                "d": float(np.clip(base_d + rng.normal(0, 0.8), 0.4, 18.0)),
                "wall_deg": float(rng.uniform(8, 22))} for c in cx]
    s = {
        "mode": mode, "grooves": grooves,
        "tilt_x_deg": float(rng.normal(0, 0.15)), "tilt_y_deg": float(rng.normal(0, 0.15)),
        "dz": 0.0, "yaw_deg": float(rng.normal(0, 0.5)), "zigzag_mm": 0.0, "zigzag_period": 60.0,
        "stones": [],
        "ambient": float(rng.uniform(0.05, 0.25)), "ambient_grad": (0.0, 0.0),
        "rubber_dn": float(rng.uniform(45, 80)),
        "blur_px": float(rng.uniform(0.6, 1.0)), "line_w_mm": float(rng.uniform(0.4, 0.7)),
        "line_peak_dn": float(rng.uniform(150, 215)), "speckle": float(rng.uniform(0.12, 0.28)),
        "glints": [], "spikes": 0, "gaps": [], "paint": None,
        "read_noise": 1.5, "seed": int(rng.integers(1 << 30)),
    }
    if stress:
        s.update({
            "tilt_x_deg": float(rng.normal(0, 1.0)), "tilt_y_deg": float(rng.normal(0, 1.0)),
            "dz": float(abs(rng.normal(0, 0.6))), "yaw_deg": float(rng.normal(0, 3.0)),
            "zigzag_mm": float(rng.choice([0, 0, 2, 4])),
            "ambient": float(rng.uniform(0.1, 1.0)),
            "ambient_grad": (float(rng.normal(0, 0.5)), float(rng.normal(0, 0.5))),
            "blur_px": float(rng.uniform(0.6, 2.4)), "line_w_mm": float(rng.uniform(0.4, 1.2)),
            "line_peak_dn": float(np.exp(rng.uniform(np.log(40), np.log(600)))),
            "speckle": float(rng.uniform(0.15, 0.45)),
            "spikes": int(rng.integers(0, 6)), "read_noise": float(rng.uniform(1.5, 4.0)),
        })
        s["glints"] = [{"u": float(rng.uniform(0, 1)), "v": float(rng.uniform(0, 1)),
                        "len": float(rng.uniform(3, 25)), "wid": float(rng.uniform(0.8, 2.5)),
                        "ang": float(rng.uniform(0, np.pi)), "amp": float(rng.uniform(60, 300))}
                       for _ in range(int(rng.integers(0, 8)))]
        if rng.random() < 0.3:                    # a stone lodged in a groove, on the laser line
            g = grooves[int(rng.integers(len(grooves)))]
            s["stones"].append({"x": g["x"] + float(rng.uniform(-2, 2)), "y": float(rng.uniform(-3, 3)),
                                "r": float(rng.uniform(2, 4)), "h": float(rng.uniform(0.3, 0.9)) * g["d"]})
        if rng.random() < 0.25:                   # mud / debris hiding a stretch of the line
            s["gaps"] = [(float(a), float(a + rng.uniform(0.03, 0.10))) for a in rng.uniform(0.05, 0.85, 2)]
        if rng.random() < 0.15:                   # white paint mark: a wide bright stripe
            s["paint"] = {"v": float(rng.uniform(0.1, 0.9)), "wid": float(rng.uniform(6, 14)),
                          "amp": float(rng.uniform(60, 160))}
    return s


def flat_scene(height_mm, rng):
    """Calibration target: a flat matt plate at Z = height_mm (negative = further away)."""
    s = sample_scene(rng, "ideal")
    s.update({"mode": "calib", "grooves": [], "flat_z": float(height_mm), "tilt_x_deg": 0.0, "tilt_y_deg": 0.0,
              "yaw_deg": 0.0, "ambient": 0.08, "rubber_dn": 90.0, "speckle": 0.12})
    return s


# ----------------------------------------------------------- height field
def groove_depth(X, Y, s):
    """Depth below the tread surface (mm, >= 0) at device point (X, Y)."""
    g = np.zeros(np.broadcast(X, Y).shape)
    if not s["grooves"]:
        return g
    yaw = np.radians(s["yaw_deg"])
    xg = X * np.cos(yaw) + Y * np.sin(yaw)
    if s["zigzag_mm"]:
        xg = xg + s["zigzag_mm"] * np.abs(((Y / s["zigzag_period"]) % 1.0) * 4 - 2) - s["zigzag_mm"]
    for gr in s["grooves"]:
        t = gr["w"] / 2 - np.abs(xg - gr["x"])
        if gr.get("shape") == "round":             # U-shaped groove: rounded floor, few deepest points
            g = np.maximum(g, gr["d"] * np.sqrt(np.clip(1 - (1 - np.clip(t / (gr["w"] / 2), 0, 1)) ** 2, 0, 1)))
        else:
            g = np.maximum(g, np.clip(t / np.tan(np.radians(gr["wall_deg"])), 0, gr["d"]))
    for st in s["stones"]:
        bump = st["h"] * np.exp(-((X - st["x"]) ** 2 + (Y - st["y"]) ** 2) / (2 * (st["r"] / 2) ** 2))
        g = np.maximum(g - bump, 0)
    return g


def surface_z(X, Y, s, dev=DEVICE):
    """Tread surface height without grooves."""
    if "flat_z" in s:
        return np.full(np.broadcast(X, Y).shape, s["flat_z"], float)
    return (dev["crown_z"] - Y ** 2 / (2 * dev["tyre_R"]) - X ** 2 / (2 * dev["cross_R"])
            + np.tan(np.radians(s["tilt_y_deg"])) * X + np.tan(np.radians(s["tilt_x_deg"])) * Y - s["dz"])


def height(X, Y, s, dev=DEVICE):
    return surface_z(X, Y, s, dev) - groove_depth(X, Y, s)


# ------------------------------------------------------------ projection
def project(P, width, height_px, dev=DEVICE):
    """Device points (..., 3) -> pixel (u, v) with radial distortion. Deeper = larger v."""
    f = focal_px(width, dev)
    zc = dev["Zc"] - P[..., 2]
    x, y = P[..., 0] / zc, -P[..., 1] / zc
    k = 1 + dev["k1"] * (x * x + y * y)
    return (width - 1) / 2 + f * x * k, (height_px - 1) / 2 + f * y * k


def pixel_to_plane(width, height_px, z_plane, dev=DEVICE):
    """Device (X, Y) seen by every pixel on the plane Z = z_plane (inverse of project)."""
    f = focal_px(width, dev)
    v, u = np.mgrid[0:height_px, 0:width].astype(float)
    xd, yd = (u - (width - 1) / 2) / f, (v - (height_px - 1) / 2) / f
    x, y = xd.copy(), yd.copy()
    for _ in range(3):                             # undo the distortion by fixed-point iteration
        k = 1 + dev["k1"] * (x * x + y * y)
        x, y = xd / k, yd / k
    zc = dev["Zc"] - z_plane
    return x * zc, -y * zc


# ----------------------------------------------------------------- laser
def trace_laser(s, dev=DEVICE, n_rays=2400, n_steps=320):
    """March every fan ray to its first hit. Returns hits P (n, 3), fan angle, visible mask."""
    E = np.array(dev["laser_exit"], float)
    th = np.radians(dev["theta_deg"])
    b = np.array([0.0, -np.sin(th), -np.cos(th)])                 # beam centre direction
    roll = np.radians(dev["laser_roll_deg"])
    xr = np.array([np.cos(roll), np.sin(roll) * np.cos(th), -np.sin(roll) * np.sin(th)])  # fan axis
    phi = np.radians(np.linspace(-dev["fan_half_deg"], dev["fan_half_deg"], n_rays))
    d = np.cos(phi)[:, None] * b + np.sin(phi)[:, None] * xr       # (n, 3) unit directions
    # distance along the ray: from a plane 12 mm above the feet to 26 mm below
    t0 = (E[2] - 12.0) / -d[:, 2]
    t1 = (E[2] + 26.0) / -d[:, 2]
    tt = t0[:, None] + (t1 - t0)[:, None] * np.linspace(0, 1, n_steps)[None, :]
    P = E + tt[..., None] * d[:, None, :]
    gap = P[..., 2] - height(P[..., 0], P[..., 1], s, dev)         # > 0 above the surface
    below = gap <= 0
    first = below.argmax(axis=1)
    hit = below.any(axis=1) & (first > 0)
    i = np.arange(n_rays)
    g0, g1 = gap[i, np.maximum(first - 1, 0)], gap[i, first]
    frac = np.where(hit, g0 / np.where(g0 - g1 == 0, 1, g0 - g1), 0)
    th_hit = tt[i, np.maximum(first - 1, 0)] + frac * (tt[i, first] - tt[i, np.maximum(first - 1, 0)])
    H = E + th_hit[:, None] * d
    # camera visibility: the segment hit -> lens must stay above the surface
    C = np.array([0.0, 0.0, dev["Zc"]])
    ss = np.linspace(0.004, 0.35, 90)[None, :, None]               # only the part near the tyre matters
    Q = H[:, None, :] + ss * (C - H)[:, None, :]
    vis = hit & np.all(Q[..., 2] - height(Q[..., 0], Q[..., 1], s, dev) > -1e-3, axis=1)
    return H, phi, vis, d


# ---------------------------------------------------------------- render
def _smooth_noise(rng, shape, sigma):
    n = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), sigma)
    return n / (n.std() + 1e-9)


def render(s, width=320, height_px=240, dev=DEVICE, jpeg_quality=95, return_off=True):
    """Returns (laser_on uint8, laser_off uint8 or None, truth dict)."""
    rng = np.random.default_rng(s["seed"])
    f = focal_px(width, dev)
    H, phi, vis, d = trace_laser(s, dev)
    u, v = project(H, width, height_px, dev)
    inside = vis & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height_px - 1)

    # --- power packets: fan profile (flat-top with soft shoulders), view cosine, range
    eps = 0.05
    hx = (height(H[:, 0] + eps, H[:, 1], s, dev) - height(H[:, 0] - eps, H[:, 1], s, dev)) / (2 * eps)
    hy = (height(H[:, 0], H[:, 1] + eps, s, dev) - height(H[:, 0], H[:, 1] - eps, s, dev)) / (2 * eps)
    nrm = np.stack([-hx, -hy, np.ones_like(hx)], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    to_cam = np.array([0, 0, dev["Zc"]]) - H
    dist = np.linalg.norm(to_cam, axis=1)
    cos_v = np.clip(np.sum(nrm * to_cam, 1) / dist, 0.05, 1)
    fan = 1 / (1 + (np.abs(phi) / np.radians(48)) ** 8)
    amp = fan * cos_v * (82.0 / dist) ** 2
    for a, b_ in s["gaps"]:                                         # debris hides part of the line
        amp = np.where((u > a * width) & (u < b_ * width), 0, amp)

    inside &= amp > 0                                              # hidden by debris: not part of the visible line
    line = np.zeros((height_px, width), np.float32)
    ui, vi = u[inside], v[inside]
    u0, v0 = np.floor(ui).astype(int), np.floor(vi).astype(int)
    fu, fv = ui - u0, vi - v0
    for du, dv, w in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
        np.add.at(line, (np.clip(v0 + dv, 0, height_px - 1), np.clip(u0 + du, 0, width - 1)), amp[inside] * w)
    px_per_mm = f / (dev["Zc"] - dev["crown_z"])
    sig = float(np.hypot(s["line_w_mm"] * px_per_mm / 2.355, s["blur_px"]))
    line = cv2.GaussianBlur(line, (0, 0), sig)
    ref = np.percentile(line[line > line.max() * 0.2], 75) if line.max() > 0 else 1.0
    line = line / max(ref, 1e-9) * s["line_peak_dn"]
    if s["speckle"] > 0:
        k = 1 / s["speckle"] ** 2
        sp = cv2.GaussianBlur(rng.gamma(k, 1 / k, line.shape).astype(np.float32), (0, 0), 0.7)
        line *= sp / sp.mean()
    for _ in range(s["spikes"]):                                    # wet rubber: specular flashes on the line
        if inside.any():
            j = rng.choice(np.nonzero(inside)[0])
            cv2.circle(line, (int(round(u[j])), int(round(v[j]))), int(rng.integers(1, 4)), 900.0, -1)
    line_clean = line.copy()

    # --- ambient image: rubber texture, dark grooves, light gradient, glints, paint
    X, Y = pixel_to_plane(width, height_px, dev["crown_z"] if "flat_z" not in s else s["flat_z"], dev)
    g = groove_depth(X, Y, s)
    tex = 1 + 0.18 * _smooth_noise(rng, line.shape, 5) + 0.07 * _smooth_noise(rng, line.shape, 1.2)
    vv, uu = np.mgrid[0:height_px, 0:width]
    grad = 1 + s["ambient_grad"][0] * (uu / width - 0.5) + s["ambient_grad"][1] * (vv / height_px - 0.5)
    amb = s["rubber_dn"] * s["ambient"] * tex * np.clip(grad, 0.15, 3) * (1 - 0.75 * g / (g + 2.5))
    for gl in s["glints"]:
        cu, cv_ = gl["u"] * width, gl["v"] * height_px
        a = (uu - cu) * np.cos(gl["ang"]) + (vv - cv_) * np.sin(gl["ang"])
        b_ = -(uu - cu) * np.sin(gl["ang"]) + (vv - cv_) * np.cos(gl["ang"])
        amb = amb + gl["amp"] * s["ambient"] * 2 * np.exp(-(a / gl["len"]) ** 2 - (b_ / gl["wid"]) ** 2)
    if s["paint"]:
        amb = amb + s["paint"]["amp"] * s["ambient"] * 2 * np.exp(-((vv - s["paint"]["v"] * height_px) / s["paint"]["wid"]) ** 4)
    amb = cv2.GaussianBlur(amb.astype(np.float32), (0, 0), max(0.5, s["blur_px"]))

    def sensor(img):
        img = np.clip(img, 0, None)
        img = img + rng.normal(0, 1, img.shape) * np.sqrt(img * 0.35 + s["read_noise"] ** 2)
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)
        if jpeg_quality:
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1], 0)
        return img

    on = sensor(amb + line)
    off = sensor(amb) if return_off else None

    # --- truth
    gd = groove_depth(H[:, 0], H[:, 1], s)
    grooves = []
    for gr in s["grooves"]:
        yl = dev["crown_z"] * np.tan(np.radians(dev["theta_deg"]))
        edge = np.array([[gr["x"] - gr["w"] / 2, yl, 0.0], [gr["x"] + gr["w"] / 2, yl, 0.0]])
        edge[:, 2] = surface_z(edge[:, 0], edge[:, 1], s, dev)
        eu, _ = project(edge, width, height_px, dev)
        on_g = inside & (np.abs(H[:, 0] - gr["x"]) < gr["w"] / 2)
        floor = on_g & (gd >= 0.97 * gr["d"])
        # what the laser can actually reach and the camera can see: the deepest visible point
        seen = float(gd[on_g].max()) if on_g.any() else 0.0
        grooves.append({"x_mm": gr["x"], "w_mm": gr["w"], "depth_mm": gr["d"], "wall_deg": gr["wall_deg"],
                        "u0": float(min(eu)), "u1": float(max(eu)), "floor_visible_px": float(np.ptp(u[floor])) if floor.any() else 0.0,
                        "max_visible_depth_mm": seen,
                        "in_view": bool(min(eu) > 2 and max(eu) < width - 3)})
    truth = {
        "u": u[inside].astype(np.float32), "v": v[inside].astype(np.float32),
        "height_mm": H[inside, 2].astype(np.float32), "groove_depth_mm": gd[inside].astype(np.float32),
        "mask": line_clean > max(8.0, 0.25 * s["line_peak_dn"]),
        "grooves": grooves, "scene": s, "px_per_mm": float(px_per_mm), "sigma_px": sig,
    }
    return on, off, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", default="ideal", choices=["ideal", "stress", "train"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        s = sample_scene(rng, args.mode)
        on, off, t = render(s)
        cv2.imwrite(os.path.join(args.out, f"{args.mode}_{i:03d}_on.png"), on)
        cv2.imwrite(os.path.join(args.out, f"{args.mode}_{i:03d}_off.png"), off)
        print(i, json.dumps([{k: round(v, 1) if isinstance(v, float) else v for k, v in g.items()} for g in t["grooves"]]))


if __name__ == "__main__":
    main()
