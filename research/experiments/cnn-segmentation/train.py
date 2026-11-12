"""train.py - train the tiny U-Net of algos/cnn_line.py on the simulated `train` set only.

    cd research
    python experiments/cnn-segmentation/train.py --out models/cnn_line.pt --aug full --iters 1000 --minutes 7
    (run again with the same arguments to resume from the checkpoint until --iters is reached)

Target: Gaussian heat-map exp(-d^2 / 2 sigma^2), d = distance to the TRUE sub-pixel line (sigma 1 px),
so the label itself is sub-pixel accurate (a hard mask is only pixel accurate).
Loss: weighted soft-label BCE + soft Dice (see notes/cnn-segmentation.md).
Validation: the last 200 frames of `train` are never trained on.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from algos.cnn_line import TinyUNet  # noqa: E402
from bench import datasets  # noqa: E402

N_VAL = 200
SIGMA = 1.0
CROP = 160


def densify(u, v, max_gap, step=0.3):
    """Truth points are in ray order: fill the gaps between neighbours (up to max_gap px)."""
    if len(u) < 2:
        return u, v
    du, dv = np.diff(u), np.diff(v)
    d = np.hypot(du, dv)
    us, vs = [u], [v]
    for i in np.nonzero((d > step) & (d < max_gap))[0]:
        t = np.arange(1, int(d[i] / step) + 1) * step / d[i]
        us.append(u[i] + t * du[i])
        vs.append(v[i] + t * dv[i])
    return np.concatenate(us), np.concatenate(vs)


def heat_target(u, v, shape, sigma=SIGMA, r=3):
    """Gaussian of the distance to the nearest line point, exact to the sub-pixel."""
    h, w = shape
    d2 = np.full(shape, 1e6, np.float32)
    ok = (u > -r) & (u < w + r) & (v > -r) & (v < h + r)
    u, v = u[ok], v[ok]
    if len(u):
        iu, iv = np.round(u).astype(int), np.round(v).astype(int)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = iu + dx, iv + dy
                k = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                np.minimum.at(d2, (y[k], x[k]), ((x[k] - u[k]) ** 2 + (y[k] - v[k]) ** 2).astype(np.float32))
    return np.exp(-d2 / (2 * sigma ** 2))


def distractors(rng, shape):
    """Bright things that are NOT a laser line: blobs, slabs, edges, soft glows (real backgrounds)."""
    h, w = shape
    layer = np.zeros(shape, np.float32)
    for _ in range(int(rng.integers(1, 5))):
        one = np.zeros(shape, np.float32)
        c = (int(rng.integers(-20, w + 20)), int(rng.integers(-20, h + 20)))
        size = (int(rng.integers(4, 110)), int(rng.integers(4, 110)))
        amp = float(rng.uniform(30, 300))
        if rng.random() < 0.5:
            cv2.ellipse(one, c, size, float(rng.uniform(0, 180)), 0, 360, amp, -1)
        else:
            box = cv2.boxPoints((c, (size[0] * 2, size[1] * 2), float(rng.uniform(0, 180))))
            cv2.fillConvexPoly(one, box.astype(np.int32), amp)
        layer += cv2.GaussianBlur(one, (0, 0), float(np.exp(rng.uniform(np.log(0.5), np.log(8)))))
    return layer


def sample(item, rng, aug):
    """-> (image float32 0..1, heat-map target) of size CROP x CROP."""
    tr = item["truth"]
    neg = rng.random() < 0.12                       # laser-off frame of the same scene: target is empty
    img = (item["off"] if neg else item["on"]).astype(np.float32)
    u, v = (np.zeros(0), np.zeros(0)) if neg else (tr["u"].astype(float), tr["v"].astype(float))
    h, w = img.shape
    if aug == "none":                               # random crop + left-right flip only
        x0, y0 = int(rng.integers(0, w - CROP + 1)), int(rng.integers(0, h - CROP + 1))
        M = np.array([[1, 0, -x0], [0, 1, -y0]], float)
        if rng.random() < 0.5:
            M = np.array([[-1, 0, x0 + CROP - 1], [0, 1, -y0]], float)
        scale = 1.0
    else:                                           # zoom (thicker line, bigger grooves), rotation, flips, shift
        scale = float(np.exp(rng.uniform(np.log(0.8), np.log(2.6))))
        ang = float(np.clip(rng.normal(0, 10), -35, 35))
        if len(u) and rng.random() < 0.8:           # centre near a random line point
            j = int(rng.integers(len(u)))
            cx, cy = u[j] + rng.normal(0, 30 / scale), v[j] + rng.normal(0, 45 / scale)
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, scale)
        M[:, 2] += CROP / 2 - np.array([cx, cy])
        if rng.random() < 0.5:
            M = np.array([[-1, 0, CROP - 1], [0, 1, 0]], float) @ np.vstack([M, [0, 0, 1]])
        if rng.random() < 0.3:
            M = np.array([[1, 0, 0], [0, -1, CROP - 1]], float) @ np.vstack([M, [0, 0, 1]])
    out = cv2.warpAffine(img, M[:2], (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=float(rng.uniform(0, 50)))
    if len(u):
        u, v = densify(u, v, 3.0)
        u, v = M[0, 0] * u + M[0, 1] * v + M[0, 2], M[1, 0] * u + M[1, 1] * v + M[1, 2]
        u, v = densify(u, v, 3.0 * scale)
    y = heat_target(u, v, (CROP, CROP))
    if aug != "none":
        if rng.random() < 0.6:
            d = distractors(rng, out.shape)
            out = out + d
            y = y * (d < 100)                       # a line under a bright object cannot be seen
        if rng.random() < 0.3:                      # bright, uneven background (white floor, sunlight)
            yy, xx = np.mgrid[0:CROP, 0:CROP] / CROP
            out = out + rng.uniform(0, 60) * (1 + rng.normal(0, 0.6) * (xx - 0.5) + rng.normal(0, 0.6) * (yy - 0.5))
        out = np.clip(out * float(np.exp(rng.uniform(np.log(0.6), np.log(2.5)))), 0, 255)
        out = 255.0 * (out / 255.0) ** float(rng.uniform(0.6, 1.6))
        if rng.random() < 0.5:
            out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.8)))
        out = out + rng.normal(0, rng.uniform(0, 6), out.shape)
    return np.clip(out, 0, 255).astype(np.float32) / 255.0, y.astype(np.float32)


def loss_fn(logit, y):
    """Thin target (~1 % of the pixels): weight the line pixels in the BCE and add a Dice term."""
    bce = F.binary_cross_entropy_with_logits(logit, y, weight=1 + 10 * y)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / ((p * p).sum((1, 2, 3)) + (y * y).sum((1, 2, 3)) + 1)
    return bce + 0.5 * dice.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--aug", default="full", choices=["full", "none"])
    ap.add_argument("--iters", type=int, default=1800)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--minutes", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    t_start = time.time()
    items = datasets.load_sim("train")
    train, val = items[:-N_VAL], items[-N_VAL:]
    ck_path = args.out + ".ckpt"
    torch.manual_seed(args.seed)
    model = TinyUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    it0 = 0
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        it0 = ck["iter"]
        print(f"resumed at iter {it0}")
    rng = np.random.default_rng(args.seed * 100003 + it0)
    vrng = np.random.default_rng(7)
    vb = [sample(val[i], vrng, args.aug) for i in range(0, N_VAL, 4)]          # fixed validation crops
    vx, vy = (torch.from_numpy(np.stack([b[k] for b in vb]))[:, None] for k in (0, 1))
    print(f"params {sum(p.numel() for p in model.parameters())}", flush=True)
    model.train()
    run, it = 0.0, it0
    while it < args.iters and time.time() - t_start < args.minutes * 60:
        for g in opt.param_groups:                                             # warm-up + cosine
            g["lr"] = float(args.lr * min(1, (it + 1) / 100) * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * it / args.iters))))
        b = [sample(train[int(rng.integers(len(train)))], rng, args.aug) for _ in range(args.batch)]
        x, y = (torch.from_numpy(np.stack([s[k] for s in b]))[:, None] for k in (0, 1))
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        it += 1
        run = loss.item() if run == 0 else 0.98 * run + 0.02 * loss.item()
        if it % 100 == 0:
            print(f"iter {it} loss {run:.4f} {time.time() - t_start:.0f}s", flush=True)
    model.eval()
    with torch.no_grad():
        vl = float(np.mean([loss_fn(model(vx[i:i + 10]), vy[i:i + 10]).item() for i in range(0, len(vx), 10)]))
    print(f"stopped at iter {it}/{args.iters}  train loss {run:.4f}  val loss {vl:.4f}")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it}, ck_path)
    torch.save({k: v.float() for k, v in model.state_dict().items()}, args.out)
    print("DONE" if it >= args.iters else "RESUME: run the same command again")


if __name__ == "__main__":
    main()
