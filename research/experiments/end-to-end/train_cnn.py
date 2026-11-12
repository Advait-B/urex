"""train_cnn.py - train algos/e2e_cnn.py on the simulated `train` set (first 1000 frames; last 200 = validation).

    python experiments/end-to-end/train_cnn.py --minutes 6      # resumable: run again to continue
Target per image column: 1 / depth_mm inside an in-view groove (truth u0..u1), 0 on the tread.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from algos import e2e_cnn as M  # noqa: E402
from bench import datasets  # noqa: E402

CKPT = "experiments/end-to-end/cnn_ckpt.pt"
EPOCHS = 40


def targets(items):
    X = np.stack([it["on"][M.R0:M.R1] for it in items])
    mask, dep = np.zeros((len(items), M.W), np.float32), np.zeros((len(items), M.W), np.float32)
    cols = np.arange(M.W)
    for i, it in enumerate(items):
        for g in it["truth"]["grooves"]:
            if g["in_view"]:
                m = (cols >= g["u0"]) & (cols <= g["u1"])
                mask[i, m], dep[i, m] = 1, g["depth_mm"]
    return torch.from_numpy(X), torch.from_numpy(mask), torch.from_numpy(dep)


def loss_fn(net, x, m, d):
    logit, depth = net(x)
    bce = F.binary_cross_entropy_with_logits(logit, m)
    l1 = (F.smooth_l1_loss(depth, d, reduction="none", beta=0.5) * m).sum() / m.sum().clamp(min=1)
    return bce + l1, bce.item(), l1.item()


def augment(x, m, d, rng):
    x = x.float() / 255.0
    if rng.random() < 0.5:                       # the device is left-right symmetric
        x, m, d = x.flip(-1), m.flip(-1), d.flip(-1)
    x = x * torch.from_numpy(rng.uniform(0.7, 1.3, (len(x), 1, 1)).astype(np.float32))
    return x.clamp(0, 1)[:, None], m, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    items = datasets.load_sim("train")
    Xt, Mt, Dt = targets(items[:1000][:args.limit])
    Xv, Mv, Dv = targets(items[1000:])
    net = M.Net()
    print("parameters:", sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = EPOCHS * (len(Xt) // 16)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    ep0 = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); ep0 = ck["epoch"]
    t0 = time.time()
    for ep in range(ep0, EPOCHS):
        if time.time() - t0 > args.minutes * 60:
            break
        rng = np.random.default_rng(ep)
        net.train()
        perm = rng.permutation(len(Xt))
        for i in range(0, len(perm) - 15, 16):
            b = perm[i:i + 16]
            x, m, d = augment(Xt[b], Mt[b], Dt[b], rng)
            loss, _, _ = loss_fn(net, x, m, d)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            _, bce, l1 = loss_fn(net, Xv.float()[:, None] / 255, Mv, Dv)
        print(f"epoch {ep + 1:2d}  val bce {bce:.4f}  val depth L1 {l1:.3f} mm  ({time.time() - t0:.0f} s)", flush=True)
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1}, CKPT)
        torch.save(net.state_dict(), M.MODEL)
    print("done" if ep + 1 >= EPOCHS else "time budget reached: run again to continue")


if __name__ == "__main__":
    main()
