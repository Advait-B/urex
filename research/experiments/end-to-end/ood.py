"""ood.py - how do the two learners and the geometric baseline behave outside the training distribution?

    python experiments/end-to-end/ood.py [shift ...]      -> experiments/end-to-end/ood_results.json
Shifts (100 fresh frames each, scenes drawn like `train` unless stated):
  in_dist       control: same distribution as training, new seed
  laser_37deg   another unit: laser at 37 deg instead of 35 deg
  deep_18_22    grooves 18-22 mm (training stops at 18)
  four_grooves  4 grooves in view (training has 1-3)
  res_640x480   same device read out at 640x480
Also scores ideal / stress / legal with the same extra metric (error of the frame's minimum depth),
which is the only depth number all three approaches can be compared on.
"""
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "sim"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holder_sim as hs  # noqa: E402
import train_cnn  # noqa: E402
from algos import baseline_steger, e2e_cnn, e2e_gbm  # noqa: E402
from bench import calib as calib_mod  # noqa: E402
from bench import datasets, run  # noqa: E402

OUT = "experiments/end-to-end/ood_results.json"
N = 100
DEV37 = {**hs.DEVICE, "theta_deg": 37.0}


def make(shift, seed, n=N):
    rng = np.random.default_rng(seed)
    dev, size, items = hs.DEVICE, (320, 240), []
    if shift == "laser_37deg":
        dev = DEV37
    if shift == "res_640x480":
        size = (640, 480)
    for _ in range(n):
        s = hs.sample_scene(rng, "train")
        if shift == "deep_18_22":
            base = rng.uniform(18.5, 21.5)
            for g in s["grooves"]:
                g["d"] = float(np.clip(base + rng.normal(0, 0.5), 18.0, 22.0))
        if shift == "four_grooves":
            while True:
                cx = np.sort(rng.uniform(-44, 44, 4))
                if np.min(np.diff(cx)) >= 22:
                    break
            d0 = s["grooves"][0]["d"]
            s["grooves"] = [{"x": float(c), "w": float(rng.uniform(8, 14)), "wall_deg": float(rng.uniform(8, 22)),
                             "d": float(np.clip(d0 + rng.normal(0, 0.8), 0.4, 18.0))} for c in cx]
        on, off, t = hs.render(s, size[0], size[1], dev)
        if any(g["in_view"] for g in t["grooves"]):
            items.append({"on": on, "off": off, "truth": t})
    return items, dev, size


def plates(dev, size):
    rng = np.random.default_rng(909)
    out = []
    for z in datasets.CALIB_HEIGHTS:
        on, off, t = hs.render(hs.flat_scene(z, rng), size[0], size[1], dev)
        out.append({"on": on, "off": off, "truth": t, "height_mm": float(z)})
    return out


def score(measure, items, cal):
    """Harness metrics + error of the minimum depth per frame (comparable across all approaches)."""
    r = run.eval_sim(measure, items, cal)
    tmin, pmin = [], []
    for it in items:
        g = [x["depth"] for x in measure(it["on"], cal, it["off"])["grooves"]]
        tmin.append(min(x["depth_mm"] for x in it["truth"]["grooves"] if x["in_view"]))
        pmin.append(min(g) if g else None)
    e = np.array([abs(p - t) for p, t in zip(pmin, tmin) if p is not None])
    keep = ["n_frames", "depth_mae_mm", "depth_p95_mm", "depth_bias_mm", "miss_rate", "false_per_frame", "verdict_acc",
            "false_pass_rate"]
    return {**{k: r[k] for k in keep}, "min_mae_mm": float(e.mean()) if len(e) else None,
            "min_median_ae_mm": float(np.median(e)) if len(e) else None,
            "no_answer": float(np.mean([p is None for p in pmin])), "_true_min": [round(t, 3) for t in tmin],
            "_pred_min": [None if p is None else round(p, 3) for p in pmin]}


def finetune(items, epochs=10):
    """'Re-calibrate' the CNN the only way a network can be: more labelled frames from the new unit."""
    torch.manual_seed(0)
    net = copy.deepcopy(e2e_cnn.load()).train()
    X, Mk, D = train_cnn.targets(items)
    opt = torch.optim.AdamW(net.parameters(), 3e-4)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(perm) - 9, 10):
            b = perm[i:i + 10]
            x, m, d = train_cnn.augment(X[b], Mk[b], D[b], rng)
            loss, _, _ = train_cnn.loss_fn(net, x, m, d)
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cal0 = calib_mod.default_calib()
    _, base = run.resolve(baseline_steger)
    todo = sys.argv[1:] or ["ideal", "stress", "legal", "in_dist", "laser_37deg", "deep_18_22", "four_grooves", "res_640x480"]
    for k, shift in enumerate(todo):
        if shift in ("ideal", "stress", "legal"):
            items, dev, size = datasets.load_sim(shift), hs.DEVICE, (320, 240)
        else:
            items, dev, size = make(shift, 7000 + k)
        r = {"e2e_cnn": score(e2e_cnn.measure, items, None), "e2e_gbm": score(e2e_gbm.measure, items, None),
             "baseline": score(base, items, cal0)}
        if shift in ("laser_37deg", "res_640x480"):
            cal, info = calib_mod.fit(plates(dev, size), baseline_steger.extract_line)
            r["baseline_recal"] = {**score(base, items, cal), "calib_rms_mm": info["rms_mm"]}
        if shift == "laser_37deg":
            for n_lab in (15, 50):
                lab, _, _ = make(shift, 9000, n_lab)                      # labelled frames from the new unit
                net0, e2e_cnn._net = e2e_cnn._net, finetune(lab)
                r[f"e2e_cnn_finetune{n_lab}"] = score(e2e_cnn.measure, items, None)
                e2e_cnn._net = net0
                t = np.array([min(g["depth_mm"] for g in it["truth"]["grooves"] if g["in_view"]) for it in lab])
                p = np.array([e2e_gbm.predict_min_depth(it["on"]) for it in lab])
                gain = float(np.sum(p * t) / np.sum(p * p))               # one-number gain correction
                r[f"e2e_gbm_gain{n_lab}"] = {**score(lambda g, c=None, o=None: {"grooves": [
                    {"u0": 0.0, "u1": g.shape[1] - 1.0, "depth": gain * e2e_gbm.predict_min_depth(g), "unit": "mm"}]}, items, None),
                    "gain": gain}
        res[shift] = r
        json.dump(res, open(OUT, "w"), indent=1)
        for a, m in r.items():
            print(f"{shift:12s} {a:20s} " + " ".join(f"{q}={m[q]:.3f}" if m[q] is not None else f"{q}=-" for q in
                  ["depth_mae_mm", "depth_bias_mm", "miss_rate", "false_per_frame", "verdict_acc", "false_pass_rate", "min_mae_mm", "no_answer"]), flush=True)


if __name__ == "__main__":
    main()
