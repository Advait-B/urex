"""fuse.py - fuse per-groove depths over N presses; error vs N for every rule. Reads presses_<cond>.pkl."""
import json
import pickle

import numpy as np

LEGAL = 1.6
NS = [1, 3, 5, 9]


def mad_rule(x):
    med = np.median(x)
    mad = max(1.4826 * np.median(np.abs(x - med)), 0.1)          # floor: do not reject 0.05 mm disagreements
    return float(np.mean(x[np.abs(x - med) <= 3 * mad]))


def trimmed(x):
    x = np.sort(x)
    return float(np.mean(x[1:-1])) if len(x) >= 3 else float(np.mean(x))    # drop the lowest and the highest press


ALL = ["mean", "median", "trimmed", "median+MAD", "quality", "consensus"]
RULES = {"mean": np.mean, "median": np.median, "trimmed": trimmed, "median+MAD": mad_rule}


def fuse(d, q, rule):
    """d: depths of one groove over the chosen presses (nan = not found), q: quality score per press."""
    ok = np.isfinite(d)
    if rule == "consensus":                     # median, but only if at least half of the presses found the groove
        return float(np.median(d[ok])) if ok.sum() >= max(1, len(d) / 2) else None
    if rule == "quality":                       # keep the best two thirds of the presses by score, then median
        if ok.sum() == 0:
            return None
        keep = np.argsort(-np.where(ok, q, -1))[:max(1, int(np.ceil(ok.sum() * 2 / 3)))]
        return float(np.median(d[keep]))
    return float(RULES[rule](d[ok])) if ok.any() else None


def analyse(cond, algo, score="inlier", K=40, seed=0):
    recs = pickle.load(open(f"experiments/onoff-fusion/presses_{cond}.pkl", "rb"))
    out = {}
    for N in NS:
        for rule in ALL:
            rng = np.random.default_rng(seed + N)            # the same press subsets for every rule
            errs, miss, n, vok, fp, nv = [], 0, 0, 0, 0, 0
            for rec in recs:
                P = len(rec["press"])
                subsets = [np.arange(P)] if N == P else [rng.choice(P, N, replace=False) for _ in range(K)]
                for sub in subsets:
                    fused, seen = [], []
                    for gi, td in enumerate(rec["true"]):
                        use = [p for p in sub if rec["press"][p]["in_view"][gi]]
                        if not use:
                            continue
                        scor = all(p["scorable"][gi] for p in rec["press"])   # fixed groove set, the same for every N
                        d = np.array([np.nan if rec["press"][p][algo]["d"][gi] is None else rec["press"][p][algo]["d"][gi] for p in use])
                        q = np.array([rec["press"][p][algo][score] for p in use])
                        f = fuse(d, q, rule)
                        seen.append(td)
                        if f is not None:
                            fused.append(f)
                        if not scor:                       # floor hidden in some press: not scored (as in the harness)
                            continue
                        n += 1
                        if f is None:
                            miss += 1
                        else:
                            errs.append(f - td)
                    if not seen:
                        continue
                    tp = all(t >= LEGAL for t in seen)
                    pp = bool(fused) and all(f >= LEGAL for f in fused)
                    nv += 1
                    vok += tp == pp
                    fp += pp and not tp
            e = np.array(errs)
            out[f"{N}|{rule}"] = {"mae": float(np.mean(np.abs(e))), "median_abs": float(np.median(np.abs(e))),
                                  "p95": float(np.percentile(np.abs(e), 95)), "bias": float(np.mean(e)),
                                  "miss": miss / max(n, 1), "verdict_acc": vok / nv, "false_pass": fp / nv}
    # noise vs bias: per groove, mean error over all presses (repeatable part) and spread between presses
    mu, sd, sdr = [], [], []
    for rec in recs:
        for gi, td in enumerate(rec["true"]):
            d = np.array([p[algo]["d"][gi] for p in rec["press"] if p["scorable"][gi] and p[algo]["d"][gi] is not None], float)
            if len(d) >= 5 and all(p["scorable"][gi] for p in rec["press"]):
                mu.append(np.mean(d) - td)
                sd.append(np.std(d, ddof=1))
                sdr.append(1.4826 * np.median(np.abs(d - np.median(d))))
                mu[-1] = np.median(d) - td
    out["decomp"] = {"n_grooves": len(mu), "median_abs_groove_bias": float(np.median(np.abs(mu))), "rms_groove_bias": float(np.sqrt(np.mean(np.square(mu)))),
                     "mean_groove_bias": float(np.mean(mu)), "median_press_sd": float(np.median(sd)), "rms_press_sd": float(np.sqrt(np.mean(np.square(sd)))),
                     "median_press_robust_sd": float(np.median(sdr))}
    return out


if __name__ == "__main__":
    res = {}
    for cond in ["ideal", "stress", "legal", "legal_stress"]:
        for algo in ["single", "onoff"]:
            r = res[f"{cond}|{algo}"] = analyse(cond, algo)
            print(f"\n== {cond} / {algo}   decomp {json.dumps({k: round(v, 3) for k, v in r['decomp'].items()})}")
            print("rule        " + "".join(f"   N={N}: MAE   P95  miss  acc  fpass" for N in NS))
            for rule in ALL:
                print(f"{rule:11s} " + "".join("   {mae:.3f}/{median_abs:.3f} {p95:.3f} {miss:.3f} {verdict_acc:.3f} {false_pass:.3f}".format(**r[f"{N}|{rule}"]) for N in NS))
    json.dump(res, open("experiments/onoff-fusion/fusion.json", "w"), indent=1)
