"""robust_depth.py - alternatives for the "line points -> groove depths -> verdict" stage.

    stage(u, v, calib, surface=..., floor=..., detect=..., correct=...) -> {"grooves": [...], ...}

Same input/output as bench.depth.measure_from_line, but every step can be swapped:
  surface  "ransac" (bench default) | "huber" | "theilsen" | "ransac_irls"
  floor    "band" (bench default) | "deepfrac" | "kde" | "trapezoid" | "changepoint" | "gmm" | "dbscan"
  detect   "default" (r > min_depth)  | "shape" (low, noise-adaptive threshold + width/area test)
  correct  True: remove the two geometric biases (mm only, needs the holder constants below)
Every groove also gets "sigma" (1-sigma uncertainty, same unit), "floor_px" (width of the floor
plateau that was found) and "lower_bound" (True when no flat floor was seen: depth >= reading).

Thin wrappers (rd_*.py) pair one configuration with the baseline line extractor for bench.run.
"""
import warnings

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from bench.depth import ransac_quadratic

# thresholds per unit. px values = mm values x ~2.1 px/mm (rounded, as in bench.depth)
MM = {"ransac": 0.30, "min_depth": 0.5, "band": 0.25, "irls": 0.10, "lo": 0.10, "bw": 0.06, "tol": 0.12,
      "pen": 0.05, "noise": 0.02, "hidden": 1.0}
PX = {"ransac": 1.2, "min_depth": 2.5, "band": 0.7, "irls": 0.30, "lo": 0.35, "bw": 0.20, "tol": 0.40,
      "pen": 0.50, "noise": 0.15, "hidden": 2.1}
# holder constants used by the bias correction (CAD values; see notes/robust-depth.md)
ZC_MM, TAN_LASER, TYRE_R_MM = 85.0, np.tan(np.radians(35.0)), 480.0
# uncertainty model, tuned on `train` (see experiments/robust-depth/uncertainty.py)
SIG = {"tilt_deg": 0.75, "floor_mm": 0.03, "corr_px": 3.0, "scale": 1.0}


# ------------------------------------------------------------------ surface
def fit_surface(u, q, P, method="ransac"):
    """Quadratic tread surface q = a*u^2 + b*u + c. Returns (coef, inlier mask)."""
    x = (u - 160.0) / 160.0
    A = np.c_[x * x, x]
    if method == "ransac":
        return ransac_quadratic(u, q, P["ransac"])
    if method == "huber":
        from sklearn.linear_model import HuberRegressor
        m = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=300).fit(A, q)
        cx = np.array([m.coef_[0], m.coef_[1], m.intercept_])
    elif method == "theilsen":
        from sklearn.linear_model import TheilSenRegressor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = TheilSenRegressor(random_state=0, max_subpopulation=2000).fit(A, q)
        cx = np.array([m.coef_[0], m.coef_[1], m.intercept_])
    elif method == "ransac_irls":                 # RANSAC start, then a redescending (Cauchy) M-estimate
        c0, _ = ransac_quadratic(u, q, P["ransac"])
        s = least_squares(lambda c: np.polyval(c, u) - q, c0, loss="cauchy", f_scale=P["irls"], x_scale="jac")
        return s.x, np.abs(q - np.polyval(s.x, u)) < P["ransac"]
    else:
        raise ValueError(method)
    coef = np.array([cx[0] / 160.0 ** 2, cx[1] / 160.0 - 2 * cx[0] / 160.0, cx[0] - cx[1] + cx[2]])
    return coef, np.abs(q - np.polyval(coef, u)) < P["ransac"]


# ---------------------------------------------------------------- detection
def find_runs(u, r, thresh, gap_px=5.0, min_points=5, min_width_px=4.0):
    idx = np.nonzero(r > thresh)[0]
    if len(idx) < min_points:
        return []
    idx = idx[np.argsort(u[idx])]
    runs = np.split(idx, np.nonzero(np.diff(u[idx]) > gap_px)[0] + 1)
    return [k for k in runs if len(k) >= min_points and np.ptp(u[k]) >= min_width_px]


def detect_shape(u, r, inl, P, lo_k=2.5, min_width_px=10.0):
    """The default rule PLUS a shallow-groove rule on what is left: a groove is 20-40 px wide, so smooth
    the profile across 9 px (running median: noise drops, a groove survives), then accept runs that are
    lower than `lo` (just above the smoothed noise) over at least `min_width_px`."""
    # default rule, minus "sipes": a tread groove is wide; a narrow AND shallow dip is a sipe or a noise blip
    runs = [k for k in find_runs(u, r, P["min_depth"]) if np.ptp(u[k]) >= min_width_px or r[k].max() > 5 * P["min_depth"]]
    cols = np.arange(int(np.floor(u.min())), int(np.ceil(u.max())) + 1)
    prof = np.full(len(cols), np.nan)
    ci = np.clip(np.round(u).astype(int) - cols[0], 0, len(cols) - 1)
    for c in np.unique(ci):
        prof[c] = np.median(r[ci == c])
    have = np.isfinite(prof)
    sm = median_filter(np.interp(cols, cols[have], prof[have]), size=9, mode="nearest")
    flat = inl & (np.abs(r) < P["lo"])
    noise = 1.4826 * np.median(np.abs(r[flat] - np.median(r[flat]))) if flat.sum() > 8 else P["noise"]
    lo = max(P["lo"], lo_k * noise)
    low = (sm > lo) & have
    for k in runs:                                         # columns already claimed by a default groove
        low[max(0, int(u[k].min()) - 3 - cols[0]):int(u[k].max()) + 4 - cols[0]] = False
    edges = np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):              # columns [a, b)
        k = np.nonzero((ci >= a) & (ci < b))[0]
        if b - a >= min_width_px and len(k) >= 6:
            runs.append(k)
    return runs


# ------------------------------------------------------------ floor estimators
# each returns (depth, boolean mask of the points it calls "floor") for one groove (uu sorted)
def floor_band(uu, rr, P):
    s = np.sort(rr)
    ref = s[-3] if len(s) >= 3 else s[-1]
    m = np.abs(rr - ref) <= P["band"]
    return rr[m].mean(), m


def floor_deepfrac(uu, rr, P, frac=0.3):
    m = rr >= np.quantile(rr, 1 - frac)
    return np.median(rr[m]), m


def floor_kde(uu, rr, P, rel=0.35):
    """Histogram peak: the DEEPEST mode that is at least `rel` as tall as the tallest one."""
    if np.ptp(rr) < 1e-6:
        return rr.mean(), np.ones(len(rr), bool)
    grid = np.linspace(rr.min() - 3 * P["bw"], rr.max() + 3 * P["bw"], 400)
    dens = gaussian_kde(rr, bw_method=P["bw"] / rr.std())(grid)
    pk, _ = find_peaks(np.r_[0, dens, 0])
    pk = pk - 1
    good = pk[dens[pk] >= rel * dens[pk].max()]
    d = grid[good.max()]
    return d, np.abs(rr - d) <= 2 * P["bw"]


def floor_trapezoid(uu, rr, P):
    """Trapezoid template D * clip((hw - |u - c|) / s, 0, 1) with a soft-L1 loss."""
    d0, _ = floor_band(uu, rr, P)

    def model(p):
        return p[3] * np.clip((p[1] - np.abs(uu - p[0])) / max(p[2], 0.3), 0, 1)
    p0 = [uu.mean(), np.ptp(uu) / 2 + 1, 3.0, d0]
    s = least_squares(lambda p: model(p) - rr, p0, loss="soft_l1", f_scale=P["tol"], x_scale=[2, 2, 1, max(d0, 0.1)])
    return s.x[3], np.abs(rr - s.x[3]) <= P["tol"]


def floor_changepoint(uu, rr, P):
    """Piecewise-constant segmentation along u (ruptures, kernel change-point detection);
    floor = the deepest segment that is at least 3 px wide."""
    import ruptures as rpt
    if len(rr) < 8:
        return floor_band(uu, rr, P)
    bk = rpt.KernelCPD(kernel="linear", min_size=3).fit(rr.reshape(-1, 1)).predict(pen=P["pen"])
    best, mask = -np.inf, None
    for a, b in zip([0] + bk[:-1], bk):
        if uu[b - 1] - uu[a] >= 3.0 and np.median(rr[a:b]) > best:
            best = np.median(rr[a:b])
            mask = np.zeros(len(rr), bool)
            mask[a:b] = True
    return (best, mask) if mask is not None else floor_band(uu, rr, P)


def floor_gmm(uu, rr, P):
    from sklearn.mixture import GaussianMixture
    if len(rr) < 12:
        return floor_band(uu, rr, P)
    X = rr.reshape(-1, 1)
    fits = [GaussianMixture(k, random_state=0, reg_covar=(P["bw"] / 3) ** 2).fit(X) for k in (1, 2, 3)]
    g = min(fits, key=lambda m: m.bic(X))
    lab = g.predict(X)
    ok = [k for k in range(g.n_components) if (lab == k).sum() >= max(4, 0.15 * len(rr))]
    k = max(ok, key=lambda k: g.means_[k, 0]) if ok else int(np.argmax(g.weights_))
    return float(rr[lab == k].mean()) if (lab == k).any() else float(g.means_[k, 0]), lab == k


def floor_dbscan(uu, rr, P):
    from sklearn.cluster import DBSCAN
    lab = DBSCAN(eps=1.0, min_samples=4).fit_predict(np.c_[uu / 3.0, rr / P["tol"]])
    ks = [k for k in set(lab) if k >= 0]
    if not ks:
        return floor_band(uu, rr, P)
    k = max(ks, key=lambda k: np.median(rr[lab == k]))
    return np.median(rr[lab == k]), lab == k


FLOORS = {"band": floor_band, "deepfrac": floor_deepfrac, "kde": floor_kde, "trapezoid": floor_trapezoid,
          "changepoint": floor_changepoint, "gmm": floor_gmm, "dbscan": floor_dbscan}


# ------------------------------------------------------------ bias correction
def correct_depth(coef, u_floor, d_raw, tyre_R=TYRE_R_MM, uc=159.5, parallax=True, curvature=True):
    """Remove the two geometric biases of d_raw = surface(u_floor) - floor (all in mm).
    1. parallax: the camera sees a floor point at depth d closer to the image centre than the tread
       above it, so the surface must be read at u_s, further out, not at u_floor.
    2. curvature: the laser (35 deg) meets the floor at Y_f = z_f*tan(35), the tread at Y_s = z_s*tan(35);
       the tyre has dropped by (Y_f^2 - Y_s^2) / 2R between the two."""
    z_f = -(np.polyval(coef, u_floor) + d_raw)             # height of the floor above the feet plane
    z_s = -np.polyval(coef, u_floor)
    if parallax:
        for _ in range(2):
            u_s = uc + (u_floor - uc) * (ZC_MM - z_f) / (ZC_MM - z_s)
            z_s = -np.polyval(coef, u_s)
    d = z_s - z_f
    if curvature:
        d -= TAN_LASER ** 2 * (z_f ** 2 - z_s ** 2) / (2 * tyre_R)
    return float(d)


# -------------------------------------------------------------------- stage
def stage(u, v, calib=None, surface="ransac", floor="band", detect="default", correct=True, tyre_R=TYRE_R_MM,
          parallax=True, curvature=True, lo_k=2.5):
    u, v = np.asarray(u, float), np.asarray(v, float)
    P = MM if calib is not None else PX
    unit = "mm" if calib is not None else "px"
    if len(u) < 12:
        return {"grooves": [], "surface": None, "inlier_frac": 0.0, "unit": unit}
    q = -calib.height(u, v) if calib is not None else v
    if calib is not None:                                  # outside the calibrated heights = not the laser on a tyre
        ok = (q > -calib.z_range[1] - 2.0) & (q < -calib.z_range[0] + 2.0)
        u, v, q = u[ok], v[ok], q[ok]
        if len(u) < 12:
            return {"grooves": [], "surface": None, "inlier_frac": 0.0, "unit": unit}
    coef, inl = fit_surface(u, q, P, surface)
    r = q - np.polyval(coef, u)
    if detect == "shape":
        runs = detect_shape(u, r, inl, P, lo_k)
        keep = np.ones(len(u), bool)                       # shallow grooves are RANSAC inliers: refit without them
        for k in runs:
            keep[(u >= u[k].min() - 2) & (u <= u[k].max() + 2)] = False
        if runs and (keep & inl).sum() >= 12 and np.ptp(u[keep & inl]) > 60:
            coef = np.polyfit(u[keep & inl], q[keep & inl], 2)
            r = q - np.polyval(coef, u)
            runs = detect_shape(u, r, inl & keep, P, lo_k)
    else:
        runs = find_runs(u, r, P["min_depth"])
    fit_pts = inl.copy()
    for k in runs:
        fit_pts[k] = False
    # covariance of the surface coefficients (for the uncertainty of the surface under a groove)
    cov, s_surf = np.zeros((3, 3)), P["noise"]
    if fit_pts.sum() >= 8:
        A = np.vander(u[fit_pts], 3)
        s_surf = max(float(np.std(r[fit_pts])), 1e-3)
        n_eff = max(4.0, np.ptp(u[fit_pts]) / SIG["corr_px"] * fit_pts.mean())
        cov = np.linalg.pinv(A.T @ A) * s_surf ** 2 * (fit_pts.sum() / n_eff)
    grooves = []
    for k in runs:
        k = k[np.argsort(u[k])]
        uu, rr = u[k], r[k]
        d, fm = FLOORS[floor](uu, rr, P)
        if not fm.any():
            d, fm = floor_band(uu, rr, P)
        u_f, floor_px = float(uu[fm].mean()), float(np.ptp(uu[fm]))
        hidden = bool(floor_px < 3.0 and d > P["hidden"])     # no flat floor seen: the best lower bound is the deepest point
        if hidden:
            d, u_f = float(rr.max()), float(uu[np.argmax(rr)])
        d_out = correct_depth(coef, u_f, d, tyre_R, parallax=parallax, curvature=curvature) if (correct and calib is not None) else float(d)
        # --- 1-sigma: floor scatter / sqrt(effective n)  (+)  surface under the groove  (+)  unknown rocking
        n_eff = max(1.0, min(fm.sum(), floor_px / SIG["corr_px"] + 1))
        s_floor = max(float(np.std(rr[fm])), SIG["floor_mm"] if unit == "mm" else P["noise"]) / np.sqrt(n_eff)
        a = np.array([u_f ** 2, u_f, 1.0])
        s_tilt = np.tan(np.radians(SIG["tilt_deg"])) * TAN_LASER * abs(d_out)
        sigma = SIG["scale"] * float(np.sqrt(s_floor ** 2 + a @ cov @ a + s_tilt ** 2))
        grooves.append({"u0": float(uu.min()), "u1": float(uu.max()), "depth": float(d_out), "unit": unit, "n": int(len(k)),
                        "sigma": sigma, "floor_px": floor_px, "raw": float(d),
                        "lower_bound": hidden})
    return {"grooves": grooves, "surface": [float(c) for c in coef], "inlier_frac": float(inl.mean()), "unit": unit}
