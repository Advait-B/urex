"""calib_models.py - px -> mm calibration models. All share one tiny interface:

    m = Model(...).fit(u, v, z)      # laser-line points (px) of flat plates and each point's plate height (mm)
    m.height(u, v)                   # height in mm above the feet plane (grooves are negative-going)

  PolyModel    height = poly(u, v)                     (bench/calib.py is PolyModel(4, 3))
  PhysModel    pinhole + one radial distortion term + a laser plane; 3-7 free numbers, each with a meaning
  SplineModel  per-column lookup: a spline z(v) through the plates, one per image column
  GPModel      Gaussian-process regression on (u, v) around a bilinear trend

PhysModel also gives the full 3-D point (xyz) and can be re-aimed from ONE plate photo (refit_plane).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

W, H = 320, 240


def _norm(u, v, w=W, h=H):
    return np.asarray(u, float) / (w - 1) - 0.5, np.asarray(v, float) / (h - 1) - 0.5


# ---------------------------------------------------------------- polynomial
class PolyModel:
    def __init__(self, deg_u=4, deg_v=3, w=W, h=H):
        self.deg_u, self.deg_v, self.w, self.h = deg_u, deg_v, w, h
        self.n_params = (deg_u + 1) * (deg_v + 1)

    def _design(self, u, v):
        a, b = _norm(u, v, self.w, self.h)
        return np.stack([a ** i * b ** j for i in range(self.deg_u + 1) for j in range(self.deg_v + 1)], axis=-1)

    def fit(self, u, v, z):
        self.coef, *_ = np.linalg.lstsq(self._design(u, v), np.asarray(z, float), rcond=None)
        return self

    def height(self, u, v):
        return self._design(u, v) @ self.coef

    def to_json(self):
        return {"deg_u": self.deg_u, "deg_v": self.deg_v, "w": self.w, "h": self.h, "coef": self.coef.tolist()}

    @classmethod
    def from_json(cls, d):
        m = cls(d["deg_u"], d["deg_v"], d["w"], d["h"])
        m.coef = np.array(d["coef"])
        return m


# ------------------------------------------------------------------ physical
class PhysModel:
    """Camera at (0, 0, Zc) looking straight down; pixel -> ray through a pinhole with one radial
    distortion term; the ray is cut with the laser plane. Parameters (all in self.p):
        cx, cy   principal point (px)          k1     radial distortion
        Zc       lens height above the feet    theta  laser angle to the camera axis (rad)
        roll     laser line roll (rad)         y0     where the laser plane crosses Z = pz at X = 0 (mm)
    The focal length f_px is FIXED at the data-sheet value: flat plates cannot separate it from
    (k1, theta) - scaling f by s, k1 by s^2 and re-aiming the plane gives exactly the same heights.
    """
    NAMES = ("cx", "cy", "k1", "Zc", "theta", "roll", "y0")
    SCALE = {"cx": 10.0, "cy": 10.0, "k1": 0.05, "Zc": 5.0, "theta": 0.05, "roll": 0.01, "y0": 1.0}

    def __init__(self, free=("k1", "Zc", "theta", "roll", "y0"), w=W, h=H, f_px=None, init=None):
        self.free, self.w, self.h = tuple(free), w, h
        self.f_px = f_px if f_px else 2.7 / 3.67 * w
        self.p = {"cx": (w - 1) / 2, "cy": (h - 1) / 2, "k1": 0.0, "Zc": 85.0, "theta": np.radians(35.0),
                  "roll": 0.0, "y0": 0.0}                      # CAD / data-sheet starting values
        self.p.update(init or {})
        self.pz = 0.0                                          # height of the plane's anchor point
        self.n_params = len(self.free)

    def xyz(self, u, v, p=None):
        """Pixel -> 3-D point on the laser plane (device mm: X across the tread, Y along it, Z up)."""
        p = p or self.p
        xd, yd = (np.asarray(u, float) - p["cx"]) / self.f_px, (np.asarray(v, float) - p["cy"]) / self.f_px
        x, y = xd, yd
        for _ in range(6):                                     # undo the distortion (fixed-point iteration)
            k = 1 + p["k1"] * (x * x + y * y)
            x, y = xd / k, yd / k
        st, ct, sr, cr = np.sin(p["theta"]), np.cos(p["theta"]), np.sin(p["roll"]), np.cos(p["roll"])
        b = np.array([0.0, -st, -ct])                          # beam centre direction
        xr = np.array([cr, sr * ct, -sr * st])                 # fan axis
        n = np.cross(b, xr)                                    # laser-plane normal
        # ray P = C + t*(x, -y, -1), C = (0, 0, Zc);  plane n.(P - A) = 0, A = (0, y0, pz)
        t = (n[1] * p["y0"] + n[2] * (self.pz - p["Zc"])) / (n[0] * x - n[1] * y - n[2])
        return x * t, -y * t, p["Zc"] - t

    def height(self, u, v):
        return self.xyz(u, v)[2]

    def project(self, X, Y, Z):
        """3-D point -> pixel (the camera half of the model, forwards)."""
        p = self.p
        x, y = X / (p["Zc"] - Z), -Y / (p["Zc"] - Z)
        k = 1 + p["k1"] * (x * x + y * y)
        return p["cx"] + self.f_px * x * k, p["cy"] + self.f_px * y * k

    def plane_y(self, z):
        """Where (along the tyre, mm) the laser plane meets height z, at X = 0."""
        return self.p["y0"] + (z - self.pz) * np.tan(self.p["theta"])

    def _solve(self, u, v, z, free):
        def resid(q):
            return self.xyz(u, v, {**self.p, **dict(zip(free, q))})[2] - z
        r = least_squares(resid, [self.p[k] for k in free], x_scale=[self.SCALE[k] for k in free])
        self.p.update(dict(zip(free, map(float, r.x))))
        self.jac, self.rms = r.jac, float(np.sqrt(np.mean(r.fun ** 2)))
        return self

    def fit(self, u, v, z):
        return self._solve(np.asarray(u, float), np.asarray(v, float), np.asarray(z, float), self.free)

    def refit_plane(self, u, v, z=0.0, pivot_z=90.1):
        """Zero check: ONE plate at known height z. Keeps the camera, assumes the laser only turned
        about its exit (pivot_z mm above the feet, from CAD) and re-solves theta and roll."""
        m = PhysModel(self.free, self.w, self.h, self.f_px, dict(self.p))
        m.p["y0"] = self.p["y0"] + (pivot_z - self.pz) * np.tan(self.p["theta"])   # the same plane, anchored at the exit
        m.pz = pivot_z
        return m._solve(np.asarray(u, float), np.asarray(v, float), np.full(len(u), float(z)), ("theta", "roll"))

    def to_json(self):
        return {"free": list(self.free), "w": self.w, "h": self.h, "f_px": self.f_px, "p": self.p, "pz": self.pz}

    @classmethod
    def from_json(cls, d):
        m = cls(d["free"], d["w"], d["h"], d["f_px"], d["p"])
        m.pz = d["pz"]
        return m


# -------------------------------------------------------- per-column spline
class SplineModel:
    """For every column: the row where each plate's line sits -> spline z(row). Each plate's line is
    first smoothed along u with a low-degree polynomial. Outside the plates it continues straight."""

    def __init__(self, line_deg=4, w=W, h=H):
        self.line_deg, self.w, self.h = line_deg, w, h

    def fit(self, u, v, z):
        u, v, z = (np.asarray(a, float) for a in (u, v, z))
        cols = np.arange(self.w, dtype=float)
        zs = np.unique(z)
        rows = np.stack([np.polyval(np.polyfit(u[z == zk] / self.w - 0.5, v[z == zk], self.line_deg), cols / self.w - 0.5)
                         for zk in zs])                      # (plates, columns)
        self.n_params = len(zs) * (self.line_deg + 1)
        vq = np.arange(self.h, dtype=float)
        self.table = np.empty((self.h, self.w))
        for c in range(self.w):
            o = np.argsort(rows[:, c])
            r, zz = rows[o, c], zs[o]
            if len(r) >= 3:
                sp = CubicSpline(r, zz, bc_type="natural")
                val, s0, s1 = sp(vq), sp(r[0], 1), sp(r[-1], 1)
            else:
                s0 = s1 = (zz[-1] - zz[0]) / (r[-1] - r[0])
                val = zz[0] + s0 * (vq - r[0])
            val = np.where(vq < r[0], zz[0] + s0 * (vq - r[0]), val)       # straight continuation outside
            val = np.where(vq > r[-1], zz[-1] + s1 * (vq - r[-1]), val)
            self.table[:, c] = val
        return self

    def height(self, u, v):
        u, v = np.clip(np.asarray(u, float), 0, self.w - 1.001), np.clip(np.asarray(v, float), 0, self.h - 1.001)
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - u0, v - v0
        T = self.table
        return ((1 - fu) * (1 - fv) * T[v0, u0] + fu * (1 - fv) * T[v0, u0 + 1]
                + (1 - fu) * fv * T[v0 + 1, u0] + fu * fv * T[v0 + 1, u0 + 1])


# ---------------------------------------------------------- Gaussian process
class GPModel:
    """GP (RBF + white noise) on the residual of a bilinear trend, trained on a random subset."""

    def __init__(self, n_max=400, w=W, h=H, seed=0):
        self.n_max, self.w, self.h, self.seed = n_max, w, h, seed
        self.n_params = n_max

    def _trend(self, a, b):
        return np.stack([np.ones_like(a), a, b, a * b], -1)

    def fit(self, u, v, z):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        a, b = _norm(u, v, self.w, self.h)
        z = np.asarray(z, float)
        self.tc, *_ = np.linalg.lstsq(self._trend(a, b), z, rcond=None)
        i = np.random.default_rng(self.seed).choice(len(z), min(self.n_max, len(z)), replace=False)
        kern = ConstantKernel(1.0, (1e-3, 1e3)) * RBF([0.5, 0.5], (0.05, 20)) + WhiteKernel(1e-3, (1e-7, 1e-1))
        self.gp = GaussianProcessRegressor(kern, normalize_y=False, random_state=0)
        self.gp.fit(np.c_[a[i], b[i]], z[i] - self._trend(a[i], b[i]) @ self.tc)
        return self

    def height(self, u, v):
        a, b = _norm(u, v, self.w, self.h)
        shp = a.shape
        a, b = a.ravel(), b.ravel()
        return (self._trend(a, b) @ self.tc + self.gp.predict(np.c_[a, b])).reshape(shp)


def make(name):
    """Short names used in the experiment tables."""
    if name.startswith("poly"):                       # "poly43" -> degree 4 in u, 3 in v
        return PolyModel(int(name[4]), int(name[5]))
    return {"phys3": lambda: PhysModel(("theta", "roll", "y0")),                 # no distortion, CAD lens height
            "phys4": lambda: PhysModel(("Zc", "theta", "roll", "y0")),            # no distortion
            "phys4k": lambda: PhysModel(("k1", "theta", "roll", "y0")),           # distortion, CAD lens height
            "phys5": lambda: PhysModel(("k1", "Zc", "theta", "roll", "y0")),
            "phys7": lambda: PhysModel(("cx", "cy", "k1", "Zc", "theta", "roll", "y0")),
            "spline": SplineModel, "gp": GPModel}[name]()
