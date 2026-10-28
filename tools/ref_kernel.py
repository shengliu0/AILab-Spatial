#!/usr/bin/env python3
"""Reference solutions for core/kernel (the kernel-trick demo), computed with scipy — an independent QP solver.

Solves the soft-margin SVM DUAL
    min_a  1/2 a^T Q a - sum(a)   s.t.  y^T a = 0,  0 <= a_i <= C,   Q_ij = y_i y_j x_i.x_j
with scipy.optimize.minimize(method="trust-constr") from a feasible start, then
    w = sum a_i y_i x_i,   b = mean over free SVs (0 < a_i < C) of (y_i - w.x_i).
The primal (w, b) is unique, so it is what KernelSvmTest compares (to 1e-5); the numbers below are pasted there.

Cases
  A  four hand-checkable points, hard margin (C = 1e3): w = (1, 0), b = 0, margin 2.
  B  ten literal 2-D points, soft margin C = 1 (overlapping classes: some a_i hit C).
  C  the demo's RINGS data set at noise level 1, LIFTED by z = x^2 + y^2, C = 1 — the generator below
     re-implements KRng / KernelData.make bit for bit (64-bit LCG, Box-Muller), so Kotlin and Python
     train on the same 40 points.
Run:  python3 tools/ref_kernel.py
"""
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

M64 = (1 << 64) - 1
A_ = 6364136223846793005
C_ = 1442695040888963407


class KRng:
    def __init__(self, seed):
        self.s = (seed * A_ + C_) & M64

    def next(self):
        self.s = (self.s * A_ + C_) & M64
        return (self.s >> 11) / float(1 << 53)

    def gauss(self):
        u = max(self.next(), 1e-12)
        v = self.next()
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


def rings(noise_level, seed=1992):
    sig = [0.0, 0.06, 0.13][noise_level]
    r = KRng(seed + 0 * 7919)
    nr = KRng(seed * 31 + 17)
    X, Y = [], []
    for i in range(40):
        c = 1 if i % 2 == 0 else -1
        th = 2 * math.pi * r.next()
        rad = 0.42 * math.sqrt(r.next()) if c > 0 else 0.70 + 0.25 * r.next()
        px, py = rad * math.cos(th), rad * math.sin(th)
        gx, gy = nr.gauss(), nr.gauss()
        px += sig * gx
        py += sig * gy
        rr = math.hypot(px, py)
        if rr > 0.97:
            px *= 0.97 / rr
            py *= 0.97 / rr
        X.append([px, py])
        Y.append(c)
    return np.array(X), np.array(Y, dtype=float)


def solve(X, y, C):
    n = len(y)
    Q = (y[:, None] * y[None, :]) * (X @ X.T)
    f = lambda a: 0.5 * a @ Q @ a - a.sum()
    g = lambda a: Q @ a - 1.0
    h = lambda a: Q
    res = minimize(f, np.zeros(n), jac=g, hess=h, method="trust-constr",
                   constraints=[LinearConstraint(y[None, :], 0.0, 0.0)], bounds=Bounds(0.0, C),
                   options=dict(gtol=1e-13, xtol=1e-15, maxiter=20000))
    a = np.clip(res.x, 0, C)
    w = (a * y) @ X
    tol = 1e-5 * max(1.0, a.max())
    free = [i for i in range(n) if tol < a[i] < C - tol]
    b = np.mean([y[i] - X[i] @ w for i in free])
    return a, w, b, f(a)


def report(name, X, y, C):
    a, w, b, obj = solve(X, y, C)
    sv = [i for i in range(len(y)) if a[i] > 1e-6 * max(1, a.max())]
    acc = np.mean(np.sign(X @ w + b) == y)
    print(f"{name}: C={C}  w = [{', '.join(f'{v:.8f}' for v in w)}]  b = {b:.8f}  dual = {obj:.8f}  "
          f"margin = {2 / np.linalg.norm(w):.6f}  #SV = {len(sv)}  acc = {acc:.4f}")


if __name__ == "__main__":
    XA = np.array([[1, 0], [2, 1], [-1, 0], [-2, -1]], float); yA = np.array([1, 1, -1, -1], float)
    report("A", XA, yA, 1e3)
    XB = np.array([[0.2, 0.9], [0.5, 0.4], [0.9, 0.7], [-0.1, 0.3], [0.6, -0.2],
                   [-0.5, -0.4], [-0.8, 0.1], [0.1, -0.7], [-0.3, 0.5], [0.3, 0.1]], float)
    yB = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], float)
    report("B", XB, yB, 1.0)
    X2, yC = rings(1)
    XC = np.c_[X2, (X2 ** 2).sum(1)]
    print("C points[0..1] =", X2[0].tolist(), X2[1].tolist())
    report("C", XC, yC, 1.0)
