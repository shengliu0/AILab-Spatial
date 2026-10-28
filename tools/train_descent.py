#!/usr/bin/env python3
"""
train_descent.py — reference numbers for the "梯度下降 · 损失曲面 / Gradient Descent" demo (core/descent).

This demo ships NO weights: the optimisers themselves are the subject. The script pins down the numbers the
pure-Kotlin core must reproduce, using torch.optim in float64 as the independent implementation:

  1. the synthetic dataset of the "real model" landscape (logistic regression, 2 weights, no bias):
     40 points, two Gaussian blobs (seed 1847), feature 2 deliberately NOT normalised (x 2.5) so the loss
     surface is a ravine — printed as a Kotlin literal (values rounded to 2 decimals, used as-is by both sides);
  2. its minimum w*, L* (Newton in float64, cross-checked with torch LBFGS);
  3. reference trajectories of torch.optim.SGD, SGD(momentum=0.9) and Adam(0.9, 0.999, eps=1e-8) on the
     Rosenbrock, Himmelblau, bowl, saddle and logistic landscapes (positions after steps 1, 2, 3, 10, 50).

Run:  python3 tools/train_descent.py      (deterministic; prints everything; writes nothing)
The numbers are pasted into app/src/test/java/com/bytedance/ailab/core/descent/DescentCoreTest.kt.
"""
import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------- dataset (logistic regression)
rng = np.random.default_rng(1847)
N_PER = 20
MU = np.array([1.0, 0.4])
Xp = rng.normal(MU, 0.9, (N_PER, 2))
Xn = rng.normal(-MU, 0.9, (N_PER, 2))
X = np.vstack([Xp, Xn])
X[:, 1] *= 2.5                       # un-normalised second feature -> a ravine-shaped loss
X = np.round(X, 2)
Y = np.array([1.0] * N_PER + [-1.0] * N_PER)
LAM = 0.05                           # L2 penalty (keeps the minimum finite and interior)

print("// ---- Kotlin literal: LogisticData (x1, x2, label) ----")
rows = [f"{X[i,0]:.2f}, {X[i,1]:.2f}, {int(Y[i]):d}" for i in range(len(Y))]
for k in range(0, len(rows), 4):
    print("        " + ",   ".join(rows[k:k + 4]) + ",")

Xt = torch.tensor(X); Yt = torch.tensor(Y)


def f_logistic(p):
    z = Yt * (Xt @ p)
    return torch.nn.functional.softplus(-z).mean() + LAM / 2 * (p * p).sum()


# Newton for the minimum (exact Hessian), then LBFGS as a cross-check
w = np.zeros(2)
for _ in range(60):
    z = Y * (X @ w); s = 1 / (1 + np.exp(z))
    g = -(X * (Y * s)[:, None]).mean(0) + LAM * w
    H = (X.T * (s * (1 - s))) @ X / len(Y) + LAM * np.eye(2)
    w = w - np.linalg.solve(H, g)
p = torch.zeros(2, requires_grad=True)
opt = torch.optim.LBFGS([p], lr=1, max_iter=500, tolerance_grad=1e-14, tolerance_change=1e-16, line_search_fn="strong_wolfe")
def closure():
    opt.zero_grad(); l = f_logistic(p); l.backward(); return l
opt.step(closure)
Lstar = f_logistic(torch.tensor(w)).item()
print(f"\nlogistic minimum (Newton): w* = ({w[0]:.12f}, {w[1]:.12f})  L* = {Lstar:.12f}")
print(f"logistic minimum (LBFGS) : w* = ({p[0].item():.12f}, {p[1].item():.12f})  |dw| = {np.linalg.norm(p.detach().numpy() - w):.2e}")
print(f"Hessian eigenvalues at w*: {np.linalg.eigvalsh(H)}  -> 2/lambda_max = {2 / np.linalg.eigvalsh(H).max():.4f}")
print(f"training accuracy of w*: {np.mean(np.sign(X @ w) == Y):.3f}")

# ---------------------------------------------------------------- the other landscapes (same formulas as core/descent)
PHI = math.radians(70.0)
C, S = math.cos(PHI), math.sin(PHI)


def f_bowl(p):
    u = C * p[0] + S * p[1]; v = -S * p[0] + C * p[1]
    return 0.5 * (1.0 * u * u + 10.0 * v * v)


def f_rosen(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def f_himmel(p):
    return (p[0] ** 2 + p[1] - 11) ** 2 + (p[0] + p[1] ** 2 - 7) ** 2


def f_saddle(p):
    return p[0] ** 2 + 0.25 * (p[1] ** 2 - 1) ** 2


def trajectory(f, start, kind, lr, steps=50):
    p = torch.tensor(start, requires_grad=True)
    if kind == "sgd":
        o = torch.optim.SGD([p], lr=lr)
    elif kind == "momentum":
        o = torch.optim.SGD([p], lr=lr, momentum=0.9)
    else:
        o = torch.optim.Adam([p], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    out = {}
    for t in range(1, steps + 1):
        o.zero_grad(); f(p).backward(); o.step()
        if t in (1, 2, 3, 10, 50):
            out[t] = (p[0].item(), p[1].item())
    return out


CASES = [
    ("ROSENBROCK", f_rosen, (-1.0, 0.7), "sgd", 0.002),
    ("ROSENBROCK", f_rosen, (-1.0, 0.7), "momentum", 0.003),
    ("ROSENBROCK", f_rosen, (-1.0, 0.7), "adam", 0.05),
    ("HIMMELBLAU", f_himmel, (0.0, 0.0), "sgd", 0.01),
    ("HIMMELBLAU", f_himmel, (-0.5, -1.5), "adam", 0.1),
    ("BOWL", f_bowl, (-1.2, -1.7), "momentum", 0.02),
    ("SADDLE", f_saddle, (-1.8, 0.02), "adam", 0.05),
    ("LOGISTIC", f_logistic, (-3.0, 3.0), "momentum", 0.1),
    ("LOGISTIC", f_logistic, (-3.0, 3.0), "adam", 0.1),
]
print("\n// ---- Kotlin literal: torch.optim reference trajectories (float64) ----")
for name, f, start, kind, lr in CASES:
    tr = trajectory(f, start, kind, lr)
    pts = ", ".join(f"{t} to doubleArrayOf({x:.15g}, {y:.15g})" for t, (x, y) in tr.items())
    print(f'        Ref(Surfaces.{name}, doubleArrayOf({start[0]}, {start[1]}), Kind.{kind.upper()}, {lr}, mapOf({pts})),')

# ---------------------------------------------------------------- the challenge, measured (no noise)
def first_hit(f, start, kind, lr, target=1e-3, steps=200, box=(-2, 2, -1, 3)):
    p = torch.tensor(start, requires_grad=True)
    o = {"sgd": lambda: torch.optim.SGD([p], lr=lr), "momentum": lambda: torch.optim.SGD([p], lr=lr, momentum=0.9),
         "adam": lambda: torch.optim.Adam([p], lr=lr)}[kind]()
    for t in range(1, steps + 1):
        o.zero_grad(); f(p).backward(); o.step()
        x, y = p[0].item(), p[1].item()
        if not (box[0] <= x <= box[1] and box[2] <= y <= box[3]) or not math.isfinite(x + y):
            return f"diverged@{t}"
        if f(p).item() < target:
            return t
    return None

SNAP = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
print("\nchallenge (Rosenbrock from (-1, 0.7), loss < 1e-3 within 200 steps), every slider value:")
for kind in ("sgd", "momentum", "adam"):
    hits = [(lr, first_hit(f_rosen, (-1.0, 0.7), kind, lr)) for lr in SNAP]
    print(f"  {kind:9s} solved at: {[(lr, h) for lr, h in hits if isinstance(h, int)]}")
