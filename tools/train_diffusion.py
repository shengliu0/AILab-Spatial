#!/usr/bin/env python3
"""
train_diffusion.py — the mini diffusion model behind the AI Lab demo "扩散模型" (id: diffusion).

DDPM (Ho, Jain & Abbeel 2020) on three SYNTHETIC 2-D point clouds (no downloads):
  0 spiral    1.5 turns, r proportional to theta, Gaussian jitter
  1 moons     the classic "two moons", Gaussian jitter
  2 glyph 光  the character rendered with PIL from a macOS system font (STHeiti Medium); points are uniform
              samples of the ink pixels with sub-pixel jitter
Each cloud is centred and scaled to unit RMS radius per axis (std 1), like normalised image data.

FORWARD PROCESS (variance preserving, discrete t = 0..T, T = 200):
  q(x_t | x_0) = N( sqrt(abar_t) x_0, (1 - abar_t) I ),   abar_t = 1 - (1 - ABAR_T) (t/T)^2
  i.e. the noise level sigma_t = sqrt(1 - abar_t) grows LINEARLY in t (sigma_T = sqrt(1 - 1e-3) = 0.9995),
  beta_t = 1 - abar_t / abar_{t-1}. (A schedule choice for this demo — it spreads the "melting" evenly over
  the eight time slices the app draws; Ho et al. used a linear-in-beta schedule, Nichol & Dhariwal 2021 a cosine.)

MODEL: one CONDITIONAL MLP eps_hat(x_t, t, dataset) (the dataset is a one-hot label; one network for all three):
  f    = [t/T, sin(pi 2^k t/T), cos(pi 2^k t/T) for k = 0..K-1]            (float64, then float32)
  e    = SiLU(A2 SiLU(A1 [f; onehot(d)] + a1) + a2)                          (E dims; per step + dataset only)
  h1   = SiLU(W1 x + U1 e + b1)                                              (H)
  h2   = SiLU(W2 h1 + U2 e + b2)                                             (H)
  h3   = SiLU(W3 h2 + U3 e + b3)                                             (H)
  eps  = Wo h3 + bo                                                          (2)
  The U_l e + b_l terms depend only on (t, dataset): the app computes them ONCE per sampling step, so a point
  costs only 2H + 2H^2 + 2H multiply-adds.
LOSS: the DDPM "simple" loss E || eps - eps_hat(sqrt(abar_t) x0 + sqrt(1-abar_t) eps, t) ||^2, t ~ U{1..T}.

SAMPLERS (both on a respaced grid tau = round(linspace(T, 0, S+1)), S = 10 / 50 / 200 steps):
  DDPM (ancestral, posterior variance):  x' = (x - b/sqrt(1-a) eps_hat)/sqrt(1-b) + sqrt(b (1-a')/(1-a)) z,
                                           a = abar_tau_i, a' = abar_tau_{i+1}, b = 1 - a/a'
  DDIM (Song, Meng & Ermon 2020, eta=0):  x0_hat = (x - sqrt(1-a) eps_hat)/sqrt(a);  x' = sqrt(a') x0_hat + sqrt(1-a') eps_hat

QUALITY METRIC (what the app's challenge uses; core/diffusion/Quality.kt implements the same):
  precision P = fraction of generated points within EPS of the reference cloud (NREF clean samples);
  coverage  C = fraction of the first NCOV reference points that have a generated point within EPS;
  quality   Q = min(P, C)   ("on the shape" AND "the whole shape").  EPS = 0.15 (data std = 1).

OUTPUTS (app/src/main/assets/diffusion/), all little-endian:
  model.bin   char[4] "DIF1", int32 T, K, E, H, D, NREF, NCOV, then float32:
                abar[T+1]
                A1[E x (1+2K+D)], a1[E], A2[E x E], a2[E]          (row-major: A[j*in + i])
                W1[H x 2], U1[H x E], b1[H]
                W2[H x H], U2[H x E], b2[H]
                W3[H x H], U3[H x E], b3[H]
                Wo[2 x H], bo[2]
                per dataset d: ref[NREF x 2] (x, y), hue[NREF] (0..1 colour key along the shape)
  testvec.bin int32 NV, then per vector: int32 d, int32 t, float32 x, y, eps_x, eps_y      (PyTorch float32)
              int32 NS, int32 S, then per dataset: NS x [x_T(2) ... ] start noise, then NS x final x_0(2)
              of the DDIM sampler with S steps (end-to-end check: 50 network calls per point)
Deterministic: fixed seeds, CPU float32 training.
Run:  python3 tools/train_diffusion.py      (~2-3 minutes on an M-series Mac)
"""
import math, os, struct, sys, time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "app", "src", "main", "assets", "diffusion")

T = 200
ABAR_T = 1e-3
K = 6            # time-feature frequencies
E = 64           # conditioning embedding width
H = int(os.environ.get("DIF_H", "64"))
D = 3
NREF = 2000
NCOV = 400
EPS_Q = 0.15
NAMES = ["spiral", "moons", "glyph"]
GLYPH_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
STEPS = int(os.environ.get("DIF_STEPS", "40000"))

torch.set_num_threads(8)


# ------------------------------------------------------------------------------------------- data
def normalise(p):
    p = p - p.mean(0, keepdims=True)
    return p / p.std()          # one scale for both axes (keeps the shape)


def spiral(n, rng):
    u = rng.random(n)
    th = 0.6 * math.pi + 3.0 * math.pi * np.sqrt(u)     # sqrt: uniform density along the arc (r ~ theta)
    r = th
    p = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    hue = (th - 0.6 * math.pi) / (3.0 * math.pi)
    return p, hue


def moons(n, rng):
    m = rng.random(n) < 0.5
    a = rng.random(n) * math.pi
    p = np.where(m[:, None], np.stack([np.cos(a), np.sin(a)], 1), np.stack([1 - np.cos(a), 0.5 - np.sin(a)], 1))
    hue = np.where(m, 0.5 * a / math.pi, 0.5 + 0.5 * a / math.pi)
    return p, hue


_GLYPH = None
def glyph_mask():
    global _GLYPH
    if _GLYPH is None:
        S = 256
        img = Image.new("L", (S, S), 0)
        dr = ImageDraw.Draw(img)
        font = ImageFont.truetype(GLYPH_FONT, 220)
        bb = dr.textbbox((0, 0), "光", font=font)
        dr.text(((S - (bb[2] - bb[0])) / 2 - bb[0], (S - (bb[3] - bb[1])) / 2 - bb[1]), "光", fill=255, font=font)
        m = np.array(img) > 127
        ys, xs = np.nonzero(m)
        _GLYPH = np.stack([xs, -ys], 1).astype(np.float64)   # y up
    return _GLYPH


def glyph(n, rng):
    px = glyph_mask()
    k = rng.integers(0, len(px), n)
    p = px[k] + rng.random((n, 2)) - 0.5
    # colour key: left -> right sweep with a little vertical tilt
    lo, hi = px[:, 0].min(), px[:, 0].max()
    hue = np.clip((p[:, 0] - lo) / (hi - lo) * 0.85 + 0.15 * (p[:, 1] - px[:, 1].min()) / (px[:, 1].max() - px[:, 1].min()), 0, 1)
    return p, hue


JITTER = [0.035, 0.05, 0.0]


def raw(d, n, rng):
    return [spiral, moons, glyph][d](n, rng)


# normalisation constants per dataset (from a big sample, fixed seed)
NORM = []
for d in range(D):
    p, _ = raw(d, 200000, np.random.default_rng(1000 + d))
    NORM.append((p.mean(0), (p - p.mean(0)).std()))


def sample(d, n, rng):
    p, hue = raw(d, n, rng)
    mu, s = NORM[d]
    p = (p - mu) / s
    p = p + rng.normal(0, JITTER[d], p.shape)
    return p.astype(np.float32), hue.astype(np.float32)


# ------------------------------------------------------------------------------------------- schedule
t_all = np.arange(T + 1, dtype=np.float64)
SCHED = os.environ.get("DIF_SCHED", "sigma_linear")
if SCHED == "sigma_linear":
    abar = 1.0 - (1.0 - ABAR_T) * (t_all / T) ** 2
elif SCHED == "cosine":
    f = np.cos((t_all / T + 0.008) / 1.008 * math.pi / 2) ** 2
    abar = f / f[0]
    beta = np.clip(1 - abar[1:] / abar[:-1], 0, 0.999)
    abar = np.concatenate([[1.0], np.cumprod(1 - beta)])
elif SCHED == "linear_beta":
    beta = np.linspace(1e-4 * 1000 / T, 0.02 * 1000 / T, T)
    abar = np.concatenate([[1.0], np.cumprod(1 - beta)])
abar_f = abar.astype(np.float32)
FREQ = np.array([math.pi * 2 ** k for k in range(K)])


def features(t):
    """t: int array -> float32 [n, 1+2K] computed in float64 (the Kotlin side does the same)."""
    s = np.asarray(t, dtype=np.float64)[:, None] / T
    return np.concatenate([s, np.sin(s * FREQ), np.cos(s * FREQ)], 1).astype(np.float32)


# ------------------------------------------------------------------------------------------- model
class Eps(nn.Module):
    def __init__(self):
        super().__init__()
        nin = 1 + 2 * K + D
        self.A1 = nn.Linear(nin, E); self.A2 = nn.Linear(E, E)
        self.W1 = nn.Linear(2, H, bias=False); self.U1 = nn.Linear(E, H)
        self.W2 = nn.Linear(H, H, bias=False); self.U2 = nn.Linear(E, H)
        self.W3 = nn.Linear(H, H, bias=False); self.U3 = nn.Linear(E, H)
        self.Wo = nn.Linear(H, 2)
        nn.init.zeros_(self.Wo.weight); nn.init.zeros_(self.Wo.bias)

    def cond(self, f, onehot):
        a = nn.functional.silu
        return a(self.A2(a(self.A1(torch.cat([f, onehot], 1)))))

    def forward(self, x, f, onehot):
        a = nn.functional.silu
        e = self.cond(f, onehot)
        h = a(self.W1(x) + self.U1(e))
        h = a(self.W2(h) + self.U2(e))
        h = a(self.W3(h) + self.U3(e))
        return self.Wo(h)


def eps_hat(net, x, t, d):
    """x float32 [n,2] tensor, t int array [n], d int array [n]."""
    f = torch.from_numpy(features(t))
    oh = torch.nn.functional.one_hot(torch.as_tensor(np.asarray(d), dtype=torch.long), D).float()
    return net(x, f, oh)


# ------------------------------------------------------------------------------------------- samplers
def grid(S):
    return np.round(np.linspace(T, 0, S + 1)).astype(int)


@torch.no_grad()
def sample_rev(net, d, xT, S, ddim, gen):
    x = xT.clone()
    tau = grid(S)
    for i in range(S):
        t, tp = tau[i], tau[i + 1]
        a, ap = float(abar[t]), float(abar[tp])
        e = eps_hat(net, x, np.full(len(x), t), np.full(len(x), d))
        if ddim:
            x0 = (x - math.sqrt(1 - a) * e) / math.sqrt(a)
            x = math.sqrt(ap) * x0 + math.sqrt(1 - ap) * e
        else:
            b = 1 - a / ap
            mean = (x - b / math.sqrt(1 - a) * e) / math.sqrt(1 - b)
            var = b * (1 - ap) / (1 - a)
            x = mean + (math.sqrt(var) * torch.randn(x.shape, generator=gen) if var > 0 else 0)
    return x


def quality(gen_pts, ref):
    g = np.asarray(gen_pts, dtype=np.float64); r = np.asarray(ref, dtype=np.float64)
    d2 = ((g[:, None, :] - r[None, :, :]) ** 2).sum(-1)
    P = float((d2.min(1) < EPS_Q ** 2).mean())
    C = float((d2[:, :NCOV].min(0) < EPS_Q ** 2).mean())
    return min(P, C), P, C


# ------------------------------------------------------------------------------------------- train
REFS = [sample(d, NREF, np.random.default_rng(500 + d)) for d in range(D)]


def train():
    torch.manual_seed(20200619)       # DDPM arXiv date
    rng = np.random.default_rng(7)
    net = Eps()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.05)
    B = 3072
    # a big pool of clean samples per dataset (resampled every 5k steps)
    pool = None
    t0 = time.time()
    sqa = torch.from_numpy(np.sqrt(abar_f)); sq1 = torch.from_numpy(np.sqrt(1 - abar_f))
    feats = torch.from_numpy(features(np.arange(T + 1)))
    gen = torch.Generator().manual_seed(11)
    ema = None
    for step in range(STEPS):
        if step % 5000 == 0:
            pool = [torch.from_numpy(sample(d, 200000, rng)[0]) for d in range(D)]
        d = torch.randint(0, D, (B,), generator=gen)
        idx = torch.randint(0, 200000, (B,), generator=gen)
        x0 = torch.stack([pool[0][idx], pool[1][idx], pool[2][idx]], 0)[d, torch.arange(B)]
        t = torch.randint(1, T + 1, (B,), generator=gen)
        eps = torch.randn((B, 2), generator=gen)
        xt = sqa[t, None] * x0 + sq1[t, None] * eps
        pred = net(xt, feats[t], nn.functional.one_hot(d, D).float())
        loss = ((pred - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()
        if step % 5000 == 0 or step == STEPS - 1:
            print(f"step {step:6d}  loss {ema:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    return net


def main():
    os.makedirs(OUT, exist_ok=True)
    refs = REFS
    net = train()

    # ---- metrics (PyTorch samplers, N = 300 points, fixed seeds)
    print(f"H={H}: quality Q = min(precision, coverage), EPS={EPS_Q}, N=300")
    g2 = torch.Generator().manual_seed(3)
    for d in range(D):
        true = sample(d, 300, np.random.default_rng(900 + d))[0]
        q0 = quality(true, refs[d][0])
        noise = np.random.default_rng(1).normal(size=(300, 2))
        qn = quality(noise, refs[d][0])
        row = [f"{NAMES[d]:7s} true {q0[0]*100:5.1f}%  noise {qn[0]*100:5.1f}%"]
        for ddim in (False, True):
            for S in (2, 5, 10, 50, 200):
                xT = torch.randn((300, 2), generator=g2)
                x = sample_rev(net, d, xT, S, ddim, g2).numpy()
                q = quality(x, refs[d][0])
                row.append(f"{'DDIM' if ddim else 'DDPM'}{S:<3d} {q[0]*100:5.1f}% (P{q[1]*100:3.0f} C{q[2]*100:3.0f})")
        print("  " + " | ".join(row), flush=True)

    # ---- export
    def f32(a):
        return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()
    with open(os.path.join(OUT, "model.bin"), "wb") as fh:
        fh.write(b"DIF1"); fh.write(struct.pack("<7i", T, K, E, H, D, NREF, NCOV))
        fh.write(f32(abar_f))
        fh.write(f32(net.A1.weight.detach())); fh.write(f32(net.A1.bias.detach()))
        fh.write(f32(net.A2.weight.detach())); fh.write(f32(net.A2.bias.detach()))
        for W, U in ((net.W1, net.U1), (net.W2, net.U2), (net.W3, net.U3)):
            fh.write(f32(W.weight.detach())); fh.write(f32(U.weight.detach())); fh.write(f32(U.bias.detach()))
        fh.write(f32(net.Wo.weight.detach())); fh.write(f32(net.Wo.bias.detach()))
        for d in range(D):
            fh.write(f32(refs[d][0])); fh.write(f32(refs[d][1]))
    rv = np.random.default_rng(42)
    NV = 48
    with open(os.path.join(OUT, "testvec.bin"), "wb") as fh:
        fh.write(struct.pack("<i", NV))
        dd = rv.integers(0, D, NV); tt = rv.integers(1, T + 1, NV); tt[:3] = [1, T // 2, T]
        xx = rv.normal(0, 1.3, (NV, 2)).astype(np.float32)
        with torch.no_grad():
            ee = eps_hat(net, torch.from_numpy(xx), tt, dd).numpy()
        for i in range(NV):
            fh.write(struct.pack("<2i", int(dd[i]), int(tt[i]))); fh.write(f32([xx[i, 0], xx[i, 1], ee[i, 0], ee[i, 1]]))
        NS, S = 16, 50
        fh.write(struct.pack("<2i", NS, S))
        g3 = torch.Generator().manual_seed(5)
        for d in range(D):
            xT = torch.randn((NS, 2), generator=g3)
            x0 = sample_rev(net, d, xT, S, True, g3)
            fh.write(f32(xT)); fh.write(f32(x0))
    for f in ("model.bin", "testvec.bin"):
        print(f"wrote {f}: {os.path.getsize(os.path.join(OUT, f))} bytes")


if __name__ == "__main__":
    main()
