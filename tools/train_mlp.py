#!/usr/bin/env python3
"""
train_mlp.py — trains the 784 -> 64 -> 32 -> 10 MLP behind the AI Lab demo "神经网络 · 手写数字" (id: mlp).

SYNTHETIC DATA ONLY (no downloads):
  * FONT digits — 0-9 rendered with PIL from macOS system fonts (/System/Library/Fonts), many faces/weights,
    each copy heavily augmented: rotation ±15°, shear, anisotropic scale, stroke thickness (grey dilation /
    erosion), elastic distortion (Simard, Steinkraus & Platt 2003), then MNIST-normalised exactly as LeCun et
    al. describe MNIST: ink bounding box scaled to fit a 20x20 box preserving aspect ratio (anti-aliased), then
    translated so the centre of mass sits at the centre of the 28x28 field; ±1 px jitter afterwards.
  * PEN digits — hand-written stroke templates (2-3 variants per digit: open/closed 4, crossed 7, flag 1, ...),
    randomly perturbed (control-point jitter, rotation ±15°, shear, aspect, hand tremor, offsets between strokes
    as produced by the app's gaze-snapped stroke starts, pen width, elastic warp of the points) and rasterised
    with the SAME vector rasteriser as the app (core/mlp/StrokeRaster.kt: bbox -> 20x20, centre of mass ->
    (14, 14), two passes). So what the user draws in the headset is in-distribution.
  * Noise: ~20 % of the examples get additive Gaussian pixel noise (sigma 0.05-0.35) — the demo's "加噪声"
    switch (sigma 0.30) is therefore partly something the network has seen; the UI says so.

MODEL: x in [0,1]^784 -> Linear(784,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,10) -> softmax(z / T).
T (temperature, Guo et al. 2017) is fitted on held-out data so the displayed confidence is calibrated on the
synthetic distribution (it is NOT a guarantee on arbitrary scribbles: the net has no "not a digit" class).

OUTPUTS (app/src/main/assets/mlp/), all little-endian:
  mlp_w.bin   float32: [784, 64, 32, 10] (sizes as floats), W1 (64 x 784, row-major, W1[j*784+i]), b1[64],
              W2 (32 x 64), b2[32], W3 (10 x 32), b3[10], act_scale1, act_scale2, temperature
  ref.bin     float32: [N], then per vector: x[784], h1[64], h2[32], logits[10], probs[10] (probs = softmax(z/T));
              vectors 0-9 = the 10 sample digits rasterised by THIS script (Kotlin must reproduce x and outputs)
  samples.bin float32: [10], per digit: [nStrokes], per stroke: [nPts, x0, y0, x1, y1, ...] (pad units 0..1, y down)
  heldout.bin int32 N, then N*784 uint8 pixels (round(255 x)), then N uint8 labels — a 1000-example slice of
              the held-out set, quantised; this script prints the accuracy on exactly these bytes
  glyphs.bin  uint8: 10 x 48 x 48 glyph alpha (row 0 = top) for the output-bar labels

Deterministic: fixed seeds, CPU training, per-example RNG seeded by (split, index).
Run:  python3 tools/train_mlp.py          (about 2-4 minutes on an M-series Mac)
"""
import math, os, struct, sys, time
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("MLP_OUT", os.path.join(ROOT, "app", "src", "main", "assets", "mlp"))   # override to test reproducibility

# ------------------------------------------------------------------ the app's rasteriser (keep in sync with
# core/mlp/StrokeRaster.kt — StrokeRasterTest checks Kotlin == these numbers to 1e-4)
GRID = 28
BOX = 20.0          # MNIST: ink fits a 20x20 box
CENTRE = 14.0       # centre of mass goes to the field centre (pixel centres at i + 0.5)
PEN_R = 1.3         # app pen radius in output pixels (stroke ~2.6 px wide, like MNIST)
EDGE = 1.5          # anti-aliasing ramp width (px)
MIN_SIDE = 0.04     # pad units: a dot / tiny scribble is not blown up to 20 px
COM_PASSES = 2

_PC = np.stack(np.meshgrid(np.arange(GRID) + 0.5, np.arange(GRID) + 0.5, indexing="xy"), -1).reshape(-1, 2)  # (784,2) x,y


def _segments(strokes):
    a, b = [], []
    for s in strokes:
        s = np.asarray(s, dtype=np.float64)
        if len(s) == 0:
            continue
        if len(s) == 1:
            a.append(s[0]); b.append(s[0])
        else:
            a.extend(s[:-1]); b.extend(s[1:])
    return np.array(a, dtype=np.float64).reshape(-1, 2), np.array(b, dtype=np.float64).reshape(-1, 2)


def _render(A, B, pen_r):
    ab = B - A                                   # (S,2)
    L2 = (ab * ab).sum(1)                        # (S,)
    ap = _PC[:, None, :] - A[None, :, :]         # (784,S,2)
    t = np.where(L2[None, :] > 1e-12, (ap * ab[None]).sum(2) / np.maximum(L2[None, :], 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cl = A[None] + t[..., None] * ab[None]
    d = np.sqrt(((_PC[:, None, :] - cl) ** 2).sum(2))
    cov = np.clip((pen_r + EDGE / 2 - d) / EDGE, 0.0, 1.0)
    return cov.max(1) if cov.shape[1] else np.zeros(GRID * GRID)


def raster_strokes(strokes, pen_r=PEN_R):
    """Pad strokes (pad units, y down) -> 28x28 in [0,1], MNIST-normalised. Mirrors StrokeRaster.rasterise."""
    A, B = _segments(strokes)
    if len(A) == 0:
        return np.zeros(GRID * GRID)
    pts = np.concatenate([A, B])
    lo = pts.min(0); hi = pts.max(0)
    side = max(hi[0] - lo[0], hi[1] - lo[1], MIN_SIDE)
    s = (BOX - 2 * pen_r) / side
    c = (lo + hi) / 2
    off = np.array([CENTRE, CENTRE])
    img = _render((A - c) * s + off, (B - c) * s + off, pen_r)
    for _ in range(COM_PASSES):
        m = img.sum()
        if m <= 1e-9:
            break
        cx = (img * _PC[:, 0]).sum() / m
        cy = (img * _PC[:, 1]).sum() / m
        off = off + np.array([CENTRE - cx, CENTRE - cy])
        img = _render((A - c) * s + off, (B - c) * s + off, pen_r)
    return img


# ------------------------------------------------------------------ pen templates (unit box, y DOWN)
# A stroke is a list of PIECES; each piece is smoothed through its control points (Catmull-Rom); pieces meet at
# sharp corners. Variant 0 of every digit is a single stroke: those are the app's sample digits.

def ell(cx, cy, rx, ry, a0, a1, n=9):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]

TEMPLATES = {
    0: [
        [[ell(0.5, 0.5, 0.34, 0.46, -95, -95 - 375, 13)]],
        [[ell(0.5, 0.5, 0.30, 0.47, -70, 270, 12)]],
        [[ell(0.52, 0.5, 0.36, 0.44, -100, -100 - 350, 12)]],
    ],
    1: [
        [[[(0.28, 0.24), (0.56, 0.03)], [(0.56, 0.03), (0.54, 0.97)]]],
        [[[(0.52, 0.03), (0.47, 0.97)]]],
        [[[(0.30, 0.22), (0.55, 0.03)], [(0.55, 0.03), (0.55, 0.96)]], [[(0.28, 0.97), (0.82, 0.97)]]],
    ],
    2: [
        [[[(0.12, 0.26), (0.30, 0.06), (0.58, 0.03), (0.82, 0.17), (0.80, 0.40), (0.55, 0.64), (0.30, 0.82), (0.10, 0.96)],
          [(0.10, 0.96), (0.92, 0.95)]]],
        [[[(0.15, 0.22), (0.45, 0.02), (0.80, 0.15), (0.75, 0.42), (0.40, 0.75), (0.18, 0.95), (0.10, 0.84), (0.30, 0.80),
           (0.60, 0.92), (0.92, 0.95)]]],
        [[[(0.10, 0.20), (0.50, 0.02), (0.85, 0.25), (0.60, 0.60), (0.08, 0.96)], [(0.08, 0.96), (0.95, 0.94)]]],
    ],
    3: [
        [[[(0.14, 0.16), (0.42, 0.02), (0.76, 0.12), (0.76, 0.33), (0.44, 0.47)],
          [(0.44, 0.47), (0.80, 0.58), (0.86, 0.80), (0.62, 0.97), (0.32, 0.97), (0.10, 0.84)]]],
        [[[(0.14, 0.05), (0.86, 0.05)], [(0.86, 0.05), (0.42, 0.44)],
          [(0.42, 0.44), (0.80, 0.55), (0.86, 0.80), (0.56, 0.98), (0.14, 0.88)]]],
        [[[(0.18, 0.12), (0.50, 0.02), (0.78, 0.20), (0.50, 0.46), (0.82, 0.70), (0.55, 0.98), (0.15, 0.85)]]],
    ],
    4: [
        [[[(0.70, 0.98), (0.70, 0.02)], [(0.70, 0.02), (0.05, 0.68)], [(0.05, 0.68), (0.95, 0.68)]]],
        [[[(0.24, 0.02), (0.12, 0.62)], [(0.12, 0.62), (0.92, 0.62)]], [[(0.68, 0.22), (0.68, 0.98)]]],
        [[[(0.62, 0.02), (0.05, 0.66)], [(0.05, 0.66), (0.95, 0.66)]], [[(0.70, 0.30), (0.70, 0.98)]]],
    ],
    5: [
        [[[(0.86, 0.04), (0.24, 0.04)], [(0.24, 0.04), (0.18, 0.46)],
          [(0.18, 0.46), (0.48, 0.38), (0.78, 0.50), (0.86, 0.72), (0.66, 0.95), (0.36, 0.97), (0.10, 0.85)]]],
        [[[(0.26, 0.05), (0.20, 0.46)], [(0.20, 0.46), (0.50, 0.38), (0.82, 0.55), (0.80, 0.85), (0.45, 0.98), (0.12, 0.86)]],
         [[(0.26, 0.05), (0.88, 0.03)]]],
        [[[(0.80, 0.03), (0.28, 0.06)], [(0.28, 0.06), (0.22, 0.42)], [(0.22, 0.42), (0.65, 0.42), (0.85, 0.70), (0.55, 0.97), (0.15, 0.90)]]],
    ],
    6: [
        [[[(0.76, 0.04), (0.46, 0.18), (0.24, 0.46), (0.16, 0.72), (0.30, 0.95), (0.60, 0.96), (0.80, 0.76), (0.66, 0.53),
           (0.38, 0.52), (0.18, 0.68)]]],
        [[[(0.66, 0.02), (0.24, 0.56)], [(0.24, 0.56), (0.18, 0.80), (0.42, 0.97), (0.74, 0.86), (0.74, 0.60), (0.45, 0.52),
                                          (0.22, 0.64)]]],
        [[[(0.80, 0.10), (0.50, 0.02), (0.22, 0.30), (0.15, 0.70), (0.45, 0.97), (0.80, 0.78), (0.60, 0.50), (0.20, 0.62)]]],
    ],
    7: [
        [[[(0.10, 0.06), (0.90, 0.06)], [(0.90, 0.06), (0.38, 0.98)]]],
        [[[(0.10, 0.06), (0.90, 0.06)], [(0.90, 0.06), (0.40, 0.98)]], [[(0.34, 0.52), (0.82, 0.52)]]],
        [[[(0.12, 0.14), (0.50, 0.05), (0.90, 0.06)], [(0.90, 0.06), (0.62, 0.45), (0.46, 0.98)]]],
    ],
    8: [
        [[[(0.78, 0.17), (0.50, 0.03), (0.22, 0.17), (0.30, 0.38), (0.70, 0.60), (0.80, 0.82), (0.50, 0.97), (0.20, 0.82),
           (0.30, 0.60), (0.70, 0.38), (0.78, 0.17)]]],
        [[ell(0.5, 0.26, 0.24, 0.22, 90, 450, 11)], [ell(0.5, 0.73, 0.30, 0.25, -90, 270, 11)]],
        [[[(0.30, 0.18), (0.50, 0.03), (0.75, 0.18), (0.25, 0.65), (0.45, 0.97), (0.78, 0.75), (0.25, 0.35), (0.30, 0.18)]]],
    ],
    9: [
        [[[(0.80, 0.24), (0.56, 0.04), (0.26, 0.10), (0.18, 0.34), (0.44, 0.50), (0.76, 0.38), (0.80, 0.20)],
          [(0.80, 0.20), (0.74, 0.98)]]],
        [[[(0.80, 0.24), (0.56, 0.04), (0.24, 0.12), (0.20, 0.36), (0.48, 0.48), (0.78, 0.30)],
          [(0.78, 0.30), (0.80, 0.60), (0.66, 0.90), (0.38, 0.98)]]],
        [[ell(0.48, 0.28, 0.30, 0.24, 0, 360, 11)], [[(0.78, 0.30), (0.62, 0.98)]]],
    ],
}


def catmull(pts, per=10):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts
    if len(pts) == 2:
        return np.linspace(pts[0], pts[1], max(2, per))
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def build_stroke(pieces, jitter=0.0, rng=None, per=10):
    out = []
    prev_end = None
    for pc in pieces:
        pc = np.asarray(pc, dtype=np.float64).copy()
        if rng is not None and jitter > 0:
            pc = pc + rng.normal(0, jitter, pc.shape)
        if prev_end is not None:
            pc[0] = prev_end                     # corners stay joined
        seg = catmull(pc, per)
        out.append(seg if not out else seg[1:])
        prev_end = pc[-1]
    return np.vstack(out)


def resample(poly, step):
    d = np.sqrt(((poly[1:] - poly[:-1]) ** 2).sum(1))
    L = d.sum()
    if L < 1e-9:
        return poly[:1]
    cum = np.concatenate([[0], np.cumsum(d)])
    n = max(2, int(math.ceil(L / step)) + 1)
    ts = np.linspace(0, L, n)
    return np.stack([np.interp(ts, cum, poly[:, 0]), np.interp(ts, cum, poly[:, 1])], 1)


def sample_strokes(digit, variant=0):
    """The canonical (un-perturbed) strokes of a template, placed in PAD units as the app's sample digits."""
    strokes = [build_stroke(p, per=12) for p in TEMPLATES[digit][variant]]
    out = []
    for s in strokes:
        s = resample(s, 0.012)
        out.append(np.stack([0.30 + 0.40 * s[:, 0], 0.14 + 0.72 * s[:, 1]], 1))
    return out


def pen_example(digit, rng):
    tpl = TEMPLATES[digit][rng.integers(len(TEMPLATES[digit]))]
    jit = rng.uniform(0.0, 0.045)
    strokes = [build_stroke(p, jit, rng, per=10) for p in tpl]
    # global affine: rotation, shear, aspect
    ang = math.radians(rng.uniform(-15, 15)); sh = rng.uniform(-0.3, 0.3); asp = math.exp(rng.uniform(-0.3, 0.3))
    M = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) @ np.array([[asp, sh], [0, 1.0]])
    # smooth elastic warp of the plane (a few random low-frequency modes)
    modes = [(rng.normal(0, 1, 2), rng.uniform(0.8, 2.5), rng.uniform(0, 2 * math.pi), rng.normal(0, 1, 2) / 2) for _ in range(3)]
    amp = rng.uniform(0.0, 0.05)
    out = []
    for k, s in enumerate(strokes):
        s = resample(s, rng.uniform(0.006, 0.02))
        # hand tremor along the path
        n = len(s); u = np.linspace(0, 1, n)
        trem = np.zeros_like(s)
        for _ in range(3):
            f = rng.uniform(1.5, 9); ph = rng.uniform(0, 2 * math.pi); a = rng.uniform(0, 0.018)
            trem += a * np.stack([np.sin(2 * math.pi * f * u + ph), np.cos(2 * math.pi * f * u * 1.3 + ph)], 1)
        s = s + trem
        for (kv, fr, ph, dirv) in modes:
            proj = s @ kv
            s = s + amp * np.sin(fr * proj + ph)[:, None] * dirv[None, :]
        # later strokes start where the gaze snapped them (app: 5x5 start cells) -> offset
        if k > 0:
            s = s + rng.normal(0, 0.045, 2)
        # truncated / overshooting ends
        if rng.random() < 0.25 and len(s) > 6:
            cut = rng.integers(0, max(1, len(s) // 10))
            s = s[: len(s) - cut] if rng.random() < 0.5 else s[cut:]
        out.append(s @ M.T)
    pen = rng.uniform(0.9, 1.8)
    return raster_strokes(out, pen)


# ------------------------------------------------------------------ fonts
FONT_FILES = [
    "Helvetica.ttc", "HelveticaNeue.ttc", "Avenir.ttc", "Avenir Next.ttc", "Avenir Next Condensed.ttc", "Courier.ttc",
    "Optima.ttc", "Palatino.ttc", "Times.ttc", "Menlo.ttc", "Monaco.ttf", "Geneva.ttf", "SFNS.ttf", "SFNSRounded.ttf",
    "SFCompact.ttf", "SFNSMono.ttf", "NewYork.ttf", "MarkerFelt.ttc", "Noteworthy.ttc", "LucidaGrande.ttc",
    "Supplemental/Futura.ttc", "Supplemental/GillSans.ttc", "Supplemental/Arial.ttf", "Supplemental/Arial Bold.ttf",
    "Supplemental/Arial Rounded Bold.ttf", "Supplemental/Arial Narrow.ttf", "Supplemental/Verdana.ttf",
    "Supplemental/Verdana Bold.ttf", "Supplemental/Georgia.ttf", "Supplemental/Tahoma.ttf", "Supplemental/Trebuchet MS.ttf",
    "Supplemental/Comic Sans MS.ttf", "Supplemental/Comic Sans MS Bold.ttf", "Supplemental/Bradley Hand Bold.ttf",
    "Supplemental/Chalkduster.ttf", "Supplemental/Andale Mono.ttf", "Supplemental/Courier New.ttf",
    "Supplemental/Times New Roman.ttf", "Supplemental/AmericanTypewriter.ttc", "Supplemental/Baskerville.ttc",
    "Supplemental/Didot.ttc", "Supplemental/Rockwell.ttc", "Supplemental/Seravek.ttc", "Supplemental/Skia.ttf",
    "Supplemental/PTSans.ttc", "Supplemental/PTSerif.ttc", "Supplemental/PTMono.ttc", "Supplemental/Charter.ttc",
    "Supplemental/Iowan Old Style.ttc", "Supplemental/Hoefler Text.ttc", "Supplemental/Cochin.ttc",
    "Supplemental/DIN Alternate Bold.ttf", "Supplemental/DIN Condensed Bold.ttf", "Supplemental/Impact.ttf",
    "Supplemental/Microsoft Sans Serif.ttf", "Supplemental/Chalkboard.ttc", "Supplemental/ChalkboardSE.ttc",
    "Supplemental/SignPainter.ttc", "Supplemental/Apple Chancery.ttf", "Supplemental/Brush Script.ttf",
    "Supplemental/Trattatello.ttf", "Supplemental/Kefa.ttc", "Supplemental/Athelas.ttc", "Supplemental/Marion.ttc",
    "Supplemental/BigCaslon.ttf", "Supplemental/Copperplate.ttc", "Supplemental/Phosphate.ttc",
]
# never trained on: the "unseen fonts" check
HOLDOUT_FONTS = {"Supplemental/Verdana.ttf", "Supplemental/Georgia.ttf", "Noteworthy.ttc", "Supplemental/Rockwell.ttc",
                 "Supplemental/PTSans.ttc", "Supplemental/Charter.ttc"}
FONT_PX = 110
CANVAS = 200


def font_faces():
    faces = []
    for rel in FONT_FILES:
        path = os.path.join("/System/Library/Fonts", rel)
        if not os.path.exists(path):
            continue
        for idx in range(8):
            try:
                ImageFont.truetype(path, FONT_PX, index=idx)
            except Exception:
                break
            faces.append((rel, path, idx))
    return faces


def render_glyph(path, idx, ch, px=FONT_PX, canvas=CANVAS):
    f = ImageFont.truetype(path, px, index=idx)
    im = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(im)
    l, t, r, b = d.textbbox((0, 0), ch, font=f)
    d.text(((canvas - (r - l)) / 2 - l, (canvas - (b - t)) / 2 - t), ch, fill=255, font=f)
    return np.asarray(im, dtype=np.float64) / 255.0


def usable_faces():
    """Drop faces whose digits are missing (tofu boxes / empty) or not distinct."""
    keep = []
    for rel, path, idx in font_faces():
        try:
            gl = [render_glyph(path, idx, str(d), 48, 96) > 0.5 for d in range(10)]
        except Exception:
            continue
        if any(g.sum() < 30 for g in gl):
            continue
        same = sum(int((gl[a] ^ gl[b]).sum() < 40) for a in range(10) for b in range(a + 1, 10))
        if same:
            continue
        keep.append((rel, path, idx))
    return keep


def mnist_normalise(img, rng=None, jitter=0.0):
    """LeCun et al.: fit the ink bbox into 20x20 (aspect kept, anti-aliased), centre of mass -> centre of 28x28."""
    ys, xs = np.nonzero(img > 0.08)
    if len(xs) == 0:
        return np.zeros(GRID * GRID)
    crop = img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    s = BOX / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS), dtype=np.float64) / 255.0
    small = np.clip(small, 0, 1)
    out = np.zeros((GRID, GRID))
    oy, ox = (GRID - nh) // 2, (GRID - nw) // 2
    out[oy: oy + nh, ox: ox + nw] = small
    m = out.sum()
    cy = (out.sum(1) * (np.arange(GRID) + 0.5)).sum() / m
    cx = (out.sum(0) * (np.arange(GRID) + 0.5)).sum() / m
    dy, dx = CENTRE - cy, CENTRE - cx
    if rng is not None and jitter > 0:
        dy += rng.uniform(-jitter, jitter); dx += rng.uniform(-jitter, jitter)
    out = ndimage.shift(out, (dy, dx), order=1, mode="constant", cval=0.0)
    return np.clip(out, 0, 1).reshape(-1)


def elastic(img, rng, alpha, sigma):
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha
    y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
    return ndimage.map_coordinates(img, [y + dy, x + dx], order=1, mode="constant")


_GLYPH_CACHE = {}


def font_example(face, digit, rng):
    rel, path, idx = face
    key = (path, idx, digit)
    if key not in _GLYPH_CACHE:
        _GLYPH_CACHE[key] = render_glyph(path, idx, str(digit))
    g = _GLYPH_CACHE[key].copy()
    # thickness: dilate / erode at high resolution
    k = rng.integers(-1, 4)
    if k > 0:
        g = ndimage.grey_dilation(g, size=(2 * k + 1, 2 * k + 1))
    elif k < 0:
        g = ndimage.grey_erosion(g, size=(3, 3))
    # rotation, shear, aspect about the centre
    ang = math.radians(rng.uniform(-15, 15)); sh = rng.uniform(-0.3, 0.3); asp = math.exp(rng.uniform(-0.25, 0.25))
    A = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]) @ np.array([[1.0, 0.0], [sh, 1.0]]) @ np.diag([1.0, asp])
    Ai = np.linalg.inv(A)
    c = np.array([CANVAS / 2, CANVAS / 2])
    g = ndimage.affine_transform(g, Ai, offset=c - Ai @ c, order=1, mode="constant")
    if rng.random() < 0.7:
        g = elastic(g, rng, alpha=rng.uniform(80, 260), sigma=rng.uniform(14, 22))
    return mnist_normalise(np.clip(g, 0, 1), rng, jitter=1.0)


def add_noise(x, rng):
    if rng.random() < 0.20:
        x = np.clip(x + rng.normal(0, rng.uniform(0.05, 0.35), x.shape), 0, 1)
    return x


# ------------------------------------------------------------------ dataset generation (parallel, deterministic)
FACES = None


def _init():
    global FACES
    FACES = usable_faces()


def _make(args):
    split, i, kind, faces_sel = args
    rng = np.random.default_rng([20261027, {"train": 1, "held": 2, "unseen": 3}[split], i])
    digit = i % 10
    if kind == "pen":
        x = pen_example(digit, rng)
    else:
        pool = [f for f in FACES if (f[0] in HOLDOUT_FONTS) == (faces_sel == "unseen")]
        x = font_example(pool[rng.integers(len(pool))], digit, rng)
    if split != "unseen":
        x = add_noise(x, rng)
    return x.astype(np.float32), digit


def make_split(pool, split, n, pen_frac=0.5, faces_sel="seen"):
    jobs = [(split, i, "pen" if (i // 10) % 100 < pen_frac * 100 else "font", faces_sel) for i in range(n)]
    res = pool.map(_make, jobs, chunksize=256)
    X = np.stack([r[0] for r in res]); y = np.array([r[1] for r in res], dtype=np.int64)
    return X, y


# ------------------------------------------------------------------ model
W1_DECAY = float(os.environ.get("W1_DECAY", "0.5"))


def train(Xtr, ytr, Xva, yva):
    import torch
    import torch.nn as nn
    torch.manual_seed(7)
    torch.set_num_threads(8)
    net = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    # strong decay on W1 only: input pixels that only ever see noise (the borders) fade to zero instead of keeping
    # random weights, so the first-layer weight maps the demo shows are readable; accuracy is unaffected
    opt = torch.optim.AdamW([{"params": [net[0].weight], "weight_decay": W1_DECAY},
                             {"params": [p for n, p in net.named_parameters() if n != "0.weight"], "weight_decay": 2e-4}], lr=2e-3)
    EPOCHS = 30
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * math.ceil(len(Xtr) / 128))
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    Xv = torch.tensor(Xva); yv = torch.tensor(yva)
    g = torch.Generator().manual_seed(11)
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for k in range(0, len(Xt), 128):
            idx = perm[k: k + 128]
            xb = Xt[idx]
            # light online augmentation: random +-1 px shifts (the app input is exactly centred; this adds slack)
            if torch.rand(1, generator=g).item() < 0.5:
                sx, sy = [int(v) for v in torch.randint(-1, 2, (2,), generator=g)]
                xb = torch.roll(xb.view(-1, 28, 28), shifts=(sy, sx), dims=(1, 2)).reshape(-1, 784)
            loss = nn.functional.cross_entropy(net(xb), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            acc = (net(Xv).argmax(1) == yv).float().mean().item()
        print(f"  epoch {ep + 1:2d}  loss {tot / len(Xt):.4f}  held-out acc {acc * 100:.2f} %", flush=True)
    return net


def fit_temperature(net, Xv, yv):
    import torch
    with torch.no_grad():
        z = net(torch.tensor(Xv)).double()
    y = torch.tensor(yv)
    best = (1e9, 1.0)
    for T in np.arange(0.5, 3.0001, 0.01):
        nll = torch.nn.functional.cross_entropy(z / T, y).item()
        if nll < best[0]:
            best = (nll, float(T))
    return best[1]


def forward_np(W, x):
    W1, b1, W2, b2, W3, b3 = W
    h1 = np.maximum(W1 @ x + b1, 0); h2 = np.maximum(W2 @ h1 + b2, 0); z = W3 @ h2 + b3
    return h1, h2, z


def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


def ece(probs, y, bins=10):
    conf = probs.max(1); pred = probs.argmax(1); e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cache = os.environ.get("DATA_CACHE")          # optional: reuse a generated dataset while iterating
    if cache and os.path.exists(cache):
        z = np.load(cache)
        Xtr, ytr, Xva, yva, Xun, yun = (z[k] for k in ["Xtr", "ytr", "Xva", "yva", "Xun", "yun"])
        print(f"data: loaded {cache}")
    else:
        with Pool(initializer=_init) as pool:
            faces = pool.apply(usable_faces)
            print(f"fonts: {len(faces)} usable faces from {len(set(f[0] for f in faces))} files "
                  f"({sum(f[0] in HOLDOUT_FONTS for f in faces)} faces held out as unseen)")
            Xtr, ytr = make_split(pool, "train", 80000)
            Xva, yva = make_split(pool, "held", 10000)
            Xun, yun = make_split(pool, "unseen", 2000, pen_frac=0.0, faces_sel="unseen")
        if cache:
            np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xun=Xun, yun=yun)
    print(f"data: train {len(Xtr)}, held-out {len(Xva)}, unseen-font {len(Xun)}  ({time.time() - t0:.0f} s)")
    if "--sheet" in sys.argv:
        sheet = np.zeros((28 * 10, 28 * 20))
        for d in range(10):
            idx = np.nonzero(ytr == d)[0][:20]
            for k, i in enumerate(idx):
                sheet[d * 28:(d + 1) * 28, k * 28:(k + 1) * 28] = Xtr[i].reshape(28, 28)
        Image.fromarray((sheet * 255).astype(np.uint8)).resize((28 * 20 * 3, 28 * 10 * 3), Image.NEAREST).save(sys.argv[sys.argv.index("--sheet") + 1])

    net = train(Xtr, ytr, Xva, yva)
    import torch
    T = fit_temperature(net, Xva, yva)
    W = [p.detach().double().numpy() for p in net.parameters()]
    W1, b1, W2, b2, W3, b3 = W

    def evaluate(X, y):
        with torch.no_grad():
            z = net(torch.tensor(X)).double().numpy()
        p = np.exp((z - z.max(1, keepdims=True)) / T); p /= p.sum(1, keepdims=True)
        return (z.argmax(1) == y).mean(), p

    acc_va, p_va = evaluate(Xva, yva)
    pen = np.array([(i // 10) % 100 < 50 for i in range(len(Xva))])
    rngn = np.random.default_rng(99)
    Xno = np.clip(Xva + rngn.normal(0, 0.30, Xva.shape), 0, 1).astype(np.float32)
    acc_no, _ = evaluate(Xno, yva)
    acc_un, _ = evaluate(Xun, yun)
    print(f"RESULT held-out synthetic accuracy {acc_va * 100:.2f} % (n={len(Xva)}; pen {evaluate(Xva[pen], yva[pen])[0] * 100:.2f} %, "
          f"font {evaluate(Xva[~pen], yva[~pen])[0] * 100:.2f} %)")
    print(f"RESULT with noise sigma 0.30: {acc_no * 100:.2f} %   unseen fonts: {acc_un * 100:.2f} % (n={len(Xun)})")
    print(f"RESULT temperature T = {T:.2f}  ECE {ece(p_va, yva) * 100:.2f} %")

    # display scales: 99th percentile of each hidden layer's activations on held-out data
    H1 = np.maximum(Xva.astype(np.float64) @ W1.T + b1, 0); H2 = np.maximum(H1 @ W2.T + b2, 0)
    s1 = float(np.percentile(H1, 99)); s2 = float(np.percentile(H2, 99))
    print(f"RESULT act_scale1 {s1:.3f}  act_scale2 {s2:.3f}; mean active (>0) h1 {(H1 > 0).sum(1).mean():.1f}/64, h2 {(H2 > 0).sum(1).mean():.1f}/32")

    # ---- weights
    flat = np.concatenate([[784, 64, 32, 10], W1.reshape(-1), b1, W2.reshape(-1), b2, W3.reshape(-1), b3, [s1, s2, T]]).astype("<f4")
    flat.tofile(os.path.join(OUT, "mlp_w.bin"))
    Wf = [w.astype(np.float32).astype(np.float64) for w in W]      # exactly what ships

    # ---- samples (pad strokes) + reference vectors
    samples = [sample_strokes(d, 0) for d in range(10)]
    buf = [10.0]
    for st in samples:
        buf.append(len(st))
        for s in st:
            s32 = s.astype(np.float32)
            buf.append(len(s32)); buf.extend(s32.reshape(-1).tolist())
    np.array(buf, dtype="<f4").tofile(os.path.join(OUT, "samples.bin"))
    refs = []
    for d in range(10):
        st32 = [s.astype(np.float32).astype(np.float64) for s in samples[d]]
        refs.append(raster_strokes(st32))
    for i in [3, 17, 4242, 9001]:
        refs.append(Xva[i].astype(np.float64))
    refs.append(Xno[5].astype(np.float64))
    rb = [len(refs)]
    sample_pred = []
    for x in refs:
        x32 = x.astype(np.float32)
        with torch.no_grad():
            xt = torch.tensor(x32)[None]
            h1 = net[1](net[0](xt)); h2 = net[3](net[2](h1)); z = net[4](h2)
        h1, h2, z = h1[0].double().numpy(), h2[0].double().numpy(), z[0].double().numpy()
        p = softmax(z / T)
        sample_pred.append((int(z.argmax()), float(p.max())))
        rb.extend(x32.tolist()); rb.extend(h1.tolist()); rb.extend(h2.tolist()); rb.extend(z.tolist()); rb.extend(p.tolist())
    np.array(rb, dtype="<f4").tofile(os.path.join(OUT, "ref.bin"))
    print("RESULT sample digits (app strokes) ->", " ".join(f"{d}:{sample_pred[d][0]}({sample_pred[d][1] * 100:.0f}%)" for d in range(10)))

    # ---- held-out slice, quantised exactly as shipped
    N = 1000
    q = np.clip(np.round(Xva[:N] * 255), 0, 255).astype(np.uint8)
    with open(os.path.join(OUT, "heldout.bin"), "wb") as f:
        f.write(struct.pack("<i", N)); f.write(q.tobytes()); f.write(yva[:N].astype(np.uint8).tobytes())
    xq = q.astype(np.float32) / 255.0
    with torch.no_grad():
        accq = (net(torch.tensor(xq)).argmax(1).numpy() == yva[:N]).sum()
    print(f"RESULT heldout.bin slice: {accq}/{N} correct")

    # ---- output-bar glyphs
    gl = np.zeros((10, 48, 48), dtype=np.uint8)
    gpath = "/System/Library/Fonts/SFNSRounded.ttf" if os.path.exists("/System/Library/Fonts/SFNSRounded.ttf") else "/System/Library/Fonts/Helvetica.ttc"
    for d in range(10):
        f = ImageFont.truetype(gpath, 40)
        try:
            f.set_variation_by_name("Bold")
        except Exception:
            pass
        im = Image.new("L", (48, 48), 0); dr = ImageDraw.Draw(im)
        l, t, r, b = dr.textbbox((0, 0), str(d), font=f)
        dr.text(((48 - (r - l)) / 2 - l, (48 - (b - t)) / 2 - t), str(d), fill=255, font=f)
        gl[d] = np.asarray(im)
    gl.tofile(os.path.join(OUT, "glyphs.bin"))

    # ---- first-layer weight maps (for looking at, not shipped)
    if "--weights" in sys.argv:
        tiles = np.zeros((8 * 30, 8 * 30, 3))
        for j in range(64):
            w = W1[j].reshape(28, 28); w = w / (np.abs(w).max() + 1e-9)
            r, c = divmod(j, 8)
            t = np.zeros((28, 28, 3)); t[..., 0] = np.clip(-w, 0, 1); t[..., 2] = np.clip(w, 0, 1); t[..., 1] = np.clip(w, 0, 1) * 0.75
            tiles[r * 30 + 1: r * 30 + 29, c * 30 + 1: c * 30 + 29] = t
        Image.fromarray((tiles * 255).astype(np.uint8)).resize((240 * 3, 240 * 3), Image.NEAREST).save(sys.argv[sys.argv.index("--weights") + 1])
    for fn in ["mlp_w.bin", "ref.bin", "samples.bin", "heldout.bin", "glyphs.bin"]:
        print(f"wrote {fn}: {os.path.getsize(os.path.join(OUT, fn))} bytes")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
