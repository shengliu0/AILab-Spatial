#!/usr/bin/env python3
"""
train_cnn.py — the REAL models behind the "卷积神经网络 / CNN" demo (demos/cnn, core/cnn).

Everything is SYNTHETIC and generated here (no downloads):
  * digits: the ten digits rendered from the Mac's system fonts (/System/Library/Fonts), each glyph
    random-affine warped (rotation, shear, scale, shift), elastically jittered, thinned / emboldened,
    coloured (bright random-hue stroke on a dark random-hue background, with a gradient) and noised.
    ~20 % of the font FACES are held out: the test set uses only fonts the network never saw.
  * multispectral scenes: 28x28 "satellite tiles" with 4 bands R, G, B, NIR. Classes: vegetation,
    water, artificial turf, roofs, bare soil (+ unlabelled asphalt roads). Vegetation and artificial
    turf are drawn from the SAME RGB distribution on purpose; only NIR tells them apart (healthy
    leaves reflect near-infrared strongly, plastic does not; water absorbs it). Multi-label task:
    "which of the 5 cover types are present (>= 12 px)?" -> 5 sigmoid outputs.

Network (identical skeleton for every model, LeNet-style, valid convolutions):
    x C x 28 x 28 -> conv 3x3 (C->8) -> ReLU -> maxpool 2 -> conv 3x3 (8->16) -> ReLU -> maxpool 2
      -> flatten 16*5*5 = 400 -> dense -> N   (digits: N = 10 softmax; scenes: N = 5 sigmoid)
Six models are trained:
    digits_learned  C=3  all layers learned
    digits_hand     C=3  conv1 FIXED to 8 hand-made kernels (edges, blur, sharpen, ...), rest learned
    scene4_learned  C=4  R,G,B,NIR
    scene4_hand     C=4  conv1 fixed: NIR-R (NDVI-like), green, water, NIR, edges, blur, sharpen
    scene3_learned  C=3  the same scenes seen by an RGB-only sensor (NIR band dropped)
    scene3_hand     C=3  the hand bank with its NIR slices dropped

Outputs (little-endian), in app/src/main/assets/cnn/:
  <model>.bin      int32 magic 'CNN1' (0x434E4E31), C, K1=8, K2=16, N, head (0 softmax / 1 sigmoid)
                   float32 conv1.w[K1][C][3][3], conv1.b[K1], conv2.w[K2][K1][3][3], conv2.b[K2],
                           fc.w[N][K2*5*5] (flatten order c,h,w as torch), fc.b[N]
  digits_eval.bin / scenes_eval.bin   (held-out sets, shipped: the demo shows them, the JVM test scores them)
                   int32 magic 'CNND', count, C, H, W; u8 label[count] (digit, or presence bitmask
                   bit c = class c); u8 pixel[count][C][H][W] (value = byte / 255)
  ref.bin          int32 magic 'CNNR', nModels; per model: int32 len, ascii name, int32 sampleIdx,
                   float32 conv1 (pre-ReLU) [K1*26*26], pool1 [K1*13*13], conv2 (pre-ReLU) [K2*11*11],
                   pool2 [K2*5*5], logits [N]; int32 M; float32 logits[M][N] for eval samples 0..M-1;
                   int32 torch's number of correct predictions on the whole shipped eval set
  metrics.txt      key=value lines (held-out accuracies printed below)
  glyphs.bin       int32 magic 'CNNG', count; per glyph: int32 len, ascii key, int32 w, h, u8 alpha[h][w]
                   (pre-rendered label text: class names zh/en, digits, channel tags, kernel names)

Run:  python3 tools/train_cnn.py        (deterministic: fixed seeds, CPU, ~3-6 min on an M-series Mac)
"""
import glob
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "app", "src", "main", "assets", "cnn")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(1989)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)
DEV = "cpu"   # deterministic, and the model is tiny

# =============================================================================================== fonts
BAD = ("Braille", "Emoji", "LastResort", "Symbol", "Webdings", "Wingdings", "Zapf", "Ornaments", "Keyboard",
       "Bodoni Ornaments", "Apple Symbols", "Fallback", "Mishafi", "Farisi")


def font_faces():
    files = sorted(glob.glob("/System/Library/Fonts/*.tt[cf]") + glob.glob("/System/Library/Fonts/*.otf")
                   + glob.glob("/System/Library/Fonts/Supplemental/*.tt[cf]") + glob.glob("/System/Library/Fonts/Supplemental/*.otf"))
    faces = []
    for p in files:
        if any(b.lower() in os.path.basename(p).lower() for b in BAD):
            continue
        for idx in range(4):                       # at most 4 faces per family: no family dominates
            try:
                f = ImageFont.truetype(p, 96, index=idx)
            except Exception:
                break
            faces.append((p, idx))
    return faces


def render_glyph(font, ch):
    im = Image.new("L", (192, 192), 0)
    ImageDraw.Draw(im).text((48, 24), ch, font=font, fill=255)
    a = np.asarray(im, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.2)
    if len(ys) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def glyph_canvas(g, thick):
    """Glyph (float mask) -> 64x64 canvas, bbox fitted to height 48 (width <= 56), centred.
    thick: -1 thin (MinFilter), 0 as drawn, 1 bold, 2 extra bold (MaxFilter) at 96 px scale."""
    im = Image.fromarray((g * 255).astype(np.uint8))
    im = im.crop((-8, -8, im.width + 8, im.height + 8))
    if thick == -1:
        im = im.filter(ImageFilter.MinFilter(5))
    elif thick == 1:
        im = im.filter(ImageFilter.MaxFilter(5))
    elif thick == 2:
        im = im.filter(ImageFilter.MaxFilter(9))
    a = np.asarray(im, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.2)
    if len(ys) == 0:
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = min(48.0 / h, 56.0 / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS), dtype=np.float32) / 255.0
    c = np.zeros((64, 64), np.float32)
    y0, x0 = (64 - nh) // 2, (64 - nw) // 2
    c[y0:y0 + nh, x0:x0 + nw] = a
    return c


def build_glyph_bank():
    faces = font_faces()
    bank = []          # (face_id, digit, canvas)
    good_faces = []
    for fid, (p, idx) in enumerate(faces):
        f = ImageFont.truetype(p, 96, index=idx)
        try:
            gs = [render_glyph(f, str(d)) for d in range(10)]
            nd = render_glyph(f, "\uE000")          # .notdef / tofu: a font without Latin digits draws this
        except OSError:                              # a few faces crash FreeType ("stack overflow"): skip
            continue
        if any(g is None for g in gs):
            continue
        keys = [g.shape + (round(float(g.sum()), 1),) for g in gs]
        if len(set(keys)) < 10:
            continue
        if nd is not None and any(g.shape == nd.shape and abs(g.sum() - nd.sum()) < 1 for g in gs):
            continue
        if any(g.shape[0] < 30 for g in gs):        # digits must be tall glyphs (rejects tiny/odd faces)
            continue
        good_faces.append((p, idx))
        for d, g in enumerate(gs):
            for th in (-1, 0, 1, 2):
                c = glyph_canvas(g, th)
                if c is not None:
                    bank.append((len(good_faces) - 1, d, c))
    return good_faces, bank


# =============================================================================================== digits
def hsv2rgb(h, s, v):
    h = (h % 1.0) * 6
    i = np.floor(h).astype(int)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v]); g = np.choose(i, [t, v, v, q, p, p]); b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


def lum(c):
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def make_digits(canv, labels, n, rng, chunk=4096):
    """canv: torch [G,1,64,64] glyph canvases, labels: np [G]. Returns uint8 [n,3,28,28], labels [n]."""
    G = canv.shape[0]
    out = np.zeros((n, 3, 28, 28), np.uint8)
    ys = np.zeros(n, np.int64)
    gen = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
    for s0 in range(0, n, chunk):
        b = min(chunk, n - s0)
        gi = rng.integers(0, G, b)
        ys[s0:s0 + b] = labels[gi]
        h = rng.uniform(15.0, 22.5, b)                      # glyph height in output pixels
        ang = np.radians(rng.uniform(-15, 15, b)); sh = rng.uniform(-0.22, 0.22, b)
        asp = np.exp(rng.uniform(-0.15, 0.15, b))            # mild aspect change
        dx = rng.uniform(-2.2, 2.2, b); dy = rng.uniform(-2.2, 2.2, b)
        s = 21.0 / h                                          # output-normalised -> canvas-normalised
        ca, sa = np.cos(ang), np.sin(ang)
        # A = s * R(ang) * Shear(sh) * diag(asp, 1)
        a00 = s * (ca * asp); a01 = s * (ca * sh - sa); a10 = s * (sa * asp); a11 = s * (sa * sh + ca)
        tx = -(a00 * dx + a01 * dy) * 2 / 28; ty = -(a10 * dx + a11 * dy) * 2 / 28
        theta = torch.tensor(np.stack([np.stack([a00, a01, tx], -1), np.stack([a10, a11, ty], -1)], 1), dtype=torch.float32)
        grid = F.affine_grid(theta, (b, 1, 28, 28), align_corners=False)
        # elastic jitter: smooth random displacement, ~0.8 px rms
        amp = torch.tensor(rng.uniform(0.0, 1.4, b), dtype=torch.float32).view(b, 1, 1, 1)
        disp = torch.randn(b, 2, 4, 4, generator=gen) * amp * (2 / 28)
        disp = F.interpolate(disp, size=(28, 28), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
        g = canv[torch.from_numpy(gi)]
        m = F.grid_sample(g, grid + disp, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0].numpy()
        m = np.clip(m * rng.uniform(0.85, 1.0, (b, 1, 1)), 0, 1)
        # colours: bright stroke, dark background, luminance contrast >= 0.30
        fg = hsv2rgb(rng.uniform(0, 1, b), rng.uniform(0.25, 1.0, b), rng.uniform(0.80, 1.0, b))
        bg = hsv2rgb(rng.uniform(0, 1, b), rng.uniform(0.2, 1.0, b), rng.uniform(0.0, 0.40, b))
        low = lum(fg) - lum(bg) < 0.30
        bg[low] *= 0.35
        yy, xx = np.mgrid[0:28, 0:28] / 27.0 - 0.5
        ga = rng.uniform(0, 2 * np.pi, b)
        grad = 1 + rng.uniform(0, 0.5, (b, 1, 1)) * (np.cos(ga)[:, None, None] * xx + np.sin(ga)[:, None, None] * yy)
        img = bg[:, :, None, None] * grad[:, None] * (1 - m[:, None]) + fg[:, :, None, None] * m[:, None]
        img += rng.normal(0, 1, img.shape) * rng.uniform(0, 0.045, (b, 1, 1, 1))
        out[s0:s0 + b] = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
    return out, ys


# =============================================================================================== scenes
VEG, WATER, TURF, ROOF, SOIL = range(5)
# per-class band ranges (R, G, B, NIR), display units 0..1. VEG and TURF share their RGB ranges ON PURPOSE.
RANGES = {
    VEG:   [(0.10, 0.22), (0.30, 0.50), (0.08, 0.18), (0.62, 0.92)],
    TURF:  [(0.10, 0.22), (0.30, 0.50), (0.08, 0.18), (0.06, 0.22)],
    WATER: [(0.04, 0.10), (0.12, 0.24), (0.26, 0.46), (0.00, 0.05)],
    ROOF:  None,  # red tiles or grey sheet, below
    SOIL:  [(0.40, 0.56), (0.30, 0.42), (0.18, 0.28), (0.44, 0.60)],
}
ROAD = 5
TEX = {VEG: 0.06, TURF: 0.06, WATER: 0.015, ROOF: 0.025, SOIL: 0.04, ROAD: 0.02}


def class_spectrum(c, rng):
    if c == ROOF:
        if rng.random() < 0.5:   # red clay tiles
            r = [rng.uniform(0.55, 0.75), rng.uniform(0.24, 0.34), rng.uniform(0.18, 0.28)]
        else:                    # grey metal / concrete
            v = rng.uniform(0.45, 0.68); r = [v + rng.uniform(-0.03, 0.03) for _ in range(3)]
        return np.array(r + [rng.uniform(0.35, 0.55)], np.float32)
    if c == ROAD:
        v = rng.uniform(0.12, 0.20)
        return np.array([v, v, v * 1.05, rng.uniform(0.14, 0.22)], np.float32)
    return np.array([rng.uniform(*rr) for rr in RANGES[c]], np.float32)


def smooth_noise(rng, n=56, cells=7):
    z = rng.normal(0, 1, (cells + 1, cells + 1)).astype(np.float32)
    t = torch.from_numpy(z)[None, None]
    return F.interpolate(t, size=(n, n), mode="bicubic", align_corners=True)[0, 0].numpy()


def shape_mask(kind, rng, n=56):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx, cy = rng.uniform(6, n - 6, 2)
    a = rng.uniform(0, np.pi)
    ca, sa = np.cos(a), np.sin(a)
    u = (xx - cx) * ca + (yy - cy) * sa; v = -(xx - cx) * sa + (yy - cy) * ca
    if kind == "ellipse":
        rx, ry = rng.uniform(5, 15, 2)
        w = smooth_noise(rng, n, 5) * 0.18
        return (u / rx) ** 2 + (v / ry) ** 2 < 1 + w
    if kind == "rect":
        hw, hh = rng.uniform(5, 16), rng.uniform(4, 11)
        return (np.abs(u) < hw) & (np.abs(v) < hh)
    if kind == "smallrect":
        hw, hh = rng.uniform(3, 7), rng.uniform(3, 6)
        return (np.abs(u) < hw) & (np.abs(v) < hh)
    if kind == "blobs":     # tree clumps
        m = np.zeros((n, n), bool)
        for _ in range(rng.integers(3, 8)):
            bx, by = cx + rng.normal(0, 7), cy + rng.normal(0, 7); r = rng.uniform(2.5, 6)
            m |= (xx - bx) ** 2 + (yy - by) ** 2 < r * r
        return m
    if kind == "road":
        wdt = rng.uniform(1.6, 2.8)
        return np.abs(v) < wdt
    raise ValueError(kind)


def make_scene(rng, layout=None):
    n = 56
    cls = np.zeros((n, n), np.int64)
    bgc = layout["bg"] if layout else rng.choice([VEG, SOIL, WATER], p=[0.45, 0.40, 0.15])
    cls[:] = bgc
    regions = [(np.ones((n, n), bool), bgc)]
    objs = layout["objs"] if layout else None
    if objs is None:
        objs = []
        for _ in range(rng.integers(1, 5)):
            k = rng.choice(["veg_blobs", "veg_field", "pond", "turf", "roof", "roofs", "soil_field", "road"],
                           p=[0.15, 0.12, 0.14, 0.20, 0.10, 0.10, 0.09, 0.10])
            objs.append(k)
    for k in objs:
        if isinstance(k, tuple):           # layout: (class, mask)
            c, m = k
        elif k == "veg_blobs": c, m = VEG, shape_mask("blobs", rng)
        elif k == "veg_field": c, m = VEG, shape_mask("rect", rng)
        elif k == "pond": c, m = WATER, shape_mask("ellipse", rng)
        elif k == "turf": c, m = TURF, shape_mask("rect", rng)
        elif k == "roof": c, m = ROOF, shape_mask("smallrect", rng)
        elif k == "roofs":
            c = ROOF; m = np.zeros((n, n), bool)
            yy, xx = np.mgrid[0:n, 0:n]
            for _ in range(rng.integers(2, 6)):
                bx, by = rng.uniform(4, n - 4, 2); hw, hh = rng.uniform(2, 5, 2)
                m |= (np.abs(xx - bx) < hw) & (np.abs(yy - by) < hh)
        elif k == "soil_field": c, m = SOIL, shape_mask("rect", rng)
        elif k == "road": c, m = ROAD, shape_mask("road", rng)
        cls[m] = c
        regions.append((m, c))
    img = np.zeros((4, n, n), np.float32)
    for m, c in regions:                    # later regions paint over earlier ones, as cls does
        spec = class_spectrum(c, rng)
        tex = smooth_noise(rng, n, 9) * TEX[c] + rng.normal(0, TEX[c] * 0.5, (n, n))
        if c == VEG:                        # canopy shadows: darker in every band, NIR included
            tex = tex - np.clip(smooth_noise(rng, n, 11), 0, None) * 0.05
        for b in range(4):
            img[b][m] = (spec[b] * (1 + tex * (2.0 if b == 3 else 1.0)))[m] if c in (VEG, TURF) else (spec[b] + tex)[m]
    img = img.reshape(4, 28, 2, 28, 2).mean((2, 4))           # 2x2 box: mixed pixels at edges
    img += rng.normal(0, 0.012, img.shape)
    lab = 0
    for c in range(5):
        if (cls == c).sum() / 4.0 >= 12:
            lab |= 1 << c
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8), lab


def showcase_layouts():
    """Hand-composed tiles shown first in the demo: meadow + trees beside an artificial-turf pitch etc."""
    n = 56
    yy, xx = np.mgrid[0:n, 0:n]
    out = []
    # 1: soil background; meadow (left), turf pitch (right), pond (bottom left), roofs (top right), road
    turf = (np.abs(xx - 41) < 11) & (np.abs(yy - 30) < 14)
    meadow = (np.abs(xx - 14) < 12) & (np.abs(yy - 22) < 17)
    pond = (xx - 13) ** 2 / 81 + (yy - 46) ** 2 / 36 < 1
    roofs = np.zeros((n, n), bool)
    for bx, by in [(34, 5), (44, 5), (52, 9)]:
        roofs |= (np.abs(xx - bx) < 3.5) & (np.abs(yy - by) < 3)
    road = np.abs(xx - 28) < 1.6
    out.append({"bg": SOIL, "objs": [(VEG, meadow), (ROAD, road), (TURF, turf), (WATER, pond), (ROOF, roofs)]})
    # 2: vegetation background (park) with a turf pitch and a lake
    turf2 = (np.abs(xx - 30) < 13) & (np.abs(yy - 18) < 9)
    lake = (xx - 18) ** 2 / 100 + (yy - 42) ** 2 / 49 < 1
    out.append({"bg": VEG, "objs": [(TURF, turf2), (WATER, lake)]})
    # 3: town: soil, roofs, a road and tree clumps
    trees = np.zeros((n, n), bool)
    for bx, by, r in [(12, 12, 5), (18, 16, 4), (40, 44, 6), (46, 38, 4)]:
        trees |= (xx - bx) ** 2 + (yy - by) ** 2 < r * r
    roofs3 = np.zeros((n, n), bool)
    for bx, by in [(34, 12), (44, 14), (14, 40), (24, 46), (36, 28)]:
        roofs3 |= (np.abs(xx - bx) < 4) & (np.abs(yy - by) < 3.5)
    road3 = np.abs(yy - 27 - (xx - 28) * 0.2) < 1.8
    out.append({"bg": SOIL, "objs": [(VEG, trees), (ROOF, roofs3), (ROAD, road3)]})
    # 4: only artificial turf on soil (an RGB camera calls it grass)
    turf4 = (np.abs(xx - 28) < 18) & (np.abs(yy - 28) < 13)
    out.append({"bg": SOIL, "objs": [(TURF, turf4)]})
    return out


def make_scenes(n, rng, showcase=False):
    xs = np.zeros((n, 4, 28, 28), np.uint8); ys = np.zeros(n, np.int64)
    lays = showcase_layouts() if showcase else []
    for i in range(n):
        xs[i], ys[i] = make_scene(rng, lays[i] if i < len(lays) else None)
    return xs, ys


# =============================================================================================== banks
def lum_kernel(k, C, nir=False):
    """A 3x3 spatial kernel applied to the luminance-like average of the RGB channels (NIR slice 0)."""
    w = np.zeros((C, 3, 3), np.float32)
    for c in range(3):
        w[c] = np.array(k, np.float32) / 3.0
    return w


SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
DIAG_UP = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
DIAG_DN = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
BLUR = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
SHARP = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
SPOT = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]


def digit_bank():
    C = 3
    b = [lum_kernel(np.array(SOBEL_X) / 4, C), lum_kernel(np.array(SOBEL_Y) / 4, C),
         lum_kernel(np.array(DIAG_UP) / 4, C), lum_kernel(np.array(DIAG_DN) / 4, C),
         lum_kernel(np.array(BLUR) / 9, C), lum_kernel(np.array(SHARP), C),
         lum_kernel(np.array(SPOT) / 8, C)]
    rb = np.zeros((C, 3, 3), np.float32); rb[0] = 1 / 9; rb[2] = -1 / 9
    b.append(rb)
    return np.stack(b)          # [8, 3, 3, 3]
DIGIT_BANK_NAMES = ["V-edge", "H-edge", "Diag-up", "Diag-down", "Blur", "Sharpen", "Spot", "R-B"]


def scene_bank(C=4):
    k = []
    w = np.zeros((4, 3, 3), np.float32); w[3] = 1 / 9; w[0] = -1 / 9; k.append(w)                # NIR - R
    w = np.zeros((4, 3, 3), np.float32); w[1] = 1 / 9; w[0] = -0.5 / 9; w[2] = -0.5 / 9; k.append(w)  # green
    w = np.zeros((4, 3, 3), np.float32); w[2] = 1 / 9; w[3] = -1 / 9; k.append(w)                # water: B - NIR
    w = np.zeros((4, 3, 3), np.float32); w[3] = 1 / 9; k.append(w)                               # NIR
    k.append(lum_kernel(np.array(SOBEL_X) / 4, 4)); k.append(lum_kernel(np.array(SOBEL_Y) / 4, 4))
    k.append(lum_kernel(np.array(BLUR) / 9, 4)); k.append(lum_kernel(np.array(SHARP), 4))
    b = np.stack(k)
    return b[:, :C]             # RGB-only sensor: the NIR slices are simply missing
SCENE_BANK_NAMES = ["NIR-R", "Green", "Water", "NIR", "V-edge", "H-edge", "Blur", "Sharpen"]


# =============================================================================================== model
class Net(nn.Module):
    def __init__(self, C, N, fixed=None):
        super().__init__()
        self.conv1 = nn.Conv2d(C, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc = nn.Linear(16 * 5 * 5, N)
        if fixed is not None:
            with torch.no_grad():
                self.conv1.weight.copy_(torch.from_numpy(fixed)); self.conv1.bias.zero_()
            self.conv1.weight.requires_grad_(False); self.conv1.bias.requires_grad_(False)

    def forward(self, x, all_=False):
        a1 = self.conv1(x)
        p1 = F.max_pool2d(F.relu(a1), 2)
        a2 = self.conv2(p1)
        p2 = F.max_pool2d(F.relu(a2), 2)
        z = self.fc(p2.flatten(1))
        return (a1, p1, a2, p2, z) if all_ else z


def to_t(x_u8):
    return torch.from_numpy(x_u8.astype(np.float32) / 255.0)


def train(name, C, N, xtr, ytr, xva, yva, multilabel, fixed=None, epochs=14, seed=0):
    torch.manual_seed(seed)
    net = Net(C, N, fixed)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=4e-3)
    steps = epochs * math.ceil(len(xtr) / 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps, pct_start=0.15)
    X = to_t(xtr); Y = torch.from_numpy(ytr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            z = net(X[idx])
            if multilabel:
                tgt = ((Y[idx][:, None] >> torch.arange(N)) & 1).float()
                loss = F.binary_cross_entropy_with_logits(z, tgt)
            else:
                loss = F.cross_entropy(z, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        if ep % 3 == 2 or ep == epochs - 1:
            print(f"  {name} epoch {ep + 1}/{epochs} loss {tot / len(X):.4f}  val {evaluate(net, xva, yva, multilabel, N)['acc']:.4f}  ({time.time() - t0:.0f} s)")
    return net


@torch.no_grad()
def evaluate(net, x, y, multilabel, N):
    net.eval()
    z = torch.cat([net(to_t(x[i:i + 1024])) for i in range(0, len(x), 1024)])
    if not multilabel:
        pred = z.argmax(1).numpy()
        return {"acc": float((pred == y).mean()), "correct": int((pred == y).sum())}
    p = (z > 0).numpy().astype(np.int64)
    t = (y[:, None] >> np.arange(N)) & 1
    per = (p == t).mean(0)
    exact = (p == t).all(1)
    return {"acc": float(exact.mean()), "correct": int(exact.sum()), "per": per}


# =============================================================================================== export
def write_model(path, net, C, N, head):
    with open(path, "wb") as f:
        f.write(struct.pack("<6i", 0x434E4E31, C, 8, 16, N, head))
        for t in (net.conv1.weight, net.conv1.bias, net.conv2.weight, net.conv2.bias, net.fc.weight, net.fc.bias):
            f.write(t.detach().numpy().astype("<f4").tobytes())


def write_set(path, x, y):
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", 0x434E4E44, len(x), x.shape[1], x.shape[2], x.shape[3]))
        f.write(y.astype(np.uint8).tobytes())
        f.write(x.astype(np.uint8).tobytes())


def _covers(font, ch):
    """True when [font] has a real glyph for [ch] (not the .notdef box)."""
    try:
        return bytes(font.getmask(ch)) != bytes(font.getmask("\uE000")) or ch == " "
    except Exception:
        return False


def render_text(text, fonts, h=30):
    """Draw [text] char by char with the first font of [fonts] that has each glyph (CJK font first for CJK),
    so e.g. "NIR−R" gets its minus sign and "斜边↗" its arrow. Returns a tight uint8 alpha mask."""
    im = Image.new("L", (16 + 2 * h * max(1, len(text)), 2 * h), 0)
    d = ImageDraw.Draw(im)
    x = 8
    for ch in text:
        cjk = "\u4e00" <= ch <= "\u9fff"
        order = [fonts["zh"]] if cjk else [fonts["base"], fonts["sf"], fonts["zh"]]
        f = next((ff for ff in order if _covers(ff, ch)), order[0])
        d.text((x, h // 4), ch, font=f, fill=255)
        x += f.getlength(ch)
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 8)
    return a[max(0, ys.min() - 3):ys.max() + 4, max(0, xs.min() - 3):xs.max() + 4]


def write_glyphs(path):
    sf30 = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 30); sf40 = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 40)
    zh30 = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 30, index=2)
    zh = {"zh": zh30, "base": ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 30, index=1), "sf": sf30}
    en = zh
    big = {"zh": zh30, "base": ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 40, index=1), "sf": sf40}
    items = []
    for d in range(10):
        items.append((f"digit_{d}", render_text(str(d), big)))
    for k, (z, e) in enumerate([("植被", "Vegetation"), ("水体", "Water"), ("人造草坪", "Turf"), ("屋顶", "Roof"), ("裸土", "Soil")]):
        items.append((f"cls_zh_{k}", render_text(z, zh))); items.append((f"cls_en_{k}", render_text(e, en)))
    for c in ["R", "G", "B", "NIR"]:
        items.append((f"ch_{c}", render_text(c, big)))
    hd = [("竖边", "V-edge"), ("横边", "H-edge"), ("斜边↗", "Diag ↗"), ("斜边↘", "Diag ↘"), ("模糊", "Blur"), ("锐化", "Sharpen"), ("亮点", "Spot"), ("红−蓝", "R − B")]
    hs = [("NIR−R", "NIR − R"), ("绿色", "Green"), ("水体", "Water"), ("近红外", "NIR"), ("竖边", "V-edge"), ("横边", "H-edge"), ("模糊", "Blur"), ("锐化", "Sharpen")]
    for k, (z, e) in enumerate(hd):
        items.append((f"hd_zh_{k}", render_text(z, zh))); items.append((f"hd_en_{k}", render_text(e, en)))
    for k, (z, e) in enumerate(hs):
        items.append((f"hs_zh_{k}", render_text(z, zh))); items.append((f"hs_en_{k}", render_text(e, en)))
    for k in range(8):
        items.append((f"learned_{k}", render_text(f"#{k + 1}", en)))
    with open(path, "wb") as f:
        f.write(struct.pack("<2i", 0x434E4E47, len(items)))
        for key, a in items:
            kb = key.encode("ascii")
            f.write(struct.pack("<i", len(kb))); f.write(kb)
            f.write(struct.pack("<2i", a.shape[1], a.shape[0])); f.write(a.astype(np.uint8).tobytes())
    return len(items)


# =============================================================================================== main
def preview(path):
    """--preview <dir>: save a grid of generated digits and scenes (RGB + NIR) to look at."""
    rng = np.random.default_rng(1)
    faces, bank = build_glyph_bank()
    canv = torch.tensor(np.stack([b[2] for b in bank]))[:, None]; lab = np.array([b[1] for b in bank])
    x, y = make_digits(canv, lab, 64, rng)
    g = x.reshape(8, 8, 3, 28, 28).transpose(0, 3, 1, 4, 2).reshape(8 * 28, 8 * 28, 3)
    Image.fromarray(g).resize((8 * 28 * 3, 8 * 28 * 3), Image.NEAREST).save(os.path.join(path, "digits.png"))
    s, sy = make_scenes(16, rng, showcase=True)
    rgb = s[:, :3].reshape(4, 4, 3, 28, 28).transpose(0, 3, 1, 4, 2).reshape(112, 112, 3)
    nir = s[:, 3].reshape(4, 4, 28, 28).transpose(0, 2, 1, 3).reshape(112, 112)
    both = np.concatenate([rgb, np.stack([nir] * 3, -1)], 1)
    Image.fromarray(both).resize((224 * 4, 112 * 4), Image.NEAREST).save(os.path.join(path, "scenes.png"))
    print("faces", len(faces), "labels", y.tolist()[:16], "scene labels", [bin(v) for v in sy.tolist()])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        preview(sys.argv[2]); return
    if len(sys.argv) > 1 and sys.argv[1] == "--glyphs":      # re-render the label glyphs only (no retraining)
        print("glyphs:", write_glyphs(os.path.join(OUT, "glyphs.bin"))); return
    t0 = time.time()
    rng = np.random.default_rng(20261027)
    faces, bank = build_glyph_bank()
    nf = len(faces)
    order = rng.permutation(nf)
    test_faces = set(order[: max(8, nf // 5)].tolist())
    tr_idx = [i for i, b in enumerate(bank) if b[0] not in test_faces]
    te_idx = [i for i, b in enumerate(bank) if b[0] in test_faces]
    canv = torch.tensor(np.stack([b[2] for b in bank]))[:, None]
    lab = np.array([b[1] for b in bank])
    print(f"fonts: {nf} usable faces ({nf - len(test_faces)} train / {len(test_faces)} held out), glyph canvases {len(bank)}")
    xtr, ytr = make_digits(canv[tr_idx], lab[tr_idx], 60000, rng)
    xva, yva = make_digits(canv[tr_idx], lab[tr_idx], 3000, rng)          # val: training fonts, fresh warps
    xte, yte = make_digits(canv[te_idx], lab[te_idx], 5000, rng)          # TEST: held-out fonts only
    print(f"digits generated ({time.time() - t0:.0f} s)")

    s_tr, sy_tr = make_scenes(40000, rng)
    s_va, sy_va = make_scenes(2000, rng)
    s_te, sy_te = make_scenes(5000, rng, showcase=True)
    print(f"scenes generated ({time.time() - t0:.0f} s); label frequency (test):",
          [round(float(((sy_te >> c) & 1).mean()), 3) for c in range(5)])

    N_DIG_SHIP, N_SCN_SHIP = 250, 160
    metrics = {}
    nets = {}
    jobs = [
        ("digits_learned", 3, 10, False, None, (xtr, ytr, xva, yva, xte, yte)),
        ("digits_hand", 3, 10, False, digit_bank(), (xtr, ytr, xva, yva, xte, yte)),
        ("scene4_learned", 4, 5, True, None, (s_tr, sy_tr, s_va, sy_va, s_te, sy_te)),
        ("scene4_hand", 4, 5, True, scene_bank(4), (s_tr, sy_tr, s_va, sy_va, s_te, sy_te)),
        ("scene3_learned", 3, 5, True, None, (s_tr[:, :3], sy_tr, s_va[:, :3], sy_va, s_te[:, :3], sy_te)),
        ("scene3_hand", 3, 5, True, scene_bank(3), (s_tr[:, :3], sy_tr, s_va[:, :3], sy_va, s_te[:, :3], sy_te)),
    ]
    for k, (name, C, N, ml, fixed, (a, b, c, d, e, f)) in enumerate(jobs):
        net = train(name, C, N, a, b, c, d, ml, fixed, epochs=14 if not ml else 16, seed=100 + k)
        r = evaluate(net, e, f, ml, N)
        ship = evaluate(net, e[:N_DIG_SHIP if not ml else N_SCN_SHIP], f[:N_DIG_SHIP if not ml else N_SCN_SHIP], ml, N)
        metrics[f"{name}.heldout_n"] = len(e)
        metrics[f"{name}.heldout_acc"] = round(r["acc"], 4)
        metrics[f"{name}.shipped_correct"] = ship["correct"]
        if ml:
            for cc, nm in enumerate(["veg", "water", "turf", "roof", "soil"]):
                metrics[f"{name}.label_acc_{nm}"] = round(float(r["per"][cc]), 4)
        nets[name] = (net, C, N, ml)
        print(f"{name}: held-out acc {r['acc']:.4f} on {len(e)}" + (f"  per-label {np.round(r['per'], 4).tolist()}" if ml else "")
              + f"  | shipped subset {ship['correct']}/{N_DIG_SHIP if not ml else N_SCN_SHIP}")
        write_model(os.path.join(OUT, f"{name}.bin"), net, C, N, 1 if ml else 0)

    # display order: the demo cycles through the first 20 shipped digits. Put 2 per class (0..9) first,
    # chosen among the ones the learned model reads confidently (the rest keep their random order; the
    # shipped SET is unchanged, so its accuracy is not affected).
    net = nets["digits_learned"][0]; net.eval()
    with torch.no_grad():
        pr = F.softmax(net(to_t(xte[:N_DIG_SHIP])), 1).numpy()
    first = []
    for d in range(10):
        ok = [i for i in range(N_DIG_SHIP) if yte[i] == d and pr[i].argmax() == d and pr[i, d] > 0.9]
        first += ok[:2]
    rest = [i for i in range(N_DIG_SHIP) if i not in first]
    perm = np.array(first + rest + list(range(N_DIG_SHIP, len(xte))))
    xte, yte = xte[perm], yte[perm]
    print("display digits:", yte[:20].tolist())
    write_set(os.path.join(OUT, "digits_eval.bin"), xte[:N_DIG_SHIP], yte[:N_DIG_SHIP])
    write_set(os.path.join(OUT, "scenes_eval.bin"), s_te[:N_SCN_SHIP], sy_te[:N_SCN_SHIP])

    # reference vectors: full intermediates for one sample, logits for the first M samples
    M = 24
    with open(os.path.join(OUT, "ref.bin"), "wb") as fo:
        fo.write(struct.pack("<2i", 0x434E4E52, len(nets)))
        for name, (net, C, N, ml) in nets.items():
            xs = (xte if not ml else s_te)[:, :C]
            ys = yte if not ml else sy_te
            n_ship = N_DIG_SHIP if not ml else N_SCN_SHIP
            si = 7 if not ml else 0
            net.eval()
            with torch.no_grad():
                a1, p1, a2, p2, z = net(to_t(xs[si:si + 1]), all_=True)
                zs = net(to_t(xs[:M]))
            nb = name.encode("ascii")
            fo.write(struct.pack("<i", len(nb))); fo.write(nb); fo.write(struct.pack("<i", si))
            for t in (a1, p1, a2, p2, z):
                fo.write(t[0].numpy().astype("<f4").tobytes())
            fo.write(struct.pack("<i", M)); fo.write(zs.numpy().astype("<f4").tobytes())
            fo.write(struct.pack("<i", evaluate(net, xs[:n_ship], ys[:n_ship], ml, N)["correct"]))

    ng = write_glyphs(os.path.join(OUT, "glyphs.bin"))
    with open(os.path.join(OUT, "metrics.txt"), "w") as f:
        f.write("# written by tools/train_cnn.py — held-out = fonts / scenes never seen in training\n")
        for k, v in metrics.items():
            f.write(f"{k}={v}\n")
    print(f"glyphs: {ng}")
    tot = 0
    for fn in sorted(os.listdir(OUT)):
        sz = os.path.getsize(os.path.join(OUT, fn)); tot += sz
        print(f"  {fn:22s} {sz:9d} B")
    print(f"assets total {tot / 1e6:.3f} MB, done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
