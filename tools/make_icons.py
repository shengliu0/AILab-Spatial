#!/usr/bin/env python3
"""Icon artwork for AI 算法实验室 Spatial — one design, every format.

The motif: a network as glowing tensor planes — four layers that shrink as they go deeper (a CNN), joined by
light threads, on deep navy, with the spectrum orbit that marks the family (物理 / 化学 / 数学 / AI 实验室 Spatial).

Outputs (run from AISpatial/): PICO 3-D icon layers + relief maps, round launcher, adaptive foreground,
docs/store/icon_1024.png (same file set as the sister apps' tools/make_icons.py).
"""
import colorsys
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_edt

SS = 2
N = 1024 * SS
Y, X = (np.mgrid[0:N, 0:N].astype(np.float32) + 0.5) / SS
rng = np.random.default_rng(20261029)


def lin(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)


def over(dst, dst_a, src, src_a):
    a = src_a[..., None]
    out_a = src_a + dst_a * (1 - src_a)
    return (src * a + dst * dst_a[..., None] * (1 - a)) / np.maximum(out_a[..., None], 1e-6), out_a


def blur(a, r):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r * SS)), np.float32) / 255.0


def background():
    r = np.hypot(X - 512, Y - 450) / 724.0
    c0, c1, c2 = lin([22, 46, 104]), lin([8, 18, 46]), lin([3, 5, 14])
    t = np.clip(r, 0, 1)[..., None]
    rgb = np.where(t < 0.45, c0 + (c1 - c0) * (t / 0.45), c1 + (c2 - c1) * ((t - 0.45) / 0.55))
    # graph paper, fading from the centre
    g = np.zeros((N, N), np.float32)
    step = 64 * SS
    for k in range(0, N, step):
        g[:, k:k + SS] = 1; g[k:k + SS, :] = 1
    fade = np.exp(-(((X - 512) ** 2 + (Y - 520) ** 2) / (2 * 400.0 ** 2)))
    rgb = rgb + lin([90, 170, 255]) * (g * fade * 0.05)[..., None]
    for _ in range(40):
        sx, sy = rng.uniform(40, 984, 2); sr = rng.uniform(0.6, 1.4); br = rng.uniform(0.3, 0.8)
        rgb = rgb + lin([220, 215, 255]) * (np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * sr ** 2)) * br * 0.7)[..., None]
    return rgb


def orbit(cx=512.0, cy=560.0, a=430.0, b=150.0, rot=math.radians(-12)):
    c, s = math.cos(-rot), math.sin(-rot)
    xr = (X - cx) * c - (Y - cy) * s; yr = (X - cx) * s + (Y - cy) * c
    F = (xr / a) ** 2 + (yr / b) ** 2 - 1
    G = 2 * np.sqrt((xr / a ** 2) ** 2 + (yr / b ** 2) ** 2)
    dist = np.abs(F / np.maximum(G, 1e-6))
    al = np.clip((6.0 - dist) * SS + 0.5, 0, 1)
    sfrac = (np.arctan2(yr / b, xr / a) % (2 * np.pi)) / (2 * np.pi)
    hue = 0.76 * (1 - np.abs(2 * sfrac - 1))
    lut = np.array([colorsys.hsv_to_rgb(h, 0.78, 1.0) for h in np.linspace(0, 0.76, 256)], np.float32)
    rgb = lin(lut[np.clip((hue / 0.76 * 255).astype(int), 0, 255)] * 255)
    front = yr > 0
    return rgb, al * np.where(front, 0, 0.6), al * np.where(front, 1, 0), blur(al, 8) * 0.8


# ------------------------------------------------------------------ the network: layers as tensor planes
LAYERS = [  # (x position, half-size, grid n, colour)
    (-300.0, 190.0, 5, [70, 220, 255]),
    (-95.0, 150.0, 4, [120, 140, 255]),
    (95.0, 110.0, 3, [205, 95, 245]),
    (270.0, 70.0, 2, [255, 190, 70]),
]
CX, CY = 548.0, 540.0


def plane_pts(x, h, u, v):
    """Point on layer plane (x) at local coords u, v in [-1, 1] (oblique view: depth leans up-right)."""
    return (CX + x + u * h * 0.42, CY - v * h + u * h * 0.30)


def draw_network():
    im = Image.new("RGBA", (N * 2, N * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    glow = Image.new("RGBA", (N * 2, N * 2), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    k = 2 * SS
    nodes = []
    for (x, h, n, col) in LAYERS:
        pts = [plane_pts(x, h, -1, -1), plane_pts(x, h, 1, -1), plane_pts(x, h, 1, 1), plane_pts(x, h, -1, 1)]
        d.polygon([(px * k, py * k) for px, py in pts], fill=tuple(col) + (60,))
        d.line([(px * k, py * k) for px, py in pts + [pts[0]]], fill=tuple(col) + (230,), width=4 * SS)
        layer = []
        for i in range(n):
            for j in range(n):
                u = -0.7 + 1.4 * (i + 0.5) / n if n > 1 else 0.0
                v = -0.7 + 1.4 * (j + 0.5) / n if n > 1 else 0.0
                if n > 1: u = -0.72 + 1.44 * i / (n - 1); v = -0.72 + 1.44 * j / (n - 1)
                layer.append((plane_pts(x, h, u, v), col))
        nodes.append(layer)
    # threads between consecutive layers (deterministic sparse fan)
    for li in range(len(nodes) - 1):
        a, b = nodes[li], nodes[li + 1]
        for ai, (p, c) in enumerate(a):
            for bj in range(len(b)):
                if (ai * 7 + bj * 3) % 4 != 0: continue
                q, c2 = b[bj]
                cm = tuple(int((c[t] + c2[t]) / 2) for t in range(3))
                d.line([(p[0] * k, p[1] * k), (q[0] * k, q[1] * k)], fill=cm + (95,), width=2 * SS)
    for layer in nodes:
        for (p, col) in layer:
            r = 13.0
            gd.ellipse([(p[0] - r * 2.2) * k, (p[1] - r * 2.2) * k, (p[0] + r * 2.2) * k, (p[1] + r * 2.2) * k], fill=tuple(col) + (150,))
            d.ellipse([(p[0] - r) * k, (p[1] - r) * k, (p[0] + r) * k, (p[1] + r) * k], fill=(245, 250, 255, 255))
            d.ellipse([(p[0] - r * 0.62) * k, (p[1] - r * 0.62) * k, (p[0] + r * 0.62) * k, (p[1] + r * 0.62) * k], fill=tuple(col) + (255,))
    glow = glow.filter(ImageFilter.GaussianBlur(10 * k))
    glow.alpha_composite(im)
    out = glow.resize((N, N), Image.LANCZOS)
    a = np.asarray(out, np.float32) / 255.0
    return lin(a[..., :3] * 255), a[..., 3]


def sdf_png(dist_inside):
    v = np.clip(128 + 2.55 * dist_inside, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([v, v, v, v]))


def to_img(rgb, a=None):
    s = (srgb(rgb) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(s) if a is None else Image.fromarray(np.dstack([s, (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)]))
    return im.resize((1024, 1024), Image.LANCZOS)


def main():
    os.makedirs("docs/store", exist_ok=True)
    disc_a = np.clip((512 - np.hypot(X - 512, Y - 512)) * SS + 0.5, 0, 1)
    one = np.ones((N, N), np.float32)
    back = background()
    o_rgb, o_back, o_front, o_glow = orbit(cy=560.0, a=440.0, b=140.0)
    back = back + o_rgb * (o_glow * 0.5)[..., None]
    back, _ = over(back, one, o_rgb, o_back)
    g = np.exp(-((X - 512) ** 2 + (Y - 540) ** 2) / (2 * 280.0 ** 2))
    back = back + lin([80, 170, 255]) * (g * 0.14)[..., None]
    n_rgb, n_a = draw_network()
    front, front_a = over(np.zeros((N, N, 3), np.float32), np.zeros((N, N), np.float32), n_rgb, n_a)
    front, front_a = over(front, front_a, o_rgb, o_front)
    to_img(back, disc_a).save("app/src/main/res/drawable/icon_3d_layer_0.png", optimize=True)
    to_img(front, front_a).save("app/src/main/res/drawable/icon_3d_layer_1.png", optimize=True)
    g1 = np.mgrid[0:1024, 0:1024].astype(np.float32) + 0.5
    sdf_png(512 - np.hypot(g1[1] - 512, g1[0] - 512)).save("app/src/main/res/drawable/icon_3d_sdf_0.png", optimize=True)
    sil = np.asarray(Image.fromarray((n_a * 255).astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)) > 110
    sdf_png(np.where(sil, distance_transform_edt(sil), -distance_transform_edt(~sil))).save("app/src/main/res/drawable/icon_3d_sdf_1.png", optimize=True)
    flat, _ = over(back, one, front, front_a)
    to_img(flat, disc_a).save("app/src/main/res/mipmap-anydpi/ic_spatial_launcher.png", optimize=True)
    sq = to_img(flat); sq.save("docs/store/icon_1024.png", optimize=True)
    sq.resize((1424, 1424), Image.LANCZOS).save("app/src/main/res/drawable/ic_launcher_foreground.png", optimize=True)
    print("icons written")


if __name__ == "__main__":
    main()
