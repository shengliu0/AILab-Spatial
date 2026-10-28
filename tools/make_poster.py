#!/usr/bin/env python3
"""Store poster for AI 算法实验室 Spatial, 2560 x 1440 (16:9): title + features on the left, a collage of HD experiment renders on
the right, on the icon's deep-space background with a sweeping spectrum orbit.

Inputs:  docs/store/icon_1024.png (tools/make_icons.py), app/build/showcase/<id>.png (ShowcaseRenderTest)
Output:  docs/store/poster_2560x1440.jpg
Fonts:   macOS system fonts (Hiragino Sans GB, Helvetica Neue), rasterised into the image.
Run from SolarSpatial/.
"""
import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2560, 1440
HIRA = "/System/Library/Fonts/Hiragino Sans GB.ttc"
HELV = "/System/Library/Fonts/HelveticaNeue.ttc"
CARDS = ["gpt", "diffusion", "cnn", "descent", "mlp", "resnet"]
rng = np.random.default_rng(7)


def background():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((x - 1500) / W, (y - 560) / H)
    c0, c1 = np.array([24, 34, 82], np.float32), np.array([4, 6, 18], np.float32)
    t = np.clip(r / 0.85, 0, 1)[..., None]
    img = c0 + (c1 - c0) * t
    # warm glow on the left behind the title (echo of the icon's sun)
    g = np.exp(-((x - 330) ** 2 + (y - 300) ** 2) / (2 * 420.0 ** 2))
    img += np.array([90, 55, 20], np.float32) * g[..., None] * 0.55
    for _ in range(260):
        sx, sy = rng.uniform(0, W), rng.uniform(0, H)
        sr = rng.uniform(0.6, 1.6); b = rng.uniform(60, 220)
        x0, x1, y0, y1 = int(max(sx - 6, 0)), int(min(sx + 7, W)), int(max(sy - 6, 0)), int(min(sy + 7, H))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += (b * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * sr ** 2)))[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")


def spectrum_orbit(img):
    """A huge tilted ellipse in spectrum colours sweeping behind everything."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy, a, b, rot = 1250, 820, 1500, 430, math.radians(-12)
    pts = []
    n = 900
    for k in range(n + 1):
        t = 2 * math.pi * k / n
        px, py = a * math.cos(t), b * math.sin(t)
        pts.append((cx + px * math.cos(rot) - py * math.sin(rot), cy + px * math.sin(rot) + py * math.cos(rot), t))
    for k in range(n):
        (x0, y0, t), (x1, y1, _) = pts[k], pts[k + 1]
        s = (t % (2 * math.pi)) / (2 * math.pi)
        h = 0.76 * (1 - abs(2 * s - 1))
        r, g, bb = colorsys.hsv_to_rgb(h, 0.75, 1.0)
        alpha = 150 if math.sin(t) > 0 else 70
        d.line([(x0, y0), (x1, y1)], fill=(int(r * 255), int(g * 255), int(bb * 255), alpha), width=9)
    glow = layer.filter(ImageFilter.GaussianBlur(16))
    img.alpha_composite(glow)
    img.alpha_composite(layer)


def rounded(im, radius):
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius, fill=255)
    out = im.convert("RGBA"); out.putalpha(m)
    return out


def shadow(size, radius, blur=26, alpha=150):
    s = Image.new("RGBA", (size[0] + blur * 4, size[1] + blur * 4), (0, 0, 0, 0))
    ImageDraw.Draw(s).rounded_rectangle([blur * 2, blur * 2, blur * 2 + size[0], blur * 2 + size[1]], radius, fill=(0, 0, 0, alpha))
    return s.filter(ImageFilter.GaussianBlur(blur))


def card(img, src, box, radius=26):
    x, y, w, h = box
    im = Image.open(src).convert("RGB")
    # crop the render to the card's aspect, keeping the centre
    sw, sh = im.size; target = w / h
    if sw / sh > target: nw = int(sh * target); im = im.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else: nh = int(sw / target); im = im.crop((0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh))
    im = im.resize((w, h), Image.LANCZOS)
    sh_ = shadow((w, h), radius)
    img.alpha_composite(sh_, (x - 52, y - 40))
    img.alpha_composite(rounded(im, radius), (x, y))
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))            # translucent strokes need their own layer:
    ImageDraw.Draw(ov).rounded_rectangle([x, y, x + w - 1, y + h - 1], radius, outline=(255, 255, 255, 46), width=2)
    img.alpha_composite(ov)                                  # ImageDraw on RGBA overwrites, it does not blend


def chip(img, x, y, text, font, fill=(255, 255, 255, 28), outline=(150, 200, 255, 110)):
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    w, h = r - l + 56, 76
    d.rounded_rectangle([x, y, x + w, y + h], 38, fill=fill, outline=outline, width=2)
    img.alpha_composite(ov)
    ImageDraw.Draw(img).text((x + 28, y + h / 2), text, font=font, fill=(235, 240, 255, 255), anchor="lm")
    return w


def main():
    img = background()
    spectrum_orbit(img)

    # ---- collage (right): 2 columns x 3 rows
    cw, ch, gap = 640, 360, 34
    x0, y0 = W - 2 * cw - gap - 120, (H - 3 * ch - 2 * gap) // 2
    for i, cid in enumerate(CARDS):
        cx = x0 + (i % 2) * (cw + gap); cy = y0 + (i // 2) * (ch + gap)
        card(img, f"app/build/showcase/{cid}.png", (cx, cy, cw, ch))

    # ---- title block (left)
    d = ImageDraw.Draw(img)
    icon = Image.open("docs/store/icon_1024.png").resize((230, 230), Image.LANCZOS)
    img.alpha_composite(shadow((230, 230), 52, 20, 170), (140 - 40, 170 - 40))
    img.alpha_composite(rounded(icon, 52), (140, 170))
    f_title = ImageFont.truetype(HIRA, 104, index=2)
    f_title_en = ImageFont.truetype(HELV, 60, index=10)
    f_sub = ImageFont.truetype(HIRA, 54, index=0)
    f_en = ImageFont.truetype(HELV, 34, index=0)
    f_chip = ImageFont.truetype(HIRA, 36, index=2)
    d.text((140, 470), "AI 算法实验室", font=f_title, fill=(255, 255, 255, 255))
    d.text((140 + d.textlength("AI 算法实验室", font=f_title) + 26, 470 + 104 - 56), "Spatial", font=f_title_en, fill=(150, 200, 255, 255))
    d.text((144, 660), "9 个真实运行的小模型，", font=f_sub, fill=(220, 228, 250, 255))
    d.text((144, 736), "把神经网络拆开来看。", font=f_sub, fill=(220, 228, 250, 255))
    d.text((146, 840), "AI Lab Spatial · 9 live, inspectable models for PICO", font=f_en, fill=(160, 175, 210, 255))
    x, y = 140, 950
    for text in ["注视 + 捏合 直接操作", "看见注意力与梯度"]:
        x += chip(img, x, y, text, f_chip) + 22
    x, y = 140, 1048
    for text in ["真实训练 · 与 PyTorch 一致", "中英双语"]:
        x += chip(img, x, y, text, f_chip) + 22
    d = ImageDraw.Draw(img)
    f_names = ImageFont.truetype(HIRA, 30, index=0)
    d.text((146, 1214), "神经网络 · 梯度下降 · 反向传播 · 核技巧 · CNN", font=f_names, fill=(140, 155, 195, 255))
    d.text((146, 1262), "ResNet · 词向量 · GPT · 扩散模型", font=f_names, fill=(140, 155, 195, 255))

    out = img.convert("RGB")
    out.save("docs/store/poster_2560x1440.jpg", quality=92, optimize=True, progressive=True)
    print("poster written", out.size)


if __name__ == "__main__":
    main()
