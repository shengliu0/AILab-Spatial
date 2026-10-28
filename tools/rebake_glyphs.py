#!/usr/bin/env python3
"""Re-render every SHIPPED text bitmap from open fonts (Noto Sans SC / Noto Sans, SIL OFL — tools/fonts/).

The training scripts originally rasterised labels with macOS system fonts, whose licences do not cover
redistributing glyph images. This re-bakes only the glyph assets — same characters, same file formats,
same layouts — without retraining any model:
  assets/gpt/glyphs.bin, assets/cnn/glyphs.bin, assets/mlp/glyphs.bin, assets/embedding/words.bin
Every macOS font path the scripts ask for is mapped to the matching Noto face (CJK -> Noto Sans SC,
everything else -> Noto Sans), at a weight close to the original. Run from AISpatial/: python3 tools/rebake_glyphs.py
"""
import os, struct, sys, zlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
SC, LATIN = os.path.join(FONTS, "NotoSansSC-VF.ttf"), os.path.join(FONTS, "NotoSans-VF.ttf")
_truetype = ImageFont.truetype
CJK_HINTS = ("Hiragino", "STHeiti", "PingFang", "Songti", "Arial Unicode", "Apple Symbols")


def noto(path, size=10, index=0, **kw):
    """Drop-in for ImageFont.truetype: system fonts -> Noto (open licence)."""
    p = str(path)
    if not p.startswith("/System/") and not p.startswith("/Library/"):
        return _truetype(path, size, index=index, **kw)
    cjk = any(h in p for h in CJK_HINTS)
    f = _truetype(SC if cjk else LATIN, size)
    bold = any(h in p for h in ("Bold", "SFNSRounded", "Medium")) or index in (1, 2) and not cjk
    f.set_variation_by_name("SemiBold" if bold else "Medium")
    return f


ImageFont.truetype = noto          # every script below now draws with Noto
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
A = lambda *p: os.path.join(ROOT, "app", "src", "main", "assets", *p)


def gpt():
    import train_gpt as g
    g.FONTS = [(SC, 0), (LATIN, 0)]
    g.ASCII_FONT = (LATIN, 0)
    path = A("gpt", "glyphs.bin")
    with open(path, "rb") as f:
        assert f.read(4) == b"GLY1"; cell, n = struct.unpack("<ii", f.read(8)); cps = struct.unpack(f"<{n}i", f.read(4 * n))
    chars = [chr(c) for c in cps]
    # Noto lacks a few symbols (e.g. the newline arrow "↵"): those cells are drawn as vector shapes below
    real_font_for = g.font_for
    g.font_for.__defaults__[0].clear()
    missing = []
    def font_for(c):
        try: return real_font_for(c)
        except SystemExit: missing.append(c); return _truetype(LATIN, 40)
    g.font_for = font_for
    g.write_glyphs(chars, path)
    if missing:
        with open(path, "rb") as f: raw = f.read()
        hdr = 12 + 4 * n; zlen = struct.unpack("<i", raw[hdr:hdr + 4])[0]
        cells = bytearray(zlib.decompress(raw[hdr + 4:hdr + 4 + zlen]))
        for c in missing:
            k = chars.index(c); cells[k * cell * cell:(k + 1) * cell * cell] = draw_symbol(c, cell)
        z = zlib.compress(bytes(cells), 9)
        with open(path, "wb") as f: f.write(raw[:hdr]); f.write(struct.pack("<i", len(z))); f.write(z)
    print(f"gpt: {len(chars)} glyphs re-rendered ({len(missing)} drawn as shapes: {''.join(missing)})")


def draw_symbol(c, cell):
    """Vector stand-ins for symbols no open font here has, drawn at 4x and downsampled (anti-aliased)."""
    S = cell * 4; im = Image.new("L", (S, S), 0); d = ImageDraw.Draw(im); w = int(S * 0.075)
    if c == "↵":       # return arrow: down from the top right, then left, with an arrowhead
        x1, y0, y1, x0 = S * 0.72, S * 0.24, S * 0.62, S * 0.26
        d.line([(x1, y0), (x1, y1), (x0 + w, y1)], fill=255, width=w, joint="curve")
        h = S * 0.13; d.polygon([(x0 - w * 0.3, y1), (x0 + h, y1 - h), (x0 + h, y1 + h)], fill=255)
    elif c == "▶":
        d.polygon([(S * 0.3, S * 0.22), (S * 0.3, S * 0.78), (S * 0.78, S * 0.5)], fill=255)
    else:
        raise SystemExit(f"no font and no shape for {c!r}")
    return im.resize((cell, cell), Image.LANCZOS).tobytes()


def cnn():
    import train_cnn as c
    print("cnn:", c.write_glyphs(A("cnn", "glyphs.bin")), "items re-rendered")


def mlp():
    gl = np.zeros((10, 48, 48), dtype=np.uint8)
    f = noto("/System/Library/Fonts/SFNSRounded.ttf", 40)
    for d in range(10):
        im = Image.new("L", (48, 48), 0); dr = ImageDraw.Draw(im)
        l, t, r, b = dr.textbbox((0, 0), str(d), font=f)
        dr.text(((48 - (r - l)) / 2 - l, (48 - (b - t)) / 2 - t), str(d), fill=255, font=f)
        gl[d] = np.asarray(im)
    gl.tofile(A("mlp", "glyphs.bin")); print("mlp: 10 digit glyphs re-rendered")


def embedding():
    import train_embedding as e
    words = e.build_vocab()
    e.bake_words(words, A("embedding", "words.bin")); print(f"embedding: atlas for {sum(1 for w in words if w[3])} shown words re-rendered")


if __name__ == "__main__":
    for p in (SC, LATIN):
        if not os.path.isfile(p): raise SystemExit(f"missing {p} — see tools/fonts/README.md")
    gpt(); cnn(); mlp(); embedding()
