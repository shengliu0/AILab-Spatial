#!/usr/bin/env python3
"""Menu-card thumbnails from the headless review renders.

Run after `./gradlew :app:testDebugUnitTest --tests "*LabReviewTest*"`:
    python3 tools/make_thumbs.py
For every app/build/review/viewer/demo_<id>_viewer.png (variants with '-' are skipped) it finds the
content bounding box (pixels that differ from the dark backdrop), frames it at the card's 16:10
aspect with a margin, and writes app/src/main/res/drawable-nodpi/thumb_<id>.png (480x300).
A new demo gets its card picture for free: render it, run this, rebuild.
"""
import glob, os, re
from PIL import Image, ImageChops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'app/build/review/viewer')
DST = os.path.join(ROOT, 'app/src/main/res/drawable-nodpi')
W, H = 480, 300
BG = (14, 17, 21)          # SoftRender.BACKDROP in sRGB bytes (0.055, 0.06, 0.075)

os.makedirs(DST, exist_ok=True)
made = 0
for p in sorted(glob.glob(os.path.join(SRC, 'demo_*_viewer.png'))):
    m = re.match(r'demo_([a-z0-9]+)_viewer\.png$', os.path.basename(p))
    if not m:
        continue                                     # skip state variants like demo_solar-closeup_viewer
    im = Image.open(p).convert('RGB')
    diff = ImageChops.difference(im, Image.new('RGB', im.size, BG)).convert('L').point(lambda v: 255 if v > 18 else 0)
    box = diff.getbbox() or (0, 0, im.width, im.height)
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * 1.12, (y1 - y0) * 1.12        # 6 % margin each side
    if w / h > W / H: h = w * H / W
    else: w = h * W / H
    crop = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    out = Image.new('RGB', (crop[2] - crop[0], crop[3] - crop[1]), BG)
    out.paste(im.crop((max(0, crop[0]), max(0, crop[1]), min(im.width, crop[2]), min(im.height, crop[3]))),
              (max(0, -crop[0]), max(0, -crop[1])))
    out.resize((W, H), Image.LANCZOS).save(os.path.join(DST, f'thumb_{m.group(1)}.png'), optimize=True)
    made += 1
print(f'{made} thumbnails -> {DST}')
