# Fonts for the shipped glyph bitmaps

Every text bitmap that ships in `app/src/main/assets/` (GPT / CNN / MLP / embedding labels) is rendered
from these open fonts by `tools/rebake_glyphs.py`, under the SIL Open Font License 1.1 (`OFL.txt`):

- `NotoSansSC-VF.ttf` — Noto Sans SC (variable), https://github.com/google/fonts/tree/main/ofl/notosanssc
- `NotoSans-VF.ttf` — Noto Sans (variable), https://github.com/google/fonts/tree/main/ofl/notosans

The font files are not committed (17 MB); download them into this folder before re-baking.
(Training data — the synthetic digit images — may use any local fonts: only trained weights ship from it.)
