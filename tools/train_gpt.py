#!/usr/bin/env python3
"""Train the tiny character-level GPT behind the "gpt" demo (Transformer · GPT).

CORPUS. Generated procedurally by the small grammar below (written for this demo; no external text).
Every line is one tiny story built from templates, with structure chosen so ATTENTION has something
visible to do:
  * copy               "猫把书给了狗。狗谢谢猫。"              (the thanker copies the giver's name)
  * brackets           "狗说：「我喜欢花。」"  "《小船》"        (a closer needs to find its opener)
  * successor          "今天是星期三，明天是星期四。" "数数：五六七八。"
  * copy from a list   "羊有三样东西：茶、伞、信。羊最喜欢伞。"
Names and objects are drawn with Zipf weights, so many next-character distributions are spread
(good for the temperature knob), while the copy positions are almost certain.

MODEL. GPT-2 style, pre-LayerNorm, causal: 2 layers x 4 heads, d_model 64, context 32, MLP 256
(tanh-GELU), learned positions, output head tied to the token embedding. Trained on the CPU with
fixed seeds (deterministic).

OUTPUTS (all little-endian):
  app/src/main/assets/gpt/model.bin   int32 header, UTF-8 vocab, float32 weights, PCA, metrics (layout: write_model)
  app/src/main/assets/gpt/glyphs.bin  glyph atlas: alpha bitmaps of every vocab char + ASCII (layout: write_glyphs)
  app/src/test/resources/gpt/ref.bin  PyTorch reference logits + attention for a few prompts (float64 forward)
  app/src/test/resources/gpt/val.txt  held-out generated text (different seed) for the perplexity test
  app/src/test/resources/gpt/train_counts.bin  int32 count per vocab id in the training text (unigram baseline)

    python3 tools/train_gpt.py
"""
import math, os, random, struct, sys, time, zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app/src/main/assets/gpt")
TESTRES = os.path.join(ROOT, "app/src/test/resources/gpt")

CTX, D, N_LAYER, N_HEAD, DFF, N_PCA = 32, 64, 2, 4, 256, 6
TRAIN_CHARS, VAL_CHARS = 400_000, 12_000
STEPS, BATCH, LR = 4000, 96, 3e-3

# ---------------------------------------------------------------------------------------------- grammar
NAMES = ["猫", "狗", "兔", "熊", "鸭", "鹅", "羊", "牛", "马", "猴"]     # one character each: every copy stays in view
OBJECTS = [("书", "本"), ("笔", "支"), ("花", "朵"), ("球", "个"), ("伞", "把"), ("信", "封"), ("糖", "块"),
           ("茶", "杯"), ("苹果", "个"), ("香蕉", "根"), ("帽子", "顶"), ("画", "张"), ("风筝", "只"), ("杯子", "个")]
PLACES = ["学校", "公园", "商店", "书店", "河边", "花园", "山上", "家里"]
TITLES = ["小船", "月亮", "大海", "星星", "森林", "火车", "雨天", "白云"]
DAYS = "一二三四五六日"
NUMS = "一二三四五六七八九十"
AMOUNT = "两三四五"


def zipf(n, s=0.9):
    return [1.0 / (i + 1) ** s for i in range(n)]


class Grammar:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def name(self, *avoid):
        while True:
            n = self.r.choices(NAMES, zipf(len(NAMES)))[0]
            if n not in avoid:
                return n

    def obj(self, *avoid):
        while True:
            o = self.r.choices(OBJECTS, zipf(len(OBJECTS)))[0]
            if o[0] not in avoid:
                return o

    def story(self):
        r = self.r
        kind = r.choices(["give", "ask", "like", "days", "count", "list", "place", "book"],
                         [0.22, 0.10, 0.07, 0.10, 0.08, 0.33, 0.05, 0.05])[0]
        if kind == "give":
            a = self.name(); b = self.name(a); o = self.obj()[0]
            first = f"{a}把{o}给了{b}。"
            return first + (f"{b}谢谢{a}。" if r.random() < 0.7 else f"{b}说：「谢谢{a}！」")
        if kind == "ask":
            a = self.name(); b = self.name(a); o, m = self.obj()
            q = f"{a}问{b}：「你有{o}吗？」"
            if r.random() < 0.6:
                return q + f"{b}说：「有，我有{r.choice(AMOUNT)}{m}{o}。」"
            return q + f"{b}说：「没有。」"
        if kind == "like":
            return f"{self.name()}说：「我喜欢{self.obj()[0]}。」"
        if kind == "days":
            k = r.randrange(7)
            if r.random() < 0.6:
                return f"今天是星期{DAYS[k]}，明天是星期{DAYS[(k + 1) % 7]}。"
            return f"昨天是星期{DAYS[k]}，今天是星期{DAYS[(k + 1) % 7]}。"
        if kind == "count":
            s = r.randrange(0, 7); n = r.randint(3, min(7, 10 - s))
            return "数数：" + NUMS[s:s + n] + "。"
        if kind == "list":
            a = self.name(); k = r.choice([2, 3]); items = []
            while len(items) < k:
                items.append(self.obj(*items)[0])
            return f"{a}有{'两' if k == 2 else '三'}样东西：{'、'.join(items)}。{a}最喜欢{r.choice(items)}。"
        if kind == "place":
            a = self.name(); b = self.name(a); p = r.choice(PLACES)
            return f"{a}去{p}。{a}在{p}看见了{b}。"
        a = self.name(); b = self.name(a); t = r.choice(TITLES)
        return f"{a}在读《{t}》。{b}也想读《{t}》。"

    def text(self, n_chars):
        out, n = [], 0
        while n < n_chars:
            s = self.story() + "\n"
            out.append(s); n += len(s)
        return "".join(out)


# ---------------------------------------------------------------------------------------------- model
def gelu(x):
    return F.gelu(x, approximate="tanh")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D); self.fc = nn.Linear(D, DFF); self.fc2 = nn.Linear(DFF, D)

    def forward(self, x, mask):
        B, T, C = x.shape; hd = C // N_HEAD
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, N_HEAD, hd).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hd).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(~mask[:T, :T], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.fc2(gelu(self.fc(self.ln2(x))))
        return x, att


class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.wte = nn.Embedding(V, D); self.wpe = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(D)
        self.register_buffer("mask", torch.tril(torch.ones(CTX, CTX, dtype=torch.bool)))
        nn.init.normal_(self.wte.weight, std=0.05); nn.init.normal_(self.wpe.weight, std=0.02)

    def forward(self, idx, keep=False):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        rows, atts = [x], []
        for b in self.blocks:
            x, a = b(x, self.mask)
            rows.append(x); atts.append(a)
        logits = self.lnf(x) @ self.wte.weight.T
        return (logits, rows, atts) if keep else logits


# ---------------------------------------------------------------------------------------------- glyphs
# System fonts, first one that really has the glyph (a missing glyph renders as the same "tofu" box
# as a private-use code point). Only small alpha bitmaps of the glyphs ship, not the font files.
FONTS = [("/System/Library/Fonts/Hiragino Sans GB.ttc", 1), ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
         ("/System/Library/Fonts/Apple Symbols.ttf", 0), ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 0)]
ASCII_FONT = ("/System/Library/Fonts/Helvetica.ttc", 1)
NEWLINE_GLYPH = "↵"
GROW = "↵▶"
CELL = 48


def _bitmap(font, c, size=64):
    from PIL import Image, ImageDraw
    im = Image.new("L", (size, size), 0); ImageDraw.Draw(im).text((4, 4), c, fill=255, font=font)
    return im.tobytes()


def font_for(c, cache={}):
    from PIL import ImageFont
    if not cache:
        for p, i in FONTS:
            if os.path.isfile(p):
                f = ImageFont.truetype(p, 40, index=i)
                cache[p] = (f, _bitmap(f, "\ue000"))
        if os.path.isfile(ASCII_FONT[0]):
            cache["ascii"] = (ImageFont.truetype(ASCII_FONT[0], 38, index=ASCII_FONT[1]), None)
    if ord(c) < 128 and "ascii" in cache:
        return cache["ascii"][0]
    for k, (f, tofu) in cache.items():
        if tofu is not None and _bitmap(f, c) != tofu:
            return f
    raise SystemExit(f"no font has a glyph for {c!r}")


def write_glyphs(chars, path):
    """GLY1 | int32 cell | int32 n | int32 codepoint[n] | int32 zlen | zlib(alpha[n][cell][cell]) — row 0 = top."""
    from PIL import Image, ImageDraw
    out = bytearray(); cps = []
    for c in chars:
        im = Image.new("L", (CELL, CELL), 0); dr = ImageDraw.Draw(im)
        f = font_for(c)
        bb = dr.textbbox((0, 0), c, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if ord(c) < 128:      # ASCII: shared baseline so digit strings line up
            asc = dr.textbbox((0, 0), "0", font=f)
            x = (CELL - w) / 2 - bb[0]; y = (CELL - (asc[3] - asc[1])) / 2 - asc[1]
        else:
            x = (CELL - w) / 2 - bb[0]; y = (CELL - h) / 2 - bb[1]
        dr.text((x, y), c, fill=255, font=f)
        if c in GROW:         # symbol fonts draw these tiny: scale the ink up to ~30 px, centred
            ink = im.crop(im.getbbox()); k = 30 / max(ink.size)
            ink = ink.resize((max(1, round(ink.width * k)), max(1, round(ink.height * k))), Image.LANCZOS)
            im = Image.new("L", (CELL, CELL), 0); im.paste(ink, ((CELL - ink.width) // 2, (CELL - ink.height) // 2))
        out += im.tobytes(); cps.append(ord(c))
    z = zlib.compress(bytes(out), 9)
    with open(path, "wb") as fo:
        fo.write(b"GLY1"); fo.write(struct.pack("<ii", CELL, len(cps)))
        fo.write(struct.pack(f"<{len(cps)}i", *cps)); fo.write(struct.pack("<i", len(z))); fo.write(z)


# ---------------------------------------------------------------------------------------------- export
def f32(a):
    return np.asarray(a, dtype="<f4").tobytes()


def write_model(path, model, vocab, pca, meta):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    vb = vocab.encode("utf-8")
    with open(path, "wb") as fo:
        fo.write(b"GPT1")
        fo.write(struct.pack("<9i", 1, len(vocab), CTX, D, N_LAYER, N_HEAD, DFF, N_PCA, len(vb)))
        fo.write(vb)
        fo.write(f32(sd["wte.weight"])); fo.write(f32(sd["wpe.weight"]))
        for l in range(N_LAYER):
            p = f"blocks.{l}."
            for n in ["ln1.weight", "ln1.bias", "qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                      "ln2.weight", "ln2.bias", "fc.weight", "fc.bias", "fc2.weight", "fc2.bias"]:
                fo.write(f32(sd[p + n]))       # Linear weights row-major [out][in]
        fo.write(f32(sd["lnf.weight"])); fo.write(f32(sd["lnf.bias"]))
        for mean, comps, std in pca:               # rows 0..N_LAYER: mean[D], comps[N_PCA][D], std[N_PCA]
            fo.write(f32(mean)); fo.write(f32(comps)); fo.write(f32(std))
        fo.write(struct.pack("<ii", meta["train_chars"], meta["val_chars"]))
        fo.write(struct.pack("<ff", meta["val_ppl"], meta["unigram_ppl"]))


def block_nll(model, ids):
    """Sum of -log p over ids scored in consecutive blocks of CTX (each token sees its in-block prefix)."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(ids) - 1, CTX):
            x = ids[b:b + CTX]; y = ids[b + 1:b + 1 + CTX]
            x = x[:len(y)]
            if len(x) == 0:
                break
            lg = model(torch.tensor([x]))[0].double()
            lp = lg.log_softmax(-1)
            tot -= lp[torch.arange(len(y)), torch.tensor(y)].sum().item(); n += len(y)
    return tot, n


def main():
    t0 = time.time()
    random.seed(0); np.random.seed(0); torch.manual_seed(20261027)
    torch.use_deterministic_algorithms(True)
    train = Grammar(1234).text(TRAIN_CHARS)
    val = Grammar(98765).text(VAL_CHARS)
    vocab = "".join(sorted(set(train) | set(val)))
    stoi = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    print(f"corpus: train {len(train)} chars, val {len(val)} chars, vocab {V}")
    for line in train.split("\n")[:4]:
        print("  sample:", line)

    tr = torch.tensor([stoi[c] for c in train]); va = [stoi[c] for c in val]
    model = GPT(V)
    n_params = sum(p.numel() for n, p in model.named_parameters())
    print(f"model: {N_LAYER} layers x {N_HEAD} heads, d={D}, ctx={CTX}, params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.98))
    g = torch.Generator().manual_seed(7)
    for step in range(STEPS + 1):
        lr = LR * min(1.0, (step + 1) / 100) * (0.5 * (1 + math.cos(math.pi * step / STEPS)) * 0.95 + 0.05)
        for pg in opt.param_groups:
            pg["lr"] = lr
        off = torch.randint(0, len(tr) - CTX - 1, (BATCH,), generator=g)
        x = torch.stack([tr[o:o + CTX] for o in off]); y = torch.stack([tr[o + 1:o + 1 + CTX] for o in off])
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            model.eval(); s, n = block_nll(model, va[:4000]); model.train()
            print(f"  step {step:5d}  train loss {loss.item():.3f}  val loss {s / n:.3f}  ({time.time() - t0:.0f} s)")
    model.eval()

    # round the weights to float32 (what ships) and evaluate THAT model
    s, n = block_nll(model, va)
    val_loss = s / n; val_ppl = math.exp(val_loss)
    counts = np.zeros(V, dtype=np.int64)
    for c in train:
        counts[stoi[c]] += 1
    pu = (counts + 1) / (counts.sum() + V)                       # add-one smoothed unigram from TRAIN
    uni_nll = 0.0; un = 0
    for b in range(0, len(va) - 1, CTX):                         # same targets as block_nll
        for t in va[b + 1:b + 1 + CTX]:
            uni_nll -= math.log(pu[t]); un += 1
    uni_ppl = math.exp(uni_nll / un)
    print(f"VALIDATION ({n} tokens): loss {val_loss:.4f} nats/char, perplexity {val_ppl:.3f}; "
          f"unigram baseline perplexity {uni_ppl:.2f}")

    # PCA of each residual-stream row over training windows (for the 3-D columns: 64 dims -> 6 cells)
    with torch.no_grad():
        offs = torch.arange(0, 600) * ((len(tr) - CTX - 1) // 600)
        xb = torch.stack([tr[o:o + CTX] for o in offs])
        _, rows, _ = model(xb, keep=True)
    pca = []
    for r in rows:
        X = r.reshape(-1, D).double().numpy()
        mean = X.mean(0); C = np.cov((X - mean).T)
        ev, evec = np.linalg.eigh(C)
        order = np.argsort(ev)[::-1][:N_PCA]
        comps = evec[:, order].T.copy()
        for k in range(N_PCA):                                  # deterministic sign
            if comps[k][np.argmax(np.abs(comps[k]))] < 0:
                comps[k] = -comps[k]
        pca.append((mean, comps, np.sqrt(ev[order])))
        print(f"  PCA row {len(pca) - 1}: top-{N_PCA} explain {ev[order].sum() / ev.sum() * 100:.0f}% of variance")

    os.makedirs(ASSETS, exist_ok=True); os.makedirs(TESTRES, exist_ok=True)
    write_model(os.path.join(ASSETS, "model.bin"), model, vocab, pca,
                dict(train_chars=len(train), val_chars=len(val), val_ppl=val_ppl, unigram_ppl=uni_ppl))

    # glyph atlas: vocab chars (newline drawn as a return arrow) + printable ASCII + a few symbols
    extra = [chr(c) for c in range(33, 127)] + ["…", "→", "←", "▶", "·", "%"]
    chars = []
    for c in list(vocab) + extra:
        c = NEWLINE_GLYPH if c == "\n" else c
        if c not in chars:
            chars.append(c)
    write_glyphs(chars, os.path.join(ASSETS, "glyphs.bin"))
    print(f"glyphs: {len(chars)} ({CELL}x{CELL} px); newline drawn as {NEWLINE_GLYPH!r}")

    # PyTorch reference (float64 forward of the float32 weights that ship)
    m64 = GPT(V).double(); m64.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in model.state_dict().items()})
    m64.eval()
    prompts = ["\n狗问猫：「你有", "\n猫把书给了狗。", "\n猫把书给了狗。狗谢谢", "\n今天是星期三，明天是星期", "\n数数：五六七",
               "\n狗说：「我喜欢花。", "\n猫有三样东西：书、苹果、笔。猫最喜欢", "\n羊有两样东西：花、球。羊最喜欢",
               "\n", val[100:100 + CTX]]
    with open(os.path.join(TESTRES, "ref.bin"), "wb") as fo:
        fo.write(b"GREF"); fo.write(struct.pack("<i", len(prompts)))
        for p in prompts:
            ids = [stoi[c] for c in p][-CTX:]
            with torch.no_grad():
                lg, _, atts = m64(torch.tensor([ids]), keep=True)
            fo.write(struct.pack("<i", len(ids))); fo.write(struct.pack(f"<{len(ids)}i", *ids))
            fo.write(f32(lg[0].numpy())); fo.write(f32(torch.stack([a[0] for a in atts]).numpy()))
            top = lg[0, -1].softmax(-1).topk(3)
            print(f"  ref {p.strip()[-12:]!s:>14} -> " + "  ".join(
                f"{vocab[i]!r} {v * 100:.1f}%" for v, i in zip(top.values.tolist(), top.indices.tolist())))
    with open(os.path.join(TESTRES, "val.txt"), "w", encoding="utf-8") as fo:
        fo.write(val)
    with open(os.path.join(TESTRES, "train_counts.bin"), "wb") as fo:
        fo.write(struct.pack(f"<{V}i", *counts.tolist()))

    # greedy samples (sanity)
    for p in ["\n猫把书给了狗。", "\n今天是星期五，", "\n兔有三样东西："]:
        ids = [stoi[c] for c in p]
        for _ in range(16):
            with torch.no_grad():
                nxt = int(model(torch.tensor([ids[-CTX:]]))[0, -1].argmax())
            ids.append(nxt)
            if vocab[nxt] == "\n":
                break
        print("  greedy:", "".join(vocab[i] for i in ids).strip())
    for f in ["model.bin", "glyphs.bin"]:
        print(f"wrote {f}: {os.path.getsize(os.path.join(ASSETS, f))} bytes")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
