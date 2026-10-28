#!/usr/bin/env python3
"""Train the mini word embeddings behind the "embedding" demo (词向量空间 · Word Embeddings, Mikolov et al. 2013).

CORPUS. Generated procedurally by the template grammar below (written for this demo; no external text, no
downloads). Every sentence talks about ONE content word and mixes in tokens that describe it:
  * people        "国王 他 是 男性 , 戴 王冠 住 宫殿"   role tokens (王冠/宫殿) shared by the pair, gender tokens
                  (他/男性 vs 她/女性) shared by the side — so 国王 ≈ royal + male, 王后 ≈ royal + female
  * places        "东京 是 首都 , 有 富士山"            pair token (富士山) shared by 日本 and 东京, type tokens
                  (首都/城市 vs 国家/国土) shared by every capital / every country
  * adjectives    "这个 速度 比 那个 更快"              pair token (速度) shared by 快/更快, form tokens (很/非常 vs 比/还要)
  * fillers       animals (农场 / 森林), fruit, colours  — category tokens only, so clusters form
The pair / role / side tokens are the hidden "context words": they are part of the vocabulary (the model
learns vectors for them too) but the demo only displays and searches the content words.
Nothing in the training objective mentions analogies: skip-gram only learns to predict a word's neighbours.

MODEL. Skip-gram with negative sampling (Mikolov et al. 2013): window 3, 5 negatives from unigram^0.75,
frequent-word subsampling t = 1e-3, D = 32, Adam lr 0.01 decayed linearly, 4 epochs. CPU, one thread,
fixed seeds -> bit-identical weights on every run (the SHA-256 of emb.bin is printed and pinned in a test).

VIEWS. 3-D = PCA of the unit word vectors, fitted per view to the relation-bearing words of that view
(gender pairs / country-capital pairs / adjective pairs); the "all" view is fitted to every shown word so the
filler clusters (animals, fruit, colours) separate too. Axis signs are fixed so that the
mean relation direction points to +x/+y/+z. Projection p = (v/|v| - mean) . W  (linear, so A-B+C projects
to exactly pA-pB+pC: the parallelogram drawn in 3-D is exact; the gap to D's dot is the real residual).

OUTPUTS (little-endian):
  app/src/main/assets/embedding/emb.bin     "EMB1" | int32 V, D, nRel, nView
      | V x { int32 nz, utf8 zh, int32 ne, utf8 en, int32 category, int32 shown }
      | float32 vec[V][D]                       (input vectors W of skip-gram, raw)
      | nRel x { int32 nPairs, nPairs x int32 (a, b) }      rel 0 gender (male,female), 1 (country,capital), 2 (base,comparative)
      | nView x { int32 n, ascii id, float32 mean[D], float32 W[3][D] }
  app/src/main/assets/embedding/words.bin   "WRD1" | int32 W, H, n | n x int32 (x, y, w, h) | int32 zlen | zlib(alpha[H][W])
      entry 2i = word i in Chinese, 2i+1 = in English (w = 0 for hidden words); row 0 = top.
      Fonts: Heiti SC Medium (STHeiti Medium.ttc, Chinese), Helvetica Bold (English) — macOS system fonts; only bitmaps ship.
  app/src/test/resources/embedding/ref.txt  PyTorch / numpy reference numbers for the JVM tests

    python3 tools/train_embedding.py
"""
import hashlib, os, random, struct, sys, time, zlib

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app/src/main/assets/embedding")
TESTRES = os.path.join(ROOT, "app/src/test/resources/embedding")

SEED = 2013
D, WINDOW, NEG, EPOCHS, BATCH, LR = 32, 3, 5, 4, 4096, 0.01
SENT_PER_WORD = 500
CROSS_TALK = 0.4      # share of sentences that also carry a describing token of some OTHER word (noise)
CO_MENTION = 0.2      # share of sentences that mention the pair partner too ("北京 是 中国 的 首都")

# ---------------------------------------------------------------------------------------------- vocabulary
# Filler words get a few attribute tokens each (farm/pet/forest, sweet/sour, warm/cool ...) so that the
# clusters have internal structure instead of collapsing to one point.
# categories: 0 person-male 1 person-female 2 country 3 capital 4 adjective 5 comparative 6 animal 7 fruit 8 colour
GENDER = [  # (male, female, en_m, en_f, role tokens)
    ("国王", "王后", "king", "queen", ["王冠", "王室"]),
    ("王子", "公主", "prince", "princess", ["城堡", "王室", "年轻"]),
    ("男人", "女人", "man", "woman", ["成年", "大人"]),
    ("男孩", "女孩", "boy", "girl", ["学校", "年轻"]),
    ("父亲", "母亲", "father", "mother", ["养育", "家庭"]),
    ("儿子", "女儿", "son", "daughter", ["出生", "家庭", "年轻"]),
    ("哥哥", "姐姐", "brother", "sister", ["兄妹", "家庭"]),
    ("爷爷", "奶奶", "grandpa", "grandma", ["拐杖", "家庭", "年迈"]),
    ("叔叔", "阿姨", "uncle", "aunt", ["亲戚", "家庭"]),
    ("丈夫", "妻子", "husband", "wife", ["婚礼", "家庭"]),
    ("男演员", "女演员", "actor", "actress", ["电影", "成年"]),
]
CONTINENT = {"中国": "亚洲", "日本": "亚洲", "韩国": "亚洲", "印度": "亚洲", "泰国": "亚洲", "埃及": "非洲",
             "美国": "美洲", "加拿大": "美洲"}   # the rest: 欧洲
CAPITAL = [  # (country, capital, en, en, pair token)
    ("中国", "北京", "China", "Beijing", "长城"),
    ("日本", "东京", "Japan", "Tokyo", "富士山"),
    ("法国", "巴黎", "France", "Paris", "铁塔"),
    ("英国", "伦敦", "UK", "London", "大本钟"),
    ("美国", "华盛顿", "USA", "Washington", "白宫"),
    ("俄罗斯", "莫斯科", "Russia", "Moscow", "红场"),
    ("德国", "柏林", "Germany", "Berlin", "啤酒节"),
    ("意大利", "罗马", "Italy", "Rome", "斗兽场"),
    ("埃及", "开罗", "Egypt", "Cairo", "金字塔"),
    ("韩国", "首尔", "Korea", "Seoul", "泡菜"),
    ("印度", "新德里", "India", "New Delhi", "恒河"),
    ("西班牙", "马德里", "Spain", "Madrid", "斗牛"),
    ("加拿大", "渥太华", "Canada", "Ottawa", "枫叶"),
    ("泰国", "曼谷", "Thailand", "Bangkok", "寺庙"),
]
COMPAR = [  # (base, comparative, en, en, pair token)
    ("大", "更大", "big", "bigger", "体积"),
    ("小", "更小", "small", "smaller", "颗粒"),
    ("快", "更快", "fast", "faster", "速度"),
    ("慢", "更慢", "slow", "slower", "步伐"),
    ("高", "更高", "tall", "taller", "高度"),
    ("冷", "更冷", "cold", "colder", "冬天"),
    ("热", "更热", "hot", "hotter", "夏天"),
    ("长", "更长", "long", "longer", "距离"),
]
FILLERS = [  # (category, [(zh, en)], category tokens, sub-group tokens)
    (6, [("猫", "cat"), ("狗", "dog"), ("马", "horse"), ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"), ("兔子", "rabbit"),
         ("老虎", "tiger"), ("狮子", "lion"), ("熊", "bear")], ["动物", "尾巴"], {"农场": [2, 3, 4, 5], "宠物": [0, 1, 6], "森林": [7, 8, 9],
         "吃草": [2, 3, 4, 6], "吃肉": [0, 7, 8, 9], "大型": [2, 3, 8, 9], "毛茸茸": [0, 4, 6, 9]}),
    (7, [("苹果", "apple"), ("香蕉", "banana"), ("橙子", "orange"), ("葡萄", "grape"), ("西瓜", "melon"), ("草莓", "strawberry"),
         ("桃子", "peach"), ("梨", "pear"), ("芒果", "mango"), ("柠檬", "lemon")], ["水果", "果汁"], {"甜": [1, 4, 5, 6, 8], "酸": [2, 9], "热带": [1, 8],
         "多汁": [2, 4, 6, 7], "圆": [0, 2, 3, 6, 7], "果园": [0, 6, 7], "浆果": [3, 5]}),
    (8, [("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("黄色", "yellow"), ("紫色", "purple"), ("白色", "white"),
         ("黑色", "black"), ("棕色", "brown"), ("粉色", "pink"), ("灰色", "grey")], ["颜色", "油漆"], {"鲜艳": [0, 3, 4, 8], "暗淡": [6, 7, 9], "暖色": [0, 3, 7, 8],
         "冷色": [1, 2, 4], "天空": [1, 5, 9], "花朵": [0, 4, 8], "树叶": [2, 3, 7]}),
]
MALE_T, FEMALE_T = ["他", "男性", "先生", "胡子"], ["她", "女性", "女士", "长裙"]
COUNTRY_T, CAPITAL_T = ["国家", "国土", "国旗", "边境"], ["首都", "城市", "市中心", "地铁"]
BASE_T, COMP_T = ["很", "非常", "有点", "相当"], ["比", "还要", "更加", "胜过"]
SIDE_N = 3             # side tokens (gender / country-vs-capital / base-vs-comparative) per sentence
FUNC = ["的", "是", "一个", "了", "在", "这个", "那个", "有", "也", "，"]


def build_vocab():
    words = []  # (zh, en, category, shown)
    for m, f, em, ef, _ in GENDER: words += [(m, em, 0, 1), (f, ef, 1, 1)]
    for c, k, ec, ek, _ in CAPITAL: words += [(c, ec, 2, 1), (k, ek, 3, 1)]
    for a, b, ea, eb, _ in COMPAR: words += [(a, ea, 4, 1), (b, eb, 5, 1)]
    for cat, items, _, _ in FILLERS: words += [(z, e, cat, 1) for z, e in items]
    hidden = []
    for *_, role in GENDER: hidden += role
    for *_, t in CAPITAL: hidden.append(t)
    hidden += ["亚洲", "欧洲", "非洲", "美洲"]
    for *_, t in COMPAR: hidden.append(t)
    for _, _, ct, sub in FILLERS: hidden += ct + list(sub.keys())
    hidden += MALE_T + FEMALE_T + COUNTRY_T + CAPITAL_T + BASE_T + COMP_T + FUNC
    seen = {w[0] for w in words}
    for h in hidden:
        if h not in seen: words.append((h, "", 9, 0)); seen.add(h)
    return words


# ---------------------------------------------------------------------------------------------- grammar
def sentences(r):
    """Yield token lists. Every sentence = one content word + its describing tokens + function words, shuffled
    within short random templates so no fixed word order carries information."""
    def sent(word, describing):
        toks = [word] + describing + r.sample(FUNC, r.randint(1, 3))
        body = toks[1:]; r.shuffle(body)
        k = r.randint(0, min(2, len(body)))
        return body[:k] + [word] + body[k:]
    jobs = []
    for m, f, _, _, role in GENDER:
        jobs += [(m, role, MALE_T), (f, role, FEMALE_T)]
    for c, k, _, _, t in CAPITAL:
        cont = CONTINENT.get(c, "欧洲")
        jobs += [(c, [t, t, cont], COUNTRY_T), (k, [t, t, cont], CAPITAL_T)]
    for a, b, _, _, t in COMPAR:
        jobs += [(a, [t], BASE_T), (b, [t], COMP_T)]
    for _, items, ct, sub in FILLERS:
        for i, (z, _) in enumerate(items):
            g = [s for s, idx in sub.items() if i in idx]
            jobs.append((z, g, ct))
    partner = {}
    for rel in (GENDER, CAPITAL, COMPAR):
        for p in rel: partner[p[0]] = p[1]; partner[p[1]] = p[0]
    for _ in range(SENT_PER_WORD):
        r.shuffle(jobs)
        for w, own, side in jobs:
            desc = [r.choice(own)] + [r.choice(side) for _ in range(SIDE_N)] + ([r.choice(own + side)] if r.random() < 0.5 else [])
            if r.random() < CROSS_TALK:
                _, o2, s2 = jobs[r.randrange(len(jobs))]; desc.append(r.choice(o2 + s2))
            if w in partner and r.random() < CO_MENTION: desc.append(partner[w])
            yield sent(w, desc)


# ---------------------------------------------------------------------------------------------- training
def train(words):
    idx = {w[0]: i for i, w in enumerate(words)}
    V = len(words)
    r = random.Random(SEED)
    corpus = [[idx[t] for t in s] for s in sentences(r)]
    counts = np.zeros(V, np.int64)
    for s in corpus:
        for t in s: counts[t] += 1
    ntok = int(counts.sum())
    # skip-gram pairs with frequent-word subsampling (Mikolov et al. 2013, eq. 5), dynamic window
    rng = np.random.RandomState(SEED)
    freq = counts / ntok
    keep_p = np.minimum(1.0, np.sqrt(1e-3 / np.maximum(freq, 1e-12)) + 1e-3 / np.maximum(freq, 1e-12))
    neg_p = counts ** 0.75; neg_p = neg_p / neg_p.sum()

    torch.manual_seed(SEED); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    Win = torch.nn.Parameter((torch.rand(V, D, dtype=torch.float32) - 0.5) / D)
    Wout = torch.nn.Parameter(torch.zeros(V, D, dtype=torch.float32))
    opt = torch.optim.Adam([Win, Wout], lr=LR)
    neg_cdf = torch.tensor(np.cumsum(neg_p), dtype=torch.float64)

    t0 = time.time(); step = 0
    for ep in range(EPOCHS):
        cs, ct = [], []
        for s in corpus:
            s = [t for t in s if rng.random_sample() < keep_p[t]]
            for i, w in enumerate(s):
                b = rng.randint(1, WINDOW + 1)
                for j in range(max(0, i - b), min(len(s), i + b + 1)):
                    if j != i: cs.append(w); ct.append(s[j])
        cs = np.array(cs, np.int64); ct = np.array(ct, np.int64)
        perm = rng.permutation(len(cs)); cs = cs[perm]; ct = ct[perm]
        nb = (len(cs) + BATCH - 1) // BATCH
        tot = 0.0
        for bi in range(nb):
            lr = LR * max(0.05, 1 - (ep * nb + bi) / (EPOCHS * nb))
            for g in opt.param_groups: g["lr"] = lr
            c = torch.from_numpy(cs[bi * BATCH:(bi + 1) * BATCH]); o = torch.from_numpy(ct[bi * BATCH:(bi + 1) * BATCH])
            u = torch.from_numpy(rng.random_sample((len(c), NEG)))
            negs = torch.searchsorted(neg_cdf, u).clamp(max=V - 1)
            vc = Win[c]
            pos = torch.nn.functional.logsigmoid((vc * Wout[o]).sum(-1))
            neg = torch.nn.functional.logsigmoid(-(Wout[negs] * vc[:, None, :]).sum(-1)).sum(-1)
            loss = -(pos + neg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); step += 1
        print(f"epoch {ep + 1}/{EPOCHS}: {len(cs)} pairs, mean loss {tot / nb:.4f}  ({time.time() - t0:.1f} s)")
    return Win.detach().numpy().astype(np.float32), ntok, len(corpus)


# ---------------------------------------------------------------------------------------------- evaluation
def unit(v):
    v = v.astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def analogy(U, cand, a, b, c, k=3):
    """3CosAdd (Mikolov et al. 2013): t = a - b + c on unit vectors; rank candidates by cos(t, x), A,B,C excluded."""
    t = U[a] - U[b] + U[c]
    t = t / np.linalg.norm(t)
    sc = [(float(U[i] @ t), i) for i in cand if i not in (a, b, c)]
    sc.sort(key=lambda x: (-x[0], x[1]))
    return [(i, s) for s, i in sc[:k]]


def relations(words):
    idx = {w[0]: i for i, w in enumerate(words)}
    return [[(idx[p[0]], idx[p[1]]) for p in rel] for rel in (GENDER, CAPITAL, COMPAR)]


def questions(rel):
    """Every ordered pair of pairs: b2 - a2 + a1 -> b1 (i.e. queen? = king - man + woman uses a1=king... see below)."""
    qs = []
    for i, (a1, b1) in enumerate(rel):
        for j, (a2, b2) in enumerate(rel):
            if i != j: qs.append((a1, a2, b2, b1))   # a1 - a2 + b2 -> b1 : king - man + woman -> queen
    return qs


def accuracy(U, cand, rels):
    out = []
    for rel in rels:
        qs = questions(rel)
        ok = sum(1 for a, b, c, d in qs if analogy(U, cand, a, b, c, 1)[0][0] == d)
        out.append((ok, len(qs)))
    return out


def pca_view(U, ids, rels_in_view):
    X = U[ids]
    mean = X.mean(0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    W = vt[:3].copy()
    dirn = np.mean([U[b] - U[a] for rel in rels_in_view for a, b in rel], axis=0)
    for k in range(3):
        if W[k] @ dirn < 0: W[k] = -W[k]
    return mean, W


# ---------------------------------------------------------------------------------------------- word atlas
ZH_FONT = ("/System/Library/Fonts/STHeiti Medium.ttc", 1)   # Heiti SC Medium
EN_FONT = ("/System/Library/Fonts/Helvetica.ttc", 1)
ATLAS_W, CELL_H = 1024, 44


def bake_words(words, path):
    from PIL import Image, ImageDraw, ImageFont
    fz = ImageFont.truetype(ZH_FONT[0], 34, index=ZH_FONT[1])
    fe = ImageFont.truetype(EN_FONT[0], 32, index=EN_FONT[1])
    strips = []
    for zh, en, cat, shown in words:
        for text, f in ((zh, fz), (en, fe)):
            if not shown: strips.append(None); continue
            tmp = ImageDraw.Draw(Image.new("L", (8, 8)))
            bb = tmp.textbbox((0, 0), text, font=f)
            asc = tmp.textbbox((0, 0), "Hg" if f is fe else "国", font=f)
            w = bb[2] - bb[0] + 12
            im = Image.new("L", (w, CELL_H), 0)
            ImageDraw.Draw(im).text((6 - bb[0], (CELL_H - (asc[3] - asc[1])) / 2 - asc[1]), text, fill=255, font=f)
            strips.append(im)
    rects = []; x = y = 0
    for s in strips:
        if s is None: rects.append((0, 0, 0, 0)); continue
        if x + s.width > ATLAS_W: x = 0; y += CELL_H
        rects.append((x, y, s.width, CELL_H)); x += s.width + 2
    H = y + CELL_H
    atlas = Image.new("L", (ATLAS_W, H), 0)
    for s, (x, y, w, h) in zip(strips, rects):
        if s is not None: atlas.paste(s, (x, y))
    z = zlib.compress(atlas.tobytes(), 9)
    with open(path, "wb") as fo:
        fo.write(b"WRD1"); fo.write(struct.pack("<iii", ATLAS_W, H, len(rects)))
        for rc in rects: fo.write(struct.pack("<4i", *rc))
        fo.write(struct.pack("<i", len(z))); fo.write(z)
    return ATLAS_W, H


# ---------------------------------------------------------------------------------------------- main
def main():
    words = build_vocab()
    V = len(words)
    shown = [i for i, w in enumerate(words) if w[3]]
    print(f"vocabulary {V} words ({len(shown)} shown content words, {V - len(shown)} hidden context words), D = {D}")
    vec, ntok, nsent = train(words)
    print(f"corpus: {nsent} generated sentences, {ntok} tokens")
    U = unit(vec)
    rels = relations(words)
    names = ["gender", "capital", "comparative"]
    acc = accuracy(U, shown, rels)
    acc_full = accuracy(U, list(range(V)), rels)
    for n, (ok, tot), (okf, totf) in zip(names, acc, acc_full):
        print(f"analogy {n:12s} top-1 {ok}/{tot} = {ok / tot:.1%}  (candidates = shown words)   full vocab {okf / totf:.1%}")
    ok = sum(a for a, _ in acc); tot = sum(t for _, t in acc)
    print(f"analogy ALL top-1 {ok}/{tot} = {ok / tot:.1%}")
    # shuffled-vector baseline (same vectors, words permuted)
    perm = np.random.RandomState(7).permutation(V)
    accs = accuracy(U[perm], shown, rels)
    print(f"shuffled-vector baseline: {sum(a for a, _ in accs)}/{tot} = {sum(a for a, _ in accs) / tot:.1%}")

    # views
    rel_ids = [sorted({i for p in rel for i in p}) for rel in rels]
    views = [("gender", rel_ids[0], [rels[0]]), ("capital", rel_ids[1], [rels[1]]), ("comparative", rel_ids[2], [rels[2]]),
             ("all", shown, rels)]
    vdata = []
    for name, ids, rv in views:
        mean, W = pca_view(U, ids, rv)
        X = U[ids] - mean; tot_var = (X ** 2).sum(); ex = ((X @ W.T) ** 2).sum() / tot_var
        vdata.append((name, mean, W))
        msg = ""
        if name != "all":   # how visible is the relation in 3-D: pair-difference length vs cloud radius, and parallelism
            P = (U - mean) @ W.T
            dif = np.array([P[b] - P[a] for a, b in rv[0]]); rad = np.sqrt(((P[ids]) ** 2).sum(1).mean())
            md = dif.mean(0); par = np.mean([d @ md / np.linalg.norm(d) / np.linalg.norm(md) for d in dif])
            msg = f"; pair arrows {np.linalg.norm(dif, axis=1).mean() / rad:.2f} x cloud radius, mean cos to their mean {par:.2f}"
            a0, b0 = rv[0][0]; msg += f"; cos(pair0) = {U[a0] @ U[b0]:.2f}"
        print(f"view {name:12s}: PCA on {len(ids)} words, 3 PCs explain {ex:.1%} of their variance{msg}")

    os.makedirs(ASSETS, exist_ok=True); os.makedirs(TESTRES, exist_ok=True)
    emb_path = os.path.join(ASSETS, "emb.bin")
    with open(emb_path, "wb") as fo:
        fo.write(b"EMB1"); fo.write(struct.pack("<4i", V, D, len(rels), len(vdata)))
        for zh, en, cat, sh in words:
            bz = zh.encode("utf-8"); be = en.encode("utf-8")
            fo.write(struct.pack("<i", len(bz))); fo.write(bz); fo.write(struct.pack("<i", len(be))); fo.write(be)
            fo.write(struct.pack("<ii", cat, sh))
        fo.write(vec.astype("<f4").tobytes())
        for rel in rels:
            fo.write(struct.pack("<i", len(rel)))
            for a, b in rel: fo.write(struct.pack("<ii", a, b))
        for name, mean, W in vdata:
            fo.write(struct.pack("<i", len(name))); fo.write(name.encode("ascii"))
            fo.write(mean.astype("<f4").tobytes()); fo.write(W.astype("<f4").tobytes())
    sha = hashlib.sha256(open(emb_path, "rb").read()).hexdigest()
    aw, ah = bake_words(words, os.path.join(ASSETS, "words.bin"))
    print(f"word atlas {aw}x{ah} px; emb.bin {os.path.getsize(emb_path)} B, words.bin {os.path.getsize(os.path.join(ASSETS, 'words.bin'))} B")
    print(f"emb.bin sha256 {sha}")

    # reference numbers for the JVM tests (computed from the float32 vectors as shipped, in float64)
    U32 = unit(vec.astype(np.float32))
    idx = {w[0]: i for i, w in enumerate(words)}
    lines = [f"sha256 {sha}", f"acc {ok} {tot}", f"shuffled {sum(a for a, _ in accs)} {tot}"]
    for n, (o, t) in zip(names, acc): lines.append(f"accrel {n} {o} {t}")
    # torch cosine for nearest neighbours (shown words, self excluded)
    Ut = torch.nn.functional.normalize(torch.from_numpy(vec.astype(np.float64)), dim=1)
    for w in ["国王", "北京", "猫", "更快", "红色", "女人"]:
        i = idx[w]
        cs = (Ut[shown] @ Ut[i]).numpy()
        order = sorted([(-cs[k], shown[k]) for k in range(len(shown)) if shown[k] != i])[:5]
        lines.append("nn %d " % i + " ".join("%d %.9f" % (j, -c) for c, j in order))
    for a, b, c in [("国王", "男人", "女人"), ("东京", "日本", "法国"), ("更大", "大", "小"), ("王子", "男孩", "女孩"), ("猫", "狗", "苹果")]:
        res = analogy(U32, shown, idx[a], idx[b], idx[c], 3)
        lines.append("an %d %d %d " % (idx[a], idx[b], idx[c]) + " ".join("%d %.9f" % (j, s) for j, s in res))
    for vi, (name, mean, W) in enumerate(vdata):
        m32 = mean.astype(np.float32).astype(np.float64); W32 = W.astype(np.float32).astype(np.float64)
        P = (U32 - m32) @ W32.T
        for i in shown: lines.append("proj %d %d %.9f %.9f %.9f" % (vi, i, *P[i]))
    with open(os.path.join(TESTRES, "ref.txt"), "w") as fo: fo.write("\n".join(lines) + "\n")
    for a, b, c in [("国王", "男人", "女人"), ("东京", "日本", "法国")]:
        res = analogy(U32, shown, idx[a], idx[b], idx[c], 3)
        print(f"  {a} - {b} + {c} -> " + ", ".join(f"{words[j][0]} {s:.3f}" for j, s in res))


if __name__ == "__main__":
    main()
