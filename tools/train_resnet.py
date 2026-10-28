#!/usr/bin/env python3
"""train_resnet.py — the real experiment behind the ResNet demo (残差网络 · 跳连).

Plain vs residual MLPs of depth D in {4, 8, 16, 32, 56} (D = number of weight layers, as in "ResNet-56"),
width 32, ReLU, NO normalisation, trained on the same synthetic two-spirals task (2000 points, 2 turns,
seeded) with the same optimiser (SGD, momentum 0.9, lr 0.01, batch 100, 60 epochs).

  D layers = stem Linear(2,32) + (D-2)/2 blocks of [Linear(32,32), ReLU, Linear(32,32)] + head Linear(32,2)
  plain block:     h <- relu(F(h))
  residual block:  h <- relu(h + F(h))           (He, Zhang, Ren & Sun 2015, "basic block", MLP version)
  init: Kaiming-normal (fan_in, ReLU), zero biases; in the residual net the second layer of every block is
  additionally scaled by 1/sqrt(#blocks) so that the forward signal does not grow with depth (no BatchNorm
  here; cf. Fixup / SkipInit). Everything else is identical.

Recorded (epoch 0 = at initialisation, then after each epoch), in full-batch eval on the training set:
  training cross-entropy loss, training accuracy, and for every weight layer l the Frobenius norm of the
  full-batch gradient ||dL/dW_l||.

Outputs (app/src/main/assets/resnet/), all little-endian:
  logs.bin     magic 'RNL1', int32 E (= epochs + 1), int32 K (= number of depths), int32 depths[K],
               then for kind in (plain, residual), for each depth D:
               float32 loss[E], float32 acc[E], float32 gnorm[E][D]   (layer 0 = stem ... D-1 = head)
  weights.bin  magic 'RNW1', int32 K, int32 depths[K], int32 width, then for kind, for depth, for layer:
               float32 W[out][in] (row-major, PyTorch layout), float32 b[out]
  tests.bin    magic 'RNT1', int32 N, float32 x[N][2], then for kind, for depth: float32 logits[N][2]
Prints the metrics and the SHA-256 of the three files (determinism check: run twice, same hashes).
"""
import hashlib, math, os, struct, sys, time
import torch

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)

DEPTHS = [4, 8, 16, 32, 56]
WIDTH = 32
EPOCHS = 60
N = 2000
LR = 0.01
BATCH = 100
SEED = 1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "src", "main", "assets", "resnet")


def spirals(n, turns=2.0, noise=0.03, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * turns * 2 * math.pi + 0.5
    r = t / (turns * 2 * math.pi + 0.5)
    lab = torch.randint(0, 2, (n,), generator=g)
    a = t + math.pi * lab
    x = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1) + noise * torch.randn(n, 2, generator=g)
    return x, lab


class Net(torch.nn.Module):
    def __init__(self, depth, res):
        super().__init__()
        self.res = res
        nb = (depth - 2) // 2
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, WIDTH)] +
                                          [torch.nn.Linear(WIDTH, WIDTH) for _ in range(2 * nb)] +
                                          [torch.nn.Linear(WIDTH, 2)])
        for m in self.layers[:-1]:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            torch.nn.init.zeros_(m.bias)
        if res:
            for b in range(nb):
                self.layers[2 + 2 * b].weight.data *= 1.0 / math.sqrt(nb)

    def forward(self, x):
        L = self.layers
        h = torch.relu(L[0](x))
        nb = (len(L) - 2) // 2
        for b in range(nb):
            f = L[2 + 2 * b](torch.relu(L[1 + 2 * b](h)))
            h = torch.relu(h + f) if self.res else torch.relu(f)
        return L[-1](h)


def measure(net, x, y):
    net.zero_grad()
    out = net(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    acc = (out.argmax(1) == y).float().mean().item()
    g = [m.weight.grad.norm().item() for m in net.layers]
    net.zero_grad()
    return loss.item(), acc, g


def train(depth, res, x, y):
    torch.manual_seed(SEED * 1000 + depth)          # the same init stream for plain and residual of a depth
    net = Net(depth, res)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    gen = torch.Generator().manual_seed(SEED)
    log = [measure(net, x, y)]
    for _ in range(EPOCHS):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), BATCH):
            b = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(x[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        log.append(measure(net, x, y))
    return net, log


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    x, y = spirals(N)
    nets, logs = {}, {}
    for res in (0, 1):
        for d in DEPTHS:
            net, log = train(d, res, x, y)
            nets[(res, d)] = net; logs[(res, d)] = log
            gl = log[-1][2]
            print(f"{'residual' if res else 'plain   '} D={d:2d}  final loss {log[-1][0]:.4f}  acc {log[-1][1]:.4f}  "
                  f"|g| layer0 @ep0 {log[0][2][0]:.3e} @end {gl[0]:.3e}  head @end {gl[-1]:.3e}", flush=True)
    E = EPOCHS + 1
    with open(os.path.join(OUT, "logs.bin"), "wb") as f:
        f.write(b"RNL1"); f.write(struct.pack("<ii", E, len(DEPTHS))); f.write(struct.pack(f"<{len(DEPTHS)}i", *DEPTHS))
        for res in (0, 1):
            for d in DEPTHS:
                lg = logs[(res, d)]
                f.write(struct.pack(f"<{E}f", *[e[0] for e in lg]))
                f.write(struct.pack(f"<{E}f", *[e[1] for e in lg]))
                for e in lg: f.write(struct.pack(f"<{d}f", *e[2]))
    with open(os.path.join(OUT, "weights.bin"), "wb") as f:
        f.write(b"RNW1"); f.write(struct.pack("<i", len(DEPTHS))); f.write(struct.pack(f"<{len(DEPTHS)}i", *DEPTHS)); f.write(struct.pack("<i", WIDTH))
        for res in (0, 1):
            for d in DEPTHS:
                for m in nets[(res, d)].layers:
                    w = m.weight.detach().contiguous().view(-1).tolist(); b = m.bias.detach().tolist()
                    f.write(struct.pack(f"<{len(w)}f", *w)); f.write(struct.pack(f"<{len(b)}f", *b))
    tx = torch.tensor([[0.0, 0.0], [0.5, -0.25], [-0.7, 0.3], [0.2, 0.9], [-0.33, -0.66], [0.9, 0.9]])
    with open(os.path.join(OUT, "tests.bin"), "wb") as f:
        f.write(b"RNT1"); f.write(struct.pack("<i", len(tx))); f.write(struct.pack(f"<{tx.numel()}f", *tx.view(-1).tolist()))
        with torch.no_grad():
            for res in (0, 1):
                for d in DEPTHS:
                    o = nets[(res, d)](tx).view(-1).tolist(); f.write(struct.pack(f"<{len(o)}f", *o))
    for n in ("logs.bin", "weights.bin", "tests.bin"):
        p = os.path.join(OUT, n)
        print(f"{n:12s} {os.path.getsize(p):8d} B  sha256 {sha(p)}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
