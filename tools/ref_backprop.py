"""PyTorch reference for core/backprop (BackpropReferenceTest).

No training data, no weights to ship: the demo trains live in Kotlin. This script only pins the maths:
  1. a 2-4-4-1 net (tanh, tanh, sigmoid; BCE loss) with weights given by a closed formula, one sample;
     autograd gradients of every parameter and the parameters after ONE SGD step (lr 0.1);
  2. Mazur's textbook 2-2-2 example (sigmoid, sigmoid, 1/2 sum of squares) after one step (lr 0.5);
  3. the hand-computed 2-2-1 tanh/BCE step from the test comment.
It prints Kotlin-ready arrays that BackpropReferenceTest hard-codes (matched there to 1e-9; float64 here).
Run: python3 tools/ref_backprop.py
"""
import math
import torch

torch.set_default_dtype(torch.float64)


def formula_net(sizes):
    """W[l][j, i] = 0.6 sin(1.3 (j*n_in + i) + 0.7 l + 0.1), b[l][j] = 0.1 cos(j + 2 l) — same formula in the Kotlin test."""
    Ws, bs = [], []
    for l in range(len(sizes) - 1):
        n_in, n_out = sizes[l], sizes[l + 1]
        W = torch.tensor([[0.6 * math.sin(1.3 * (j * n_in + i) + 0.7 * l + 0.1) for i in range(n_in)] for j in range(n_out)], requires_grad=True)
        b = torch.tensor([0.1 * math.cos(j + 2 * l) for j in range(n_out)], requires_grad=True)
        Ws.append(W); bs.append(b)
    return Ws, bs


def forward(Ws, bs, x, hidden, out):
    a = x
    for l, (W, b) in enumerate(zip(Ws, bs)):
        z = W @ a + b
        a = hidden(z) if l < len(Ws) - 1 else out(z)
    return a


def kt(name, t):
    print(f"    val {name} = doubleArrayOf(" + ", ".join(f"{v:.15e}" for v in t.detach().reshape(-1).tolist()) + ")")


def case_formula():
    Ws, bs = formula_net([2, 4, 4, 1])
    x = torch.tensor([0.3, -0.7]); y = torch.tensor([1.0]); lr = 0.1
    p = forward(Ws, bs, x, torch.tanh, torch.sigmoid)
    loss = torch.nn.functional.binary_cross_entropy(p, y)
    loss.backward()
    print(f"    // case 1: 2-4-4-1 tanh/tanh/sigmoid BCE, x=(0.3,-0.7), y=1, lr=0.1: p={p.item():.15f} loss={loss.item():.15f}")
    print(f"    val P1 = {p.item():.15e}; val LOSS1 = {loss.item():.15e}")
    for l, (W, b) in enumerate(zip(Ws, bs)):
        kt(f"GW1_{l}", W.grad); kt(f"GB1_{l}", b.grad)
        kt(f"NW1_{l}", W - lr * W.grad)


def case_mazur():
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    W1 = torch.tensor([[0.15, 0.20], [0.25, 0.30]], requires_grad=True); b1 = torch.tensor([0.35, 0.35], requires_grad=True)
    W2 = torch.tensor([[0.40, 0.45], [0.50, 0.55]], requires_grad=True); b2 = torch.tensor([0.60, 0.60], requires_grad=True)
    x = torch.tensor([0.05, 0.10]); t = torch.tensor([0.01, 0.99])
    o = forward([W1, W2], [b1, b2], x, torch.sigmoid, torch.sigmoid)
    E = 0.5 * ((t - o) ** 2).sum(); E.backward()
    print(f"    // case 2 (Mazur): E_total={E.item():.9f}  out={o.tolist()}")
    kt("MAZ_W1", W1 - 0.5 * W1.grad); kt("MAZ_W2", W2 - 0.5 * W2.grad)


def case_hand():
    W1 = torch.tensor([[0.5, -0.5], [0.25, 1.0]], requires_grad=True); b1 = torch.tensor([0.0, 0.0], requires_grad=True)
    W2 = torch.tensor([[1.0, -1.0]], requires_grad=True); b2 = torch.tensor([0.0], requires_grad=True)
    x = torch.tensor([1.0, 0.5]); y = torch.tensor([1.0])
    p = forward([W1, W2], [b1, b2], x, torch.tanh, torch.sigmoid)
    loss = torch.nn.functional.binary_cross_entropy(p, y); loss.backward()
    print(f"    // case 3 (hand): p={p.item():.12f} loss={loss.item():.12f}")
    kt("HAND_GW1", W1.grad); kt("HAND_GW2", W2.grad); kt("HAND_GB1", b1.grad)
    kt("HAND_NW1", W1 - 0.5 * W1.grad); kt("HAND_NW2", W2 - 0.5 * W2.grad)


if __name__ == "__main__":
    case_formula(); case_mazur(); case_hand()
