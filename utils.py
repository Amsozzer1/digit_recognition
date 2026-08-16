import json
import math
import random
from pathlib import Path

from consts import LR
from matrix import matrix
from typedefs import ConvCache, DenseCache, Layer


def he_kernel(k=3, c_in=1):
    std = (2.0 / (k * k * c_in)) ** 0.5      # 0.4714 for 3x3x1
    return matrix([[random.gauss(0.0, std) for _ in range(k)] for _ in range(k)])

def flatten_3d(input: list[matrix]) -> list[float]:
    r = []
    for m in range(len(input)):
        mat = input[m]
        r+=mat.flatten()
    return r

def he_dense(n_out, n_in) -> matrix:
    std = (2.0 / n_in) ** 0.5
    return matrix([[random.gauss(0.0, std) for _ in range(n_in)] for _ in range(n_out)])

def dense(x: list[float], W: matrix, b: list[float]) -> list[float]:
    # 𝑧 = 𝑤₁𝑥₁ + 𝑤₂𝑥₂ + 𝑤₃𝑥₃ + 𝑏
    X = matrix([[row] for row in x]) #[[1,2,3,4,...,n]] -> [ [1],[2],[3]....[n] ]
    B = matrix([[row] for row in b])
    return W.mult(X).add(B).flatten() # 1 * 64

def relu_vec(x):
    return [max(0.0, v) for v in x]

def softmax(z):
    hi = max(z)
    e = [math.exp(v - hi) for v in z]
    t = sum(e)
    return [v / t for v in e]

def cross_entropy(probs, label):
    """Negative log-likelihood of the true class. Clamped so a probability that
    underflows to 0.0 gives a large loss instead of a math domain error."""
    return -math.log(max(probs[label], 1e-12))
def onehot(label: int, n: int = 10) -> list[float]:
    return [1.0 if i==label else 0.0 for i in range(n)]
def predict(probs):
    return max(range(len(probs)), key=lambda i: probs[i])

def feature_extraction(
    layers: list[Layer], curr: list[matrix], dump: str | None = None
) -> tuple[list[matrix], list[ConvCache]]:
    """Run the conv stack, returning the final maps and one ConvCache per layer.

    Iterative rather than recursive so each layer's input survives the call —
    the backward pass needs them to compute conv gradients.

    `dump` names a subdirectory of out/ to write feature maps into, as
    out/<dump>/L<layer>/<filter>.png — one image per filter, written after the
    channels are accumulated and pooled so it matches what the next layer
    actually receives."""
    caches: list[ConvCache] = []

    for i, (filters, stride) in enumerate(layers):
        pre_act = []
        for j,stack in enumerate(filters):
            acc = None
            for ch_map, kern in zip(curr, stack):
                m = ch_map.conv2D(kern,stride)
                acc = m if acc is None else acc.add(m)

            assert acc is not None
            if dump is not None:
                acc.saveAs("out/"+dump+"/L"+str(i + 1)+"/"+str(j))
            pre_act.append(acc)

        caches.append((curr, pre_act))
        curr = [(z.reLu()).maxPool(2) for z in pre_act]

    return curr, caches
# [[]]
def classification(features: list[matrix],W1,b1,W2,b2) -> tuple[list[float], DenseCache]:
    flattened_features = flatten_3d(features) 
    l1 = dense(flattened_features, W1, b1) # z = wx + b
    h = relu_vec(l1)
    l2 = dense(h, W2, b2)
    probs = softmax(l2)
    caches: DenseCache = flattened_features, h
    return probs, caches

def dense_backward(
    dz: list[float],
    x:  list[float],
    W:  matrix,
) -> tuple[matrix, list[float], list[float]]:
    n_out, n_in = W.rows, W.cols
    dW = matrix([[dz[j] * x[i] for i in range(n_in)] for j in range(n_out)])
    db = list(dz)
    dx = [sum(W.values[j][i] * dz[j] for j in range(n_out)) for i in range(n_in)]
    return dW, db, dx

def sgd_update(W: matrix, b, dW: matrix, db, lr=LR):
    for j in range(W.rows):
        for i in range(W.cols):
            W.values[j][i] -= lr * dW.values[j][i]   # mutate in place
    for j in range(len(b)):
        b[j] -= lr * db[j]

def save_checkpoint(path: str, epoch: int,
                    weights: list[matrix], biases: list[list[float]], **meta) -> None:
    p = Path(path)
    if p.is_dir() or path.endswith(("/", "\\")):
        raise ValueError(f"path must be a file, got directory: {path!r}")
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch":   epoch,
        "shapes":  [[m.rows, m.cols] for m in weights],
        "weights": [m.values for m in weights],
        "biases":  biases,
        **meta,
    }

    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)


def load_checkpoint(path: str) -> tuple[int, list[matrix], list[list[float]], dict]:
    data = json.loads(Path(path).read_text())
    weights = [matrix(v) for v in data["weights"]]
    for i, (m, (r, c)) in enumerate(zip(weights, data["shapes"])):
        assert (m.rows, m.cols) == (r, c), \
            f"checkpoint weight {i}: got {m.rows}x{m.cols}, file says {r}x{c}"
    meta = {k: v for k, v in data.items()
            if k not in ("epoch", "shapes", "weights", "biases")}
    return data["epoch"], weights, data["biases"], meta 