import math
import random

from matrix import matrix
from typedefs import ConvCache, Layer


def he_kernel(k=3, c_in=1):
    std = (2.0 / (k * k * c_in)) ** 0.5      # 0.4714 for 3x3x1
    return matrix([[random.gauss(0.0, std) for _ in range(k)] for _ in range(k)])

def flatten_3d(input: list[matrix]):
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
    return [sum(w * xi for w, xi in zip(row, x)) + bj for row, bj in zip(W, b)] # type: ignore

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
            acc = acc.maxPool(2)
            if dump is not None:
                acc.saveAs("out/"+dump+"/L"+str(i + 1)+"/"+str(j))
            pre_act.append(acc)

        caches.append((curr, pre_act))
        curr = [z.reLu() for z in pre_act]

    return curr, caches

def classification(features: list[matrix]) -> list[float]:
    size = features[0].size()
    FLAT = len(features) * size[0] * size[1]
    W1, b1 = he_dense(128, FLAT), [0.0] * 128
    W2, b2 = he_dense(10, 128),   [0.0] * 10
    flattened_features = flatten_3d(features) 
    h = relu_vec(dense(flattened_features, W1, b1))
    probs = softmax(dense(h, W2, b2))
    return probs