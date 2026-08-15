import math
import random

from matrix import matrix
from typedefs import Layer


def he_kernel(k=3, c_in=1):
    std = (2.0 / (k * k * c_in)) ** 0.5      # 0.4714 for 3x3x1
    return matrix([[random.gauss(0.0, std) for _ in range(k)] for _ in range(k)])

def flatten_3d(input: list[matrix]):
    r = []
    for m in range(len(input)):
        mat = input[m]
        r+=mat.flatten()
    return r

def he_dense(n_out, n_in):
    std = (2.0 / n_in) ** 0.5
    return [[random.gauss(0.0, std) for _ in range(n_in)] for _ in range(n_out)]

def dense(x, W, b):
    return [sum(w * xi for w, xi in zip(row, x)) + bj for row, bj in zip(W, b)]

def relu_vec(x):
    return [max(0.0, v) for v in x]

def softmax(z):
    hi = max(z)
    e = [math.exp(v - hi) for v in z]
    t = sum(e)
    return [v / t for v in e]

def feature_extraction(layers: list[Layer], curr: list[matrix]) -> list[matrix]:
    if len(layers) == 0: return curr

    filters, stride = layers[0]
    out = []
    for stack in filters:
        acc = None
        for ch_map, kern in zip(curr, stack):
            m = ch_map.conv2D(kern, stride)
            acc = m if acc is None else acc.add(m)
        assert acc is not None
        out.append(acc.reLu())
    return feature_extraction(layers[1:], out)

def classification(input: list[matrix], w1,b1, w2,b2) -> list[float]:
    flattened = flatten_3d(input)
    h = relu_vec(dense(flattened, w1, b1))
    probs = softmax(dense(h, w2, b2))
    return probs