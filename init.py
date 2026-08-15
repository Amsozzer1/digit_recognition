"""
IDEA: use MNIST dataset for digit classification

Approach:
    FEATURE EXTRACTION
        - Take in the input file
        - Convolution 32 filters, 3x3, stride 2   28x28 -> 13x13
        - ReLU
        - Convolution 64 filters, 3x3, stride 2   13x13 -> 6x6
        - ReLU

      Downsampling is the stride-2 convolutions, not pooling. matrix.maxPool
      works but is not wired into the stack.

    CLASSIFICATION
        - Flatten                                 64x6x6 -> 2304
        - Dense 128 + ReLU
        - Dense 10 + softmax

    TRAINING
        - Not implemented. feature_extraction now returns the per-layer caches
          the backward pass will need; weights are still never updated.

"""
import math
import random

import numpy as np

from matrix import matrix
from mnist_loader import load_mnist
from typedefs import Layer
from utils import (
    classification,
    cross_entropy,
    feature_extraction,
    he_dense,
    he_kernel,
    predict,
)

random.seed(0)

conv1_filters = [[he_kernel(3, 1)] for _ in range(32)]
conv2_filters = [[he_kernel(3, 32) for _ in range(32)] for _ in range(64)]


FLAT = 64 * 6 * 6
W1, b1 = he_dense(128, FLAT), [0.0] * 128
W2, b2 = he_dense(10, 128),   [0.0] * 10



if __name__ == "__main__":

    imgs, lbls = load_mnist("train", limit=5)

    # for i in range(len(imgs)):
    #     image = np.array(imgs[i])
    #     # image = (image * 255).astype(np.uint8)
    #     # img = Image.fromarray(image.astype(np.uint8), mode="L")
    #     # img.save("out/trial.png")
    #     x = matrix(imgs[i])
    #     saveImage(x, "out/x/trial")


    layers: list[Layer] = [
        (conv1_filters,2),
        (conv2_filters,2),
    ]

    total_loss = 0.0

    for img, label in zip(imgs, lbls):
        input = matrix([[p / 255.0 for p in row] for row in img])
        features, caches = feature_extraction(layers, [input]) # FEATURE EXTRACTION
    #     probs = classification(features, matrix(W1), b1, matrix(W2), b2) # CLASSIFICATION

    #     loss = cross_entropy(probs, label)
    #     total_loss += loss

    #     rows, cols = features[0].size()
    #     print(f"maps={len(features)}  {rows}x{cols}   flat={FLAT}   "
    #           f"label={label}  pred={predict(probs)}  loss={loss:.3f}")

    # print(f"\navg loss {total_loss / len(imgs):.4f}      "
    #       f"target ln(10) = {math.log(10):.4f}")

        