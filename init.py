"""
IDEA: use MNIST dataset for digit classification

Approach:
    FEATURE EXTRACTION
        - Take in the input file
        - Convolution 32 filters, 3x3, stride 1   26x26
        - ReLU
        - maxPool  -> 26x26 -> 13x13
        - Convolution 64 filters, 3x3, stride 2   13x13
        - ReLU
        - maxPool  -> 13x13 -> 5x5

    CLASSIFICATION
        - Flatten                                 64x5x5 -> 640
        - Dense 128 + ReLU
        - Dense 10 + softmax

    TRAINING
        - Not implemented. feature_extraction now returns the per-layer caches
          the backward pass will need; weights are still never updated.

"""
import math
import random

from matrix import matrix
from mnist_loader import load_mnist
from typedefs import Layer
from utils import (
    classification,
    cross_entropy,
    feature_extraction,
    he_kernel,
    predict,
)

random.seed(0)

conv1_filters = [[he_kernel(3, 1)] for _ in range(32)]
conv2_filters = [[he_kernel(3, 32) for _ in range(32)] for _ in range(64)]

if __name__ == "__main__":

    imgs, lbls = load_mnist("train", limit=5)

    layers: list[Layer] = [
        (conv1_filters,1),
        (conv2_filters,1),
    ]

    total_loss = 0.0

    for n, (img, label) in enumerate(zip(imgs, lbls)):
        input = matrix([[p for p in row] for row in img])
        features, caches = feature_extraction(layers, [input], dump=str(n)) # FEATURE EXTRACTION
        probs = classification(features) # CLASSIFICATION

        loss = cross_entropy(probs, label)
        total_loss += loss

        rows, cols = features[0].size()
        print(f"maps={len(features)}  {rows}x{cols}   flat={len(features)*rows*cols}   "
              f"label={label}  pred={predict(probs)}  loss={loss:.3f}")

    print(f"\navg loss {total_loss / len(imgs):.4f}      "
          f"target ln(10) = {math.log(10):.4f}")

        