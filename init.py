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
    dense_backward,
    feature_extraction,
    he_dense,
    he_kernel,
    onehot,
    predict,
    sgd_update,
)

random.seed(0)

conv1_filters = [[he_kernel(3, 1)] for _ in range(32)] # 16 3*3 matrices 
conv2_filters = [[he_kernel(3, 32) for _ in range(32)] for _ in range(64)] # 32 16 3*3 matrices 
W1, b1 = he_dense(64, 1600), [0.0] * 64
W2, b2 = he_dense(10, 64),   [0.0] * 10

if __name__ == "__main__":

    imgs, lbls = load_mnist("train", limit=15)

    layers: list[Layer] = [
        (conv1_filters,1),
        (conv2_filters,1),
    ]

    total_loss = 0.0

    for n, (img, label) in enumerate(zip(imgs, lbls)):
        input = matrix([[p for p in row] for row in img])
        features, feature_caches = feature_extraction(layers, [input]) # dump=str(n) stores images per layer # FEATURE EXTRACTION
        probs, probs_caches = classification(features,W1,b1,W2,b2) # CLASSIFICATION
        loss = cross_entropy(probs, label)
        total_loss += loss

        # Backprop L2 
        onehot1 = onehot(int(label), len(b2))
        dz = [probs[i] - onehot1[i] for i in range(min(len(probs), len(onehot1)))]
        dW2, db2, dh = dense_backward(dz, probs_caches[1], W2)

        # Backprop L1
        dz2 = [dh[i] * (1.0 if probs_caches[1][i] > 0.0 else 0.0) for i in range(len(dh))]
        dW1, db1, dflat = dense_backward(dz2, probs_caches[0], W1)

        sgd_update(W2, b2, dW2, db2)
        sgd_update(W1, b1, dW1, db1)

        rows, cols = features[0].size()
        print(f"maps={len(features)}  {rows}x{cols}   flat={len(features)*rows*cols}   "
              f"label={label}  pred={predict(probs)}  loss={loss:.3f}")

    print(f"\navg loss {total_loss / len(imgs):.4f}      "
          f"target ln(10) = {math.log(10):.4f}")
    W1.saveAs("W1")
    W2.saveAs("W2")

# input = matrix([[i*8 + j for j in range(1,9)] for i in range(8)])  
# pool2 = input.maxPool(2)

# input.print()
# pool2.print()
