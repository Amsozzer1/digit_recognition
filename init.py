"""
IDEA: use MNIST dataset for digit classification

Approach:
    FEATURE EXTRACTION
        - Take in the input file
        - Convolution 32 filters, 3x3, stride 1   28x28 -> 26x26
        - ReLU
        - maxPool  -> 26x26 -> 13x13
        - Convolution 64 filters, 3x3, stride 1   13x13 -> 11x11
        - ReLU
        - maxPool  -> 11x11 -> 5x5

    CLASSIFICATION
        - Flatten                                 64x5x5 -> 1600
        - Dense 64 + ReLU
        - Dense 10 + softmax

    TRAINING
        - Dense head only: per-sample SGD on W1/b1/W2/b2. The conv filters are
          still frozen at their He init - feature_extraction returns the caches
          the conv backward pass will need, but nothing consumes them yet.

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
    save_checkpoint,
    sgd_update,
)

random.seed(0)

conv1_filters = [[he_kernel(3, 1)] for _ in range(32)]                      # 32 filters, 1 channel each
conv2_filters = [[he_kernel(3, 32) for _ in range(32)] for _ in range(64)]  # 64 filters, 32 channels each
W1, b1 = he_dense(64, 1600), [0.0] * 64
W2, b2 = he_dense(10, 64),   [0.0] * 10

train_size = 6400
batch_size = 64
epochs = 4
layers: list[Layer] = [
        (conv1_filters,1),
        (conv2_filters,1),
    ]

imgs, lbls = load_mnist("train", limit=train_size)   # parsed once, not per batch

def run_batch(idx: int, size: int = batch_size) -> tuple[float, int]:
    """Train on images [idx*size, idx*size+size). Returns (total loss, correct)."""
    total, correct = 0.0, 0
    for n in range(idx * size, idx * size + size):
        label = lbls[n]
        input = matrix([[p for p in row] for row in imgs[n]])
        features, _ = feature_extraction(layers, [input]) # dump=str(n) stores images per layer # FEATURE EXTRACTION
        probs, probs_caches = classification(features,W1,b1,W2,b2) # CLASSIFICATION
        total += cross_entropy(probs, label)
        correct += (predict(probs) == label)

        # Backprop L2
        onehot1 = onehot(int(label), len(b2))
        dz = [probs[i] - onehot1[i] for i in range(len(probs))]
        dW2, db2, dh = dense_backward(dz, probs_caches[1], W2)

        # Backprop L1
        dz2 = [dh[i] * (1.0 if probs_caches[1][i] > 0.0 else 0.0) for i in range(len(dh))]
        dW1, db1, _ = dense_backward(dz2, probs_caches[0], W1)

        sgd_update(W2, b2, dW2, db2)
        sgd_update(W1, b1, dW1, db1)
    return total, correct


if __name__ == "__main__":
    for epoch in range(epochs):
        total_loss, total_correct = 0.0, 0
        for i in range(train_size // batch_size):
            loss, correct = run_batch(i)
            total_loss += loss
            total_correct += correct
        print(f"epoch {epoch}  avg loss {total_loss / train_size:.4f}  "
              f"acc {total_correct}/{train_size}      target ln(10) = {math.log(10):.4f}")
        save_checkpoint(f"weights/epoch{epoch}.json", epoch, [W1, W2], [b1, b2])

    W1.saveAs("W1")
    W2.saveAs("W2")
