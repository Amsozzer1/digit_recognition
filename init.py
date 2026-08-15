"""
IDEA: use MNIST dataset for digit classification

Approach:
    FEATURE EXTRACTION
        - Take in the input file 
        - convolution 32 filters
        - ReLU
        - Max Pool
        - Convolution 64 filters
        - ReLU
        - Max Pool

    CLASSIFICATION
        - Flatten
        - Dense 128
        - Dense 10

    TRAINING

"""
import random

from matrix import matrix
from mnist_loader import load_mnist
from typedefs import Layer
from utils import classification, feature_extraction, he_dense, he_kernel

random.seed(0)

conv1_filters = [[he_kernel(3, 1)] for _ in range(32)]
conv2_filters = [[he_kernel(3, 32) for _ in range(32)] for _ in range(64)]

FLAT = 64 * 6 * 6
W1, b1 = he_dense(128, FLAT), [0.0] * 128
W2, b2 = he_dense(10, 128),   [0.0] * 10

if __name__ == "__main__":

    imgs, lbls = load_mnist("train", limit=5)

    layers: list[Layer] = [
        (conv1_filters,2),
        (conv2_filters,2)
    ]

    for img, label in zip(imgs, lbls):
        input = matrix([[p / 255.0 for p in row] for row in img])
        features = feature_extraction(layers, [input]) # FEATURE EXTRACTION
        dense10 = classification(features, W1, b1, W2, b2) # CLASSIFICATION

        