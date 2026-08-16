# Source - https://stackoverflow.com/a/20199213
# Posted by ubomb, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-16, License - CC BY-SA 4.0

import json

import numpy as np

from matrix import matrix
from mnist_loader import load_mnist
from typedefs import Layer
from utils import classification, feature_extraction

with open('weights/epoch3.json') as f:
    d = json.load(f)
    shapes: tuple[int,int] = d['shapes']
    weights: list[matrix] = [matrix(i) for i in d["weights"]]
    biases: list[list[float]] = d['biases']
    kernels: list[list[list[matrix]]] = [[[matrix(k) for k in stack] for stack in layer]
                                         for layer in d["kernels"]]
    correct = 0
    imgs,lbls = load_mnist("test", limit=1000)
    layers: list[Layer] = [
        (kernels[0], 1),
        (kernels[1], 1),
    ]
    for l in range(len(imgs)):
        img, lbl = imgs[l], lbls[l]
        input = matrix([[p for p in row] for row in img])
        features, _ = feature_extraction(layers, [input]) # dump=str(n) stores images per layer # FEATURE EXTRACTION
        probs, probs_caches = classification(features,weights[0],biases[0],weights[1],biases[1]) # CLASSIFICATION
        curr = - np.inf,0
        for i in range(len(probs)):
            if probs[i] > curr[0]:
                curr = probs[i], i

        prediction = curr[1]
        if prediction == lbl:
            correct+=1
        confidence = curr[0] * 100
        print(prediction, lbl, confidence)
    print(correct, len(imgs), f"{100 * correct / len(imgs):.1f}%")