"""
Minimal MNIST loader - standard library only, no numpy, no frameworks.

Downloads the four IDX files once into ./mnist_data/ and parses them into
plain Python lists so they drop straight into your `matrix` class.

    from mnist_loader import load_mnist
    train_images, train_labels = load_mnist("train")
    test_images,  test_labels  = load_mnist("test")

    img = matrix(train_images[0])      # 28x28 nested list of floats, 0.0-1.0
    lbl = train_labels[0]              # int, 0-9
"""

import gzip
import os
import struct
import urllib.request
from array import array

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
DATA_DIR = "mnist_data"

FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test":  ("t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"),
}


def _download(filename):
    """Fetch one .gz into DATA_DIR if it isn't already there."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def _read_images(path, limit=None):
    """IDX3 format: magic(2051), count, rows, cols, then count*rows*cols bytes."""
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad magic number {magic} in {path}, expected 2051")
        if limit is not None:
            count = min(count, limit)
        buf = array("B", f.read(count * rows * cols))   # only read what we need

    images = []
    stride = rows * cols
    for n in range(count):
        flat = buf[n * stride:(n + 1) * stride]
        # reshape to rows x cols and scale 0-255 -> 0.0-1.0
        images.append([[flat[r * cols + c] / 255.0 for c in range(cols)]
                       for r in range(rows)])
    return images


def _read_labels(path):
    """IDX1 format: magic(2049), count, then count bytes."""
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad magic number {magic} in {path}, expected 2049")
        return list(array("B", f.read()))


def load_mnist(split="train", limit=None):
    """Return (images, labels). `limit` caps how many you parse - use it while
    developing, parsing all 60,000 into Python lists takes a while."""
    if split not in FILES:
        raise ValueError("split must be 'train' or 'test'")
    img_file, lbl_file = FILES[split]

    images = _read_images(_download(img_file), limit)
    labels = _read_labels(_download(lbl_file))[:len(images)]

    if len(images) != len(labels):
        raise ValueError(f"count mismatch: {len(images)} images, {len(labels)} labels")
    return images, labels


def one_hot(label, classes=10):
    """3 -> [0,0,0,1,0,0,0,0,0,0]   (what the loss function needs)"""
    v = [0.0] * classes
    v[label] = 1.0
    return v


def ascii_art(image, threshold=0.5):
    """Print a 28x28 image as text so you can eyeball it without a plot window."""
    return "\n".join("".join("#" if px > threshold else
                             ("+" if px > threshold / 2 else ".") for px in row)
                     for row in image)


if __name__ == "__main__":
    imgs, lbls = load_mnist("train", limit=5)
    print(f"loaded {len(imgs)} images, each {len(imgs[0])}x{len(imgs[0][0])}")
    print(f"pixel range: {min(min(r) for r in imgs[0])} .. {max(max(r) for r in imgs[0])}")
    for i in range(3):
        print(f"\nlabel = {lbls[i]}   one-hot = {one_hot(lbls[i])}")
        print(ascii_art(imgs[i]))