# digit-recognition

A convolutional neural network for MNIST handwritten-digit classification, written from
scratch in pure Python — no NumPy, no PyTorch, no TensorFlow. Every matrix multiply,
convolution, ReLU and softmax is implemented by hand in [matrix.py](matrix.py) and
[utils.py](utils.py).

The point is to understand a CNN by building one, not to be fast.

## Status

The **forward pass is complete and runs end to end**: an MNIST image goes in, a
10-class probability distribution comes out.

**Backpropagation and the training loop are not implemented yet.** Weights are
He-initialized and never updated, so predictions are currently random.

## Architecture

| Stage | Operation | Output shape |
| --- | --- | --- |
| Input | 28×28 grayscale image | `1 × 28 × 28` |
| Conv 1 | 32 filters, 3×3, stride 2 → ReLU | `32 × 13 × 13` |
| Conv 2 | 64 filters, 3×3, stride 2 → ReLU | `64 × 6 × 6` |
| Flatten | | `2304` |
| Dense 1 | 2304 → 128 → ReLU | `128` |
| Dense 2 | 128 → 10 → softmax | `10` |

Downsampling is done with **stride-2 convolutions** rather than max pooling. A
`maxPool` method exists on `matrix` and works, but it is not currently wired into
the layer stack.

## Requirements

Python **3.12+** (the `type X = ...` alias syntax in [typedefs.py](typedefs.py)
requires it). Developed on 3.14.

The network itself has **no third-party dependencies** — the standard library is
enough. [mnist_loader.py](mnist_loader.py) downloads and parses the IDX files
directly.

## Quick start

```bash
git clone https://github.com/<your-username>/digit-recognition.git
cd digit-recognition

# Inspect the dataset — downloads ~10 MB into mnist_data/ on first run
python mnist_loader.py

# Run a forward pass over the first 5 training images
python init.py
```

`mnist_loader.py` prints a few digits as ASCII art so you can eyeball the data
without a plotting library:

```
label = 5   one-hot = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
................+##.###+....
..........+############+....
........##########+++.......
........##########..........
........+#+###...#..........
...........##+..............
...........###..............
............##+.............
.............###+...........
.............+###+..........
...............####.........
................+###........
.................###+.......
...............#####........
.............#######........
...........+######+.........
.........+######+...........
.......#######+.............
.....########...............
....#######.................
```

Pure-Python convolution is slow — expect a couple of seconds per image. Keep the
`limit=` argument to `load_mnist` small while experimenting.

## Files

| File | Purpose |
| --- | --- |
| [matrix.py](matrix.py) | The `matrix` class: multiply, element-wise product, sliding `window`, `conv2D`, `reLu`, `maxPool`, `flatten` |
| [utils.py](utils.py) | He initialization, dense layers, ReLU, softmax, and the `feature_extraction` / `classification` pipelines |
| [typedefs.py](typedefs.py) | The `Layer` type alias — a `(filters, stride)` pair |
| [mnist_loader.py](mnist_loader.py) | Stdlib-only IDX downloader and parser, plus `one_hot` and `ascii_art` helpers |
| [init.py](init.py) | Entry point: builds the network and runs the forward pass |
| [consts.py](consts.py) | A library of classic hand-designed 3×3 kernels (Sobel, Scharr, Prewitt, Gaussian blur, emboss…), useful for sanity-checking `conv2D` |
| [setup.py](setup.py) | Optional alternative: pulls MNIST from Kaggle via `kagglehub`. Not needed — `mnist_loader.py` handles downloading on its own |

`requirements.txt` pins the dependencies for that optional Kaggle path only. You
can skip it entirely.

## Roadmap

- [ ] Backward pass: gradients for dense, ReLU, softmax + cross-entropy
- [ ] Gradient for `conv2D`
- [ ] SGD training loop with mini-batches
- [ ] Evaluate on the 10k test split
- [ ] Save and load trained weights

## License

[MIT](LICENSE)
