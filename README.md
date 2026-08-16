# digit-recognition

A convolutional neural network for MNIST handwritten-digit classification, written from
scratch in pure Python — no NumPy, no PyTorch, no TensorFlow in the network itself. Every
matrix multiply, convolution, ReLU, softmax and gradient is implemented by hand in
[matrix.py](matrix.py) and [utils.py](utils.py).

The point is to understand a CNN by building one, not to be fast.

## Status

The **forward pass runs end to end**: an MNIST image goes in, a 10-class probability
distribution comes out.

The **dense head trains**. Backprop through softmax + cross-entropy, `Dense 2`, ReLU and
`Dense 1` is implemented and numerically gradient-checked (worst relative error ~1e-9
against central differences). `W1/b1/W2/b2` are updated by SGD each image.

**The conv filters are still frozen** at their He initialization. `feature_extraction`
returns the per-layer caches that a conv backward pass will need, but nothing consumes
them yet — so the network is effectively training a classifier on top of random (but
fixed) convolutional features.

## Architecture

| Stage | Operation | Output shape |
| --- | --- | --- |
| Input | 28×28 grayscale image | `1 × 28 × 28` |
| Conv 1 | 32 filters, 3×3×1, stride 1 | `32 × 26 × 26` |
| | ReLU → maxPool 2×2 | `32 × 13 × 13` |
| Conv 2 | 64 filters, 3×3×32, stride 1 | `64 × 11 × 11` |
| | ReLU → maxPool 2×2 | `64 × 5 × 5` |
| Flatten | | `1600` |
| Dense 1 | 1600 → 64 → ReLU | `64` |
| Dense 2 | 64 → 10 → softmax | `10` |

Downsampling is done with **2×2 max pooling**, applied after the ReLU. Convolutions use
stride 1 and no padding, so each one shrinks the map by 2 pixels per side.

Note that pooling floors: `maxPool(2)` on the 11×11 output of Conv 2 gives 5×5 and
discards the last row and column.

Weight initialization is He: `sqrt(2 / fan_in)`, where `fan_in` is `k*k*c_in` for kernels
and `n_in` for dense layers.

## Training

[init.py](init.py) is the entry point. The loop is:

```
for epoch in range(epochs):
    for i in range(train_size // batch_size):
        run_batch(i)          # images [i*batch_size, i*batch_size + batch_size)
    save_checkpoint(...)
```

Configured at the top of [init.py](init.py) — `train_size`, `batch_size`, `epochs` — with
the learning rate in [consts.py](consts.py) (`LR = 0.01`).

Two things worth knowing about the loop:

- **`batch_size` only slices the data.** `sgd_update` fires once per image, so this is
  per-sample SGD, not mini-batch gradient descent. Real mini-batching would mean
  accumulating `dW`/`db` across the batch and applying one averaged update.
- **The data is not shuffled.** Images are consumed in dataset order, the same order
  every epoch.

Average loss and training accuracy are printed per epoch. `ln(10) ≈ 2.3026` is the
reference point — that is the loss of a uniform 10-class guess.

## Checkpoints

`save_checkpoint` writes `weights/epoch<N>.json` after every epoch, via a `.tmp` file and
an atomic rename. It stores the dense weights, the biases, **and the conv kernels**, so a
checkpoint fully describes the model rather than relying on the RNG seed to reproduce the
filters:

```json
{
  "epoch":   3,
  "shapes":  [[64, 1600], [10, 64]],
  "weights": [[...]],
  "biases":  [[...]],
  "kernels": [[[...]]]
}
```

`kernels` is indexed `[layer][c_out][c_in]`, each entry a 3×3 kernel. `load_checkpoint`
returns `(epoch, weights, biases, kernels, meta)`; any extra keyword arguments passed to
`save_checkpoint` are stored alongside and come back in `meta`.

`weights/` is gitignored.

## Requirements

Python **3.12+** (the `type X = ...` alias syntax in [typedefs.py](typedefs.py)
requires it). Developed on 3.14.

The network math is standard-library only, but two things need third-party packages:

```bash
pip install numpy pillow
```

- [matrix.py](matrix.py) uses NumPy and Pillow in `saveAs`, to write feature maps and
  weight matrices out as PNGs.
- [infer.py](infer.py) uses NumPy.

[mnist_loader.py](mnist_loader.py) downloads and parses the IDX files with the standard
library alone, so there is nothing to install for the data itself.

## Quick start

```bash
git clone https://github.com/<your-username>/digit-recognition.git
cd digit-recognition

# Inspect the dataset — downloads ~10 MB into mnist_data/ on first run
python mnist_loader.py

# Train (writes weights/epoch<N>.json after each epoch)
python init.py

# Score a saved checkpoint against the test split
python infer.py
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

**Pure-Python convolution is slow** — roughly 1.6 s per image, essentially all of it in
`feature_extraction`. That cost is paid on every epoch, so a full run is
`train_size × epochs × 1.6 s`. Lower `train_size` while experimenting.

## Files

| File | Purpose |
| --- | --- |
| [matrix.py](matrix.py) | The `matrix` class: multiply, element-wise product, sliding `window`, `conv2D`, `reLu`, `maxPool`, `flatten`, `saveAs` |
| [utils.py](utils.py) | He initialization, dense layers, ReLU, softmax, cross-entropy, `dense_backward`, `sgd_update`, checkpoint save/load, and the `feature_extraction` / `classification` pipelines |
| [typedefs.py](typedefs.py) | The `Layer`, `ConvCache` and `DenseCache` type aliases |
| [mnist_loader.py](mnist_loader.py) | Stdlib-only IDX downloader and parser, plus `one_hot` and `ascii_art` helpers |
| [init.py](init.py) | Entry point: builds the network, runs the training loop, checkpoints each epoch |
| [infer.py](infer.py) | Loads a checkpoint and scores it against the test split. Currently reads the JSON by hand and rebuilds the conv filters from `random.seed(0)` rather than calling `load_checkpoint` — correct only while the filters stay frozen |
| [consts.py](consts.py) | The learning rate, plus a library of classic hand-designed 3×3 kernels (Sobel, Scharr, Prewitt, Gaussian blur, emboss…) useful for sanity-checking `conv2D` |
| [del.py](del.py) | Scratch script for eyeballing `maxPool` on a small matrix |

Passing `dump="<name>"` to `feature_extraction` writes every feature map to
`out/<name>/L<layer>/<filter>.png`, which is the quickest way to see what the conv stack
is actually responding to. `out/` is gitignored.

## Roadmap

- [x] Cache per-layer activations during the forward pass (`ConvCache`, `DenseCache`)
- [x] Backward pass: gradients for dense, ReLU, softmax + cross-entropy
- [x] SGD training loop over epochs and batches
- [x] Save and load trained weights, biases and kernels
- [x] Evaluate against the test split ([infer.py](infer.py))
- [ ] Gradient for `conv2D` — train the filters instead of freezing them
- [ ] True mini-batch updates (accumulate gradients across a batch)
- [ ] Shuffle the training order between epochs
- [ ] Held-out validation loss during training, not just training loss

## License

[MIT](LICENSE)
