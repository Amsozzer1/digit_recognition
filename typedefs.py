from matrix import matrix

type Layer = tuple[
    list[list[matrix]], # filters: [C_out][C_in]
    int,                # stride
]

# What the backward pass needs from each conv layer. dW needs the maps that
# were fed in; the ReLU mask needs the conv sums from before the activation.
type ConvCache = tuple[
    list[matrix],       # inputs:  maps entering the layer  [C_in]
    list[matrix],       # pre_act: conv sums before ReLU    [C_out]
]

type DenseCache = tuple[
    list[float],   # flattened: input to dense 1        [2304]
    list[float],   # h: post-ReLU hidden activations    [128]
]