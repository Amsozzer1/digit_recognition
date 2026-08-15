from matrix import matrix

filters: dict[str, matrix] = {
    "identity": matrix([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]),

    "box_blur": matrix([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
    ]),

    "gaussian_blur": matrix([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16],
    ]),

    "sharpen": matrix([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ]),

    "laplacian": matrix([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0],
    ]),

    "sobel_x": matrix([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1],
    ]),

    "sobel_y": matrix([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
    ]),

    "prewitt_x": matrix([
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1],
    ]),

    "prewitt_y": matrix([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1],
    ]),

    "scharr_x": matrix([
        [ -3, 0,  3],
        [-10, 0, 10],
        [ -3, 0,  3],
    ]),

    "scharr_y": matrix([
        [ -3, -10, -3],
        [  0,   0,  0],
        [  3,  10,  3],
    ]),

    "emboss": matrix([
        [-2, -1,  0],
        [-1,  1,  1],
        [ 0,  1,  2],
    ]),
}