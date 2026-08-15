from matrix import matrix

type Layer = tuple[
    list[list[matrix]], # filters: [C_out][C_in]
    int,                # stride
]