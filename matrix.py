from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from PIL import Image


class matrix:
    def __init__(self, ls: list[list[float]] | None = None,r:int=0, c:int=0) -> None:
        if ls is None:
            ls = [[]]
        self.rows: int = r if r != 0 else len(ls)
        self.cols: int = c if c != 0 else len(ls[0])
        self.values: list[list[float]] = [[ls[i][j] for j in range(self.cols)] for i in range(self.rows)]

    def size(self) -> tuple[int,int]:
        return self.rows, self.cols

    def __getitem__(self, index: int) -> list[float] | float:
        return self.values[index]

    def __setitem__(self, index: int, value: list[float] | float) -> None:
        if isinstance(value, list):
            if len(value) != self.cols:
                raise ValueError(
                    f"Row must contain {self.cols} values, got {len(value)}"
                )
            self.values[index] = value[:]  # copy the supplied row
        else:
            self.values[index] = [float(value)] * self.cols

    def print(self):
        print("[")
        for i in range(self.rows):
            print(" ", self.values[i])
        print("]")
        
    def mult(self, b: "matrix") -> "matrix":
        if self.cols != b.rows:
            raise Exception("Matrices can not multiplied C1 != R2")
        
        ret = matrix([[0.0] * b.cols for _ in range(self.rows)])
        for i in range(self.rows):
            for j in range(b.cols):
                s:float=0
                for k in range(self.cols):
                    s += self.values[i][k] * b.values[k][j]
                ret[i][j] = s # type: ignore
                
        return ret

    Mode: TypeAlias = Literal["Center", "Top", "Bottom"]
    def _positions(self, mode: Mode, scale: int) -> tuple[int, int]:
        middle = scale // 2
        last = scale - 1

        positions: dict[Mode, tuple[int, int]] = { # type: ignore  # noqa: F821
            "Center": (middle, middle),
            "Top": (0, 0),        
            "Bottom": (last, last),
        }
        return positions[mode]

    def window(
    self,
    idx_r: int,
    idx_c: int,
    scale: int,
    mode: Mode = "Center",
) -> "matrix":
        ret = [[0.0] * scale for _ in range(scale)]

        anchor_r, anchor_c = self._positions(mode, scale)

        start_r = idx_r - anchor_r
        start_c = idx_c - anchor_c

        for out_r in range(scale):
            source_r = start_r + out_r

            for out_c in range(scale):
                source_c = start_c + out_c

                if 0 <= source_r < self.rows and 0 <= source_c < self.cols:
                    ret[out_r][out_c] = self.values[source_r][source_c]
                else:
                    ret[out_r][out_c] = 0.0

        return matrix(ret)

    def sum(self) -> float:
        s = 0 
        for i in range(self.rows):
            for j in range(self.cols):
                s+=self.values[i][j]
        return s

    def dotProduct(self, b: "matrix") -> "matrix":
        if (self.rows, self.cols) != (b.rows, b.cols):
            raise Exception("Element-wise product needs identical shapes")
        return matrix([[self.values[i][j] * b.values[i][j]
                        for j in range(self.cols)] for i in range(self.rows)])

    def conv2D(self, kernel: "matrix", x: int, y: int,z:int, stride:int=1, pos: Mode = "Top") -> "matrix":
        s=kernel.rows
        ls = []
        for i in range(0, self.rows-s+1, stride):
            t = []
            for j in range(0, self.cols-s+1, stride):
                curr_window = self.window(i,j,s,pos)
                product = curr_window.dotProduct(kernel)
                # product.saveAs("out/conv2D/"+str(i)+"/"+str(j))
                t.append(product.sum())
            ls.append(t)
        mat = matrix(ls)
        # if mat.size() == (26,26):
        #     mat.saveAs("out/conv2D/"+str(x)+"/"+str(y))
        mat.saveAs("out/conv2D/"+str(x)+"/"+str(y)+"/"+str(z))
        return mat

    def reLu(self) -> "matrix":
        m = [[0.0]*self.cols for i in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                m[i][j] = max(0.0, self.values[i][j])
        return matrix(m)

    def _max(self) -> float:
        curr = self.values[0][0]
        for i in range(self.rows):
            for j in range(self.cols):
                curr = max(curr, self.values[i][j])
        return curr

    def maxPool(self, k: int = 2) -> "matrix":
        out_r, out_c = self.rows // k, self.cols // k
        out = [[0.0] * out_c for _ in range(out_r)]
        for i in range(out_r):
            for j in range(out_c):
                out[i][j] = self.window(i * k, j * k, k, "Top")._max()
        return matrix(out)

    def add(self, x: "matrix") -> "matrix":
        if self.rows != x.rows or self.cols != x.cols:
            raise Exception("Matrices can not added")
        return matrix([[self.values[i][j] + x.values[i][j] for j in range(self.cols)]
               for i in range(self.rows)])

    def flatten(self) -> list[float]:
        r = []
        for i in range(self.rows):
            for j in range(self.cols):
                r.append(self.values[i][j])
        return r

    def saveAs(self, location: str):
        """Min-max normalise per image before writing. Conv pre-activations are
        signed and zero-mean, so clipping to [0,1] would render most of the map
        as black."""
        path = Path(location)
        path.parent.mkdir(parents=True, exist_ok=True)

        img = np.array(self.values, dtype=float)
        lo, hi = img.min(), img.max()
        img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(path.with_name(path.name + ".png"))