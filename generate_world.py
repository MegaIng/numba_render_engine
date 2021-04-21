from __future__ import annotations

from noise import snoise2, snoise3

import numpy as np


def gen_world(size=(100, 100, 100)):
    world = np.zeros(size, dtype=np.uint8)
    for x in range(size[0]):
        for y in range(size[1]):
            h = int(round(size[2] / 2 + snoise2(x / 100, y / 100) * size[2] / 6))
            world[x, y, :h - 3] = 1
            world[x, y, h - 3:h - 1] = 2
            world[x, y, h - 1:h] = 3
            for z in range(size[2]):
                m = snoise3(x / 50, y / 50, z / 50)
                if m > 0.8:
                    world[x, y, z] = 0
    return world
