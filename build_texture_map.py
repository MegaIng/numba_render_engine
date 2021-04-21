from __future__ import annotations
import pygame as pg
import numba.cuda
import numba as nb
import numpy as np

from raycasting_3d import array, TextureIndex


def build_texture_map(blocks: list[tuple[str, str, str, str, str, str]]):
    texture_map = array(TextureIndex, (len(blocks), 6))
    textures_to_load = {}
    for i, txs in enumerate(blocks):
        for s, tx in enumerate(txs):
            if tx in textures_to_load:
                j = textures_to_load[tx][0]
            else:
                j = len(textures_to_load)
                img = pg.image.load(tx)
                textures_to_load[tx] = j, img
            texture_map[i][s].x = j
            texture_map[i][s].y = 0
            texture_map[i][s].colored = False
    textures = np.empty((16 * len(textures_to_load), 16, 3), dtype=np.uint8)
    for _, (i, tex) in textures_to_load.items():
        pixels = pg.surfarray.pixels3d(tex)
        textures[i * 16:(i + 1) * 16, :, :] = pixels
    print(texture_map, textures_to_load)
    return texture_map, textures
