from __future__ import annotations

from dataclasses import field, dataclass
from math import copysign, floor, ceil
from pathlib import Path

from numba.core.typing import signature

import numpy as np
import numba.cuda
import numba as nb

from build_texture_map import build_texture_map
from generate_world import gen_world

Vector3 = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32)
])

CameraInfo = np.dtype([
    ('pos', Vector3),
    ('plane_offset', Vector3),
    ('plane_x_size', Vector3),
    ('plane_y_size', Vector3),
])


def scalar(dtype):
    return np.rec.array(None, shape=(), dtype=dtype)[()]


def array(dtype, shape):
    return np.rec.array(None, shape=shape, dtype=dtype)


s = scalar(Vector3)

COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0)
)

TextureIndex = np.dtype([
    ('x', np.uint8),
    ('y', np.uint8),
    ('colored', np.bool_),
    ('color', np.uint8, 3)
])

@nb.cuda.jit(signature(nb.types.UniTuple(nb.types.uint8, 3), nb.types.float32, nb.types.float32,
                       nb.typeof(scalar(CameraInfo)), nb.types.uint8[:, :, :]),
             max_registers=32, fastmath=True, device=True,
             debug=True, opt=False, inline='never'
             )
def raycast(sx, sy, camera, world):
    fx = nb.float32(sx * 2 - 1)
    fy = nb.float32(sy * 2 - 1)
    dx = nb.float32(camera.plane_offset.x + camera.plane_x_size.x * fx + camera.plane_y_size.x * fy)
    dy = nb.float32(camera.plane_offset.y + camera.plane_x_size.y * fx + camera.plane_y_size.y * fy)

    ddx = nb.float32(abs(1 / dx) if dx != 0 else np.inf)
    ddy = nb.float32(abs(1 / dy) if dy != 0 else np.inf)
    tx = int(camera.pos.x // 1)
    ty = int(camera.pos.y // 1)
    ox = camera.pos.x % 1
    oy = camera.pos.y % 1
    sx = nb.cuda.selp(dx < 0, -1, 1)
    ox = nb.cuda.selp(dx < 0, ox, (1 - ox)) * ddx
    sy = nb.cuda.selp(dy < 0, -1, 1)
    oy = nb.cuda.selp(dy < 0, oy, (1 - oy)) * ddy

    finished = False
    while not finished:
        ox += ddx
        tx += sx
        if not (0 <= tx < world.shape[0]):
            finished = True
            continue
    return 0, 0, 0


@nb.cuda.jit(signature(nb.void, nb.types.int32[:,:],
                       nb.typeof(scalar(CameraInfo)), nb.types.uint8[:, :, :]),
             max_registers=32, fastmath=True,
    debug=True, opt=False
)
def render(array, camera, world):
    x, y = nb.cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        r, g, b = raycast(x / array.shape[0], y / array.shape[1], camera, world)
        # r, g, b = x,y,1
        array[x, y] = b | g << 8 | r << 16


NORTH = 0  # positive x
SOUTH = 1  # negative x
WEST = 2  # positive y
EAST = 3  # negative y
TOP = 4  # positive z
BOTTOM = 5  # negative z
world = gen_world((25,25,25))
world_mirror = nb.cuda.to_device(world)
camera = scalar(CameraInfo)
camera.pos = (12.5, 12.5, 12.5)
camera.plane_offset = (1., 0., 0.)
camera.plane_x_size = (6.123234e-17, 1., 0.)
camera.plane_y_size =  (6.123234e-17, 0., -1.)
print(camera)

W, H = 750, 750

BLOCK_TEXTURES = Path("textures", "block")

DIRT = BLOCK_TEXTURES / "dirt.png"
GRASS_TOP = BLOCK_TEXTURES / "grass_block_top.png"
GRASS_SIDE = BLOCK_TEXTURES / "grass_block_side.png"
STONE = BLOCK_TEXTURES / "stone.png"

tex_map, texs = build_texture_map([
    (STONE,) * 6,
    (STONE,) * 6,
    (DIRT,) * 6,
    (GRASS_SIDE,) * 4 + (GRASS_TOP, DIRT)
])
tex_map[3][4].colored = True
tex_map[3][4].color = (175, 255, 111)
tex_map_mirror = nb.cuda.to_device(tex_map)
texs_mirror = nb.cuda.to_device(texs)

arr = np.empty((W, H), np.int32)
arr_mirror = nb.cuda.to_device(arr)

print(arr_mirror.size*8/1024/1024, "MiB")

render[(int(ceil(W/4)), int(ceil(H/4))),(4,4)](arr_mirror, camera, world_mirror)