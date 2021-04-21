from __future__ import annotations

from dataclasses import field, dataclass
from math import copysign, floor, ceil

from numba.core.typing import signature

from render_manager import rgb_render_function
import numpy as np
import numba.cuda
import numba as nb

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


@nb.cuda.jit(signature(*(nb.float32,) * 4), device=True)
def inter1d(a, b, p):
    return (b - a) * p + a


@nb.cuda.jit(signature(*(nb.float32,) * 7), device=True)
def inter2d(a1, b1, a2, b2, p1, p2):
    c1 = inter1d(a1, b1, p1)
    c2 = inter1d(a2, b2, p1)
    return inter1d(c1, c2, p2)


@nb.cuda.jit(device=True)
# @nb.jit()
def get_color(u, v, tex_i, textures):
    x = int(tex_i.x * 16 + u * 16)
    y = int(tex_i.y * 16 + v * 16)
    r = textures[x][y][0]
    g = textures[x][y][1]
    b = textures[x][y][2]
    if tex_i.colored:
        f = nb.float32(r / 255)
        return tex_i.color[0] * f, tex_i.color[1] * f, tex_i.color[2] * f
    return r, g, b


@nb.cuda.jit(device=True)
# @nb.jit()
def get_color_smooth(u, v, tex_i, textures):
    x = u * 16
    y = v * 16
    x1 = tex_i.x * 16 + int(floor(x))
    y1 = tex_i.y * 16 + int(floor(y))
    x2 = tex_i.x * 16 + int(ceil(x)) % 16
    y2 = tex_i.y * 16 + int(ceil(y)) % 16
    r = inter2d(textures[x1, y1, 0], textures[x2, y1, 0],
                textures[x1, y2, 0], textures[x2, y2, 0],
                x % 1, y % 1)
    g = inter2d(textures[x1, y1, 1], textures[x2, y1, 1],
                textures[x1, y2, 1], textures[x2, y2, 1],
                x % 1, y % 1)
    b = inter2d(textures[x1, y1, 2], textures[x2, y1, 2],
                textures[x1, y2, 2], textures[x2, y2, 2],
                x % 1, y % 1)
    if tex_i.colored:
        f = r / 255
        return tex_i.color[0] * f, tex_i.color[1] * f, tex_i.color[2] * f
    return r, g, b


@rgb_render_function(nb.typeof(scalar(CameraInfo)), nb.types.uint8[:, :, :], nb.typeof(array(TextureIndex, (1, 6))),
                     nb.types.uint8[:, :, :],
                     max_registers=32, fastmath=True, error_model='numpy',
                     debug=False, opt=True
                     )
def raycast(sx, sy, camera, world, texture_map, textures):
    fx = nb.float32(sx * 2 - 1)
    fy = nb.float32(sy * 2 - 1)
    dx = nb.float32(camera.plane_offset.x + camera.plane_x_size.x * fx + camera.plane_y_size.x * fy)
    dy = nb.float32(camera.plane_offset.y + camera.plane_x_size.y * fx + camera.plane_y_size.y * fy)
    dz = nb.float32(camera.plane_offset.z + camera.plane_x_size.z * fx + camera.plane_y_size.z * fy)

    ddx = nb.float32(abs(1 / dx))
    ddy = nb.float32(abs(1 / dy))
    ddz = nb.float32(abs(1 / dz))
    tx = int(camera.pos.x // 1)
    ty = int(camera.pos.y // 1)
    tz = int(camera.pos.z // 1)
    ox = camera.pos.x % 1
    oy = camera.pos.y % 1
    oz = camera.pos.z % 1
    sx = nb.cuda.selp(dx < 0, -1, 1)
    ox = nb.cuda.selp(dx < 0, ox, (1 - ox)) * ddx
    sy = nb.cuda.selp(dy < 0, -1, 1)
    oy = nb.cuda.selp(dy < 0, oy, (1 - oy)) * ddy
    sz = nb.cuda.selp(dz < 0, -1, 1)
    oz = nb.cuda.selp(dz < 0, oz, (1 - oz)) * ddz

    finished = False
    while not finished:
        if oz > ox < oy:
            ox += ddx
            tx += sx
            side = int(0 + (sx + 1) // 2)
        elif oz > oy < ox:
            oy += ddy
            ty += sy
            side = int(2 + (sy + 1) // 2)
        else:
            oz += ddz
            tz += sz
            side = int(4 + (sz + 1) // 2)
        if not ((fx := (0 <= tx < world.shape[0])) and
                (fy := (0 <= ty < world.shape[1])) and
                (fz := (0 <= tz < world.shape[2]))):
            finished = True  # (fx and sx*tx>0) or (fy and sy*ty>0) or ((not 0 <= tz < world.shape[2]) and sz*tz>0)
            continue
        t = world[tx, ty, tz]
        if t != 0:
            if side < 2:
                dist = (tx - camera.pos.x + (1 - sx) / 2) / dx
                tex_u = nb.float32((dist * dy + camera.pos.y) % 1)
                tex_v = nb.float32(-(dist * dz + camera.pos.z) % 1)
            elif side < 4:
                dist = (ty - camera.pos.y + (1 - sy) / 2) / dy
                tex_u = nb.float32((dist * dx + camera.pos.x) % 1)
                tex_v = nb.float32(-(dist * dz + camera.pos.z) % 1)
            else:
                dist = (tz - camera.pos.z + (1 - sz) / 2) / dz
                tex_u = nb.float32((dist * dx + camera.pos.x) % 1)
                tex_v = nb.float32((dist * dy + camera.pos.y) % 1)
            return get_color(tex_u, tex_v, texture_map[t][side], textures)
    return 0, 0, 0


print(raycast.inspect_types())

NORTH = 0  # positive x
SOUTH = 1  # negative x
WEST = 2  # positive y
EAST = 3  # negative y
TOP = 4  # positive z
BOTTOM = 5  # negative z


def angles_to_vec(alpha, beta, length):
    return (np.cos(alpha) * np.cos(beta) * length,
            np.sin(alpha) * np.cos(beta) * length,
            np.sin(beta) * length)


@dataclass
class Camera:
    cuda_camera: CameraInfo = field(default_factory=lambda: scalar(CameraInfo))
    camera_lr: float = 0
    camera_td: float = 0
    camera_distance: float = 1
    camera_x_plane: float = 1
    camera_y_plane: float = 1

    def correct_y_plane(self, w, h):
        self.camera_y_plane = h / w * self.camera_x_plane

    @property
    def pos(self):
        return self.cuda_camera.pos

    @pos.setter
    def pos(self, value):
        self.cuda_camera.pos = value

    def move_fb(self, d):
        direction = angles_to_vec(self.camera_lr, 0, d)
        self.cuda_camera.pos.x += direction[0]
        self.cuda_camera.pos.y += direction[1]
        self.cuda_camera.pos.z += direction[2]

    def move_lr(self, d):
        direction = angles_to_vec(self.camera_lr + np.pi / 2, 0, d)
        self.cuda_camera.pos.x += direction[0]
        self.cuda_camera.pos.y += direction[1]
        self.cuda_camera.pos.z += direction[2]

    def move_ud(self, d):
        self.cuda_camera.pos.z += d

    def update(self):
        self.camera_lr %= 2 * np.pi
        self.camera_td = min(max(self.camera_td, -np.pi / 2), np.pi / 2)
        self.cuda_camera.plane_offset = angles_to_vec(
            self.camera_lr,
            self.camera_td,
            self.camera_distance)
        self.cuda_camera.plane_x_size = angles_to_vec(
            self.camera_lr + np.pi / 2,
            0,
            self.camera_x_plane)
        self.cuda_camera.plane_y_size = angles_to_vec(
            self.camera_lr,
            self.camera_td - np.pi / 2,
            self.camera_y_plane)
