from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable

import numpy as np
import numba.cuda
import numba as nb
import pygame as pg
from numba.cuda import Out


@dataclass
class Object:
    vertices: nb.cuda.devicearray.DeviceNDArray
    faces: nb.cuda.devicearray.DeviceNDArray

    @classmethod
    def from_file(cls, file):
        vertices = []
        faces = []
        for line in file:
            t = line.split()
            if t[0] == 'v':
                x, y, z, *_ = map(float, t[1:])
                vertices.append((x, y, z))
            elif t[0] == 'f':
                a, b, c = map(int, t[1:])
                assert all(x >= 0 for x in (a, b, c))
                faces.append((a, b, c))
        vert = np.array(vertices, dtype=np.float32)
        face = np.array(faces, dtype=np.uint16)
        return cls(nb.cuda.to_device(vert), nb.cuda.to_device(face))


class ObjectRenderManager:
    screen: pg.Surface = None
    screen_buffer: nb.cuda.devicearray.DeviceNDArray = None
    threads_per_block: int = None
    render_function: Callable[[nb.cuda.devicearray.DeviceNDArray, nb.float32[:], nb.uint16[:]], None] = None

    def __init__(self, screen: pg.Surface, threads_per_block=256):
        self.screen = screen
        self.threads_per_block = threads_per_block
        self.recalculate()
        self.stream = nb.cuda.stream()
        self.render_function = wireframe

    def recalculate(self):
        screen_size = self.screen.get_size()
        if self.screen_buffer is None or self.screen_buffer.shape != screen_size:
            pixels = pg.surfarray.pixels2d(self.screen)
            self.screen_buffer = nb.cuda.device_array_like(pixels)
            del pixels

    def render(self, *objects: Object):
        self.screen_buffer[:, :, :] = 0
        for obj in objects:
            args = obj.vertices, obj.faces
            if not getattr(self.render_function, '__no_cuda__', False):
                bpg = int(ceil(obj.faces.shape[0] / self.threads_per_block))
                self.render_function[bpg, self.threads_per_block](Out(self.screen_buffer), *args)
            else:
                self.render_function(self.screen_buffer, *args)
        pixels = pg.surfarray.pixels2d(self.screen)
        self.screen_buffer.copy_to_host(pixels)
        del pixels


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


def project_vertices(camera, vertices)


def wireframe(screen_buffer, vertices, faces):
    i = nb.cuda.grid(1)
    if i >= faces.shape[0]:
        return
