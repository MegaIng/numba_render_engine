from __future__ import annotations

from math import ceil

import pygame as pg
import numba.cuda
import numba as nb
import numpy as np
from numba.core.typing import signature
from numba.cuda import Out
from numba.cuda.compiler import Dispatcher


class RenderManager:
    screen: pg.Surface = None
    cuda_buffers: list[nb.cuda.devicearray.DeviceNDArray] = None
    threads_per_block: tuple[int, int] = None
    blocks_per_grid: tuple[int, int] = None

    def __init__(self, screen: pg.Surface, block_size: tuple[int, int] = (16, 16), scale=(1, 1), sections=1):
        self.screen = screen
        self.threads_per_block = block_size
        self.scale = scale
        self.sections = sections
        self.recalculate()
        self.stream = nb.cuda.stream()

    def recalculate(self):
        screen_size = self.screen.get_size()
        scale = self.scale
        buffer_size = (screen_size[0] + scale[0] - 1) // scale[0], (screen_size[1] + scale[1] - 1) // scale[1]
        small_buffer_size = buffer_size[0] // self.sections, buffer_size[1]
        if self.cuda_buffers is None or not all(b.shape == buffer_size for b in self.cuda_buffers):
            pixels = pg.surfarray.pixels2d(self.screen)
            self.cuda_buffers = []
            if self.sections == 1 and self.scale == (1,1):
                self.cuda_buffers.append(nb.cuda.device_array_like(pixels))
            else:
                for _ in range(self.sections):
                    self.cuda_buffers.append(nb.cuda.device_array(buffer_size, pixels.dtype))
            del pixels
        bs = self.threads_per_block
        self.blocks_per_grid = (int(ceil((small_buffer_size[0] / bs[0]))),
                                int(ceil((small_buffer_size[1] / bs[1]))))

    def render(self, cuda_function, *args):
        if not getattr(cuda_function, '__no_cuda__', False):
            cuda_function[self.blocks_per_grid, self.threads_per_block](Out(self.cuda_buffers[0]), *args)
        else:
            cuda_function(self.cuda_buffers[0], *args)
        pixels = pg.surfarray.pixels2d(self.screen)
        if self.scale == (1, 1):
            self.cuda_buffers[0].copy_to_host(pixels)
        else:
            pixels[...] = np.repeat(self.cuda_buffers[0], self.scale[0], axis=0).repeat(self.scale[1], axis=1)
        del pixels


# We have to manually generate the code since nb.cuda currently can't deal with *args
# https://github.com/numba/numba/issues/6891
RENDER_TEMPLATE = """
def get_render(device_function, kwargs):
    @nb.cuda.jit(**kwargs)
    def render(array, {args}):
        x, y = nb.cuda.grid(2)
        if x < array.shape[0] and y < array.shape[1]:
            r, g, b = device_function(x/array.shape[0], y/array.shape[1], {args})
            array[x, y] = b | g << 8 | r << 16
    return render
"""


def rgb_render_function(*extra_types, simulate=False, **kwargs):
    df_signature = signature(nb.types.UniTuple(nb.types.uint8, 3), nb.types.float32, nb.types.float32, *extra_types)

    if not simulate:
        def inner(func):
            device_function = nb.cuda.jit(df_signature, device=True, **kwargs)(func)
            args = ', '.join(f'arg{i}' for i in range(len(extra_types)))
            code = RENDER_TEMPLATE.format(args=args)
            lcs = {}
            exec(code, globals(), lcs)
            # device_function.inspect_llvm()
            return lcs['get_render'](device_function, kwargs)
    else:
        def inner(func):
            device_function = nb.jit(df_signature, nopython=True)(func)

            @nb.njit(parallel=True)
            def render(array, *args):
                for x in nb.prange(array.shape[0]):
                    for y in nb.prange(array.shape[1]):
                        r, b, g = device_function(nb.float32(x / array.shape[0]), nb.float32(y / array.shape[1]), *args)
                        array[x, y] = r | g << 8 | b << 16

            render.__no_cuda__ = True
            return render
    return inner
