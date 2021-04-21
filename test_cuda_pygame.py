import pygame as pg
import numpy as np
import numba as nb
import numba.types
from numba import cuda
from math import sqrt, sin, cos, radians

from numba.core.typing import signature

from render_manager import RenderManager, rgb_render_function

pg.init()

font = pg.font.SysFont('arial', 20)

screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
W, H = screen.get_size()

CXS = tuple(W / 2 + cos(radians(a)) * 600 for a in (90, 210, 330))
CYS = tuple(H / 2 - sin(radians(a)) * 600 for a in (90, 210, 330))


@cuda.jit
def sin_circle_cuda(array, t):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        c = 0
        for i in nb.prange(3):
            dist = sqrt((x - CXS[i]) ** 2 + (y - CYS[i]) ** 2)
            b = (sin(dist / 20 - t * 1) + 1) / 2
            b = round(b)
            # c <<= 8
            # c |= int(b * 255)
            c += b
        # array[x, y] = (c%2)*0xFFFFFF
        array[x, y] = int(c / 3 * 0xFF) * 0x010101


@rgb_render_function(nb.types.float64, simulate=True)
def sin_circle_render(x, y, t) -> nb.types.UniTuple(nb.types.uint8, 3):
    dr = sqrt((x - CXS[0]) ** 2 + (y - CYS[0]) ** 2)
    dg = sqrt((x - CXS[1]) ** 2 + (y - CYS[1]) ** 2)
    db = sqrt((x - CXS[2]) ** 2 + (y - CYS[2]) ** 2)
    b = (sin(dr / 100 - t * 1) + 1) / 2
    b += (sin(dg / 100 - t * 1) + 1) / 2
    b += (sin(db / 100 - t * 1) + 1) / 2
    return b / 3 * 127, b / 3 * 127, 0


screen.fill((0, 0, 0))
renderer = RenderManager(screen)
# pixels = pg.surfarray.pixels2d(screen)
# screen_buffer = cuda.device_array_like(pixels)
# sin_circle_cuda[((W + 15) // 16, (H + 15) // 16), (16, 16)](screen_buffer, 0.0)
# pixels[...] = screen_buffer
# del pixels
clock = pg.time.Clock()
t = 0
pause = False
running = True
while running:
    dt = clock.tick() / 1000
    if not pause:
        t += dt
    # EVENTS
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                running = False
            elif event.key == pg.K_SPACE:
                pause = not pause

    # LOGIC

    # RENDERING
    screen.fill((0, 0, 0))
    renderer.render(sin_circle_render, t)
    # sin_circle_cuda[((W + 15) // 16, (H + 15) // 16), (16, 16)](screen_buffer, t)
    # pixels = pg.surfarray.pixels2d(screen)
    # screen_buffer.copy_to_host(pixels)
    # del pixels

    s = font.render(f"FPS: {clock.get_fps():2.2f}", True, (0, 0, 255))
    r = s.get_rect(topright=(W - 5, 5))
    screen.blit(s, r)

    pg.display.update()
