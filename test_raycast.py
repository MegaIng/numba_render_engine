from operator import iadd
from pathlib import Path
from random import choice

import pygame as pg
import numpy as np
import numba.cuda
import numba as nb
from numba.cuda import In

from build_texture_map import build_texture_map
from controller import Controller
from generate_world import gen_world
from render_manager import RenderManager
from raycasting_3d import raycast, CameraInfo, scalar, Vector3, angles_to_vec, Camera

world = gen_world()
world_mirror = nb.cuda.to_device(world)
camera = Camera()
camera.cuda_camera.pos = 50.5, 50.5, 50.5

screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
W, H = screen.get_size()
camera.correct_y_plane(W, H)
camera.camera_distance -= 0.5

BLOCK_TEXTURES = Path("textures", "block")

DIRT = BLOCK_TEXTURES / "dirt.png"
GRASS_TOP = BLOCK_TEXTURES / "grass_block_top.png"
GRASS_SIDE = BLOCK_TEXTURES / "grass_block_side.png"
STONE = BLOCK_TEXTURES / "stone.png"
pg.init()
tex_map, texs = build_texture_map([
    (STONE,) * 6,
    (STONE,) * 6,
    (DIRT,) * 6,
    (GRASS_SIDE,) * 4 + (GRASS_TOP, DIRT)
])
tex_map[3][4].colored = True
tex_map[3][4].color = (175,255,111)
tex_map_mirror = nb.cuda.to_device(tex_map)
texs_mirror = nb.cuda.to_device(texs)

pg.key.set_repeat(10, 1)
font = pg.font.SysFont('arial', 20)

renderer = RenderManager(screen, block_size=(16,16))
camera.update()
renderer.render(raycast, In(camera.cuda_camera), In(world_mirror), In(tex_map_mirror), In(texs_mirror))
print(next(iter(raycast.inspect_llvm().values())))
print(raycast.inspect_types())
controller = Controller()
clock = pg.time.Clock()

def setter(obj, attr, value, op=lambda a, b: b):
    setattr(obj, attr, op(getattr(obj, attr), value))


controller.add_action(pg.K_UP, lambda dt: setter(camera, 'camera_td', np.pi / 2 * dt, iadd))
controller.add_action(pg.K_DOWN, lambda dt: setter(camera, 'camera_td', -np.pi / 2 * dt, iadd))
controller.add_action(pg.K_LEFT, lambda dt: setter(camera, 'camera_lr', -np.pi * dt, iadd))
controller.add_action(pg.K_RIGHT, lambda dt: setter(camera, 'camera_lr', np.pi * dt, iadd))
controller.add_action(pg.K_w, lambda dt: camera.move_fb(dt * 10))
controller.add_action(pg.K_s, lambda dt: camera.move_fb(-dt * 10))
controller.add_action(pg.K_a, lambda dt: camera.move_lr(-dt * 10))
controller.add_action(pg.K_d, lambda dt: camera.move_lr(dt * 10))
controller.add_action(pg.K_LSHIFT, lambda dt: camera.move_ud(-dt * 5))
controller.add_action(pg.K_SPACE, lambda dt: camera.move_ud(dt * 5))


def move(a, b, f):
    a.x += b.x * f
    a.y += b.y * f
    a.z += b.z * f


running = True
while running:
    dt = clock.tick() / 1000
    # EVENTS
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                running = False

    # LOGIC
    controller.update(dt)

    # RENDERING
    screen.fill((0, 0, 0))
    camera.update()
    renderer.render(raycast, In(camera.cuda_camera), In(world_mirror), In(tex_map_mirror), In(texs_mirror))
    # renderer.render(raycast, camera.cuda_camera, world, tex_map, texs)

    s = font.render(f"FPS: {clock.get_fps():2.2f}", True, (255, 255, 255))
    r = s.get_rect(topright=(W - 5, 5))
    screen.blit(s, r)
    s = font.render(f"Pos: {camera.pos}", True, (0, 0, 255))
    r = s.get_rect(topright=r.bottomright)
    screen.blit(s, r)

    pg.display.update()
