from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import pygame as pg

@dataclass
class Controller:
    actions: dict[int, Callable[[float], None]] = field(default_factory=dict)

    def add_action(self, key, callback: Callable[[float], None]):
        self.actions[key] = callback
    
    def update(self, dt: float):
        pressed = pg.key.get_pressed()
        for k, a in self.actions.items():
            if pressed[k]:
                a(dt)

