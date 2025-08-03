import pygame
from pygame.math import Vector2
from .config import cell_size, OFFSET

from pathlib import Path

ASSETS = Path(__file__).parent / "assets"

class Snake:
    def __init__(self, sound=False):
        self.body = [Vector2(6, 9), Vector2(5, 9), Vector2(4, 9)]
        self.direction = Vector2(1, 0)
        self.add_segment = False

        if sound:
            self.eat_sound = pygame.mixer.Sound(str(ASSETS / "Sounds" / "eat.mp3"))
            self.wall_hit_sound = pygame.mixer.Sound(str(ASSETS / "Sounds" / "wall.mp3"))

    def update(self):
        self.body.insert(0, self.body[0] + self.direction)
        if not self.add_segment:
            self.body.pop()
        else:
            self.add_segment = False

    def reset(self):
        self.__init__()
