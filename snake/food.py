import random
from pygame.math import Vector2
from .config import number_of_cells

class Food:
    def __init__(self, snake_body):
        self.position = self.generate_random_pos(snake_body)

    def generate_random_cell(self):
        x = random.randint(0, number_of_cells - 1)
        y = random.randint(0, number_of_cells - 1)
        return Vector2(x, y)

    def generate_random_pos(self, snake_body):
        position = self.generate_random_cell()
        while position in snake_body:
            position = self.generate_random_cell()
        return position
