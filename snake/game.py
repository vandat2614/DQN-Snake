from .snake import Snake
from .food import Food
from .config import number_of_cells

import math

class Game:
    def __init__(self, sound=False):
        self.snake = Snake()
        self.food = Food(self.snake.body)
        self.state = "RUNNING"
        self.score = 0
        self.sound = sound

    def distance_to_food(self) -> float:
        head = self.snake.body[0]
        food = self.food_position
        return (head - food).length()  # dùng Euclidean
        # hoặc: return abs(head.x - food.x) + abs(head.y - food.y)  # Manhattan


    def update(self):
        if self.state != "RUNNING":
            return
        self.snake.update()
        self.check_collision_with_food()
        self.check_collision_with_edges()
        self.check_collision_with_tail()

    def check_collision_with_food(self):
        if self.snake.body[0] == self.food.position:
            self.food.position = self.food.generate_random_pos(self.snake.body)
            self.snake.add_segment = True
            self.score += 1
            if self.sound:
                self.snake.eat_sound.play()

    def check_collision_with_edges(self):
        head = self.snake.body[0]
        if head.x < 0 or head.x >= number_of_cells or head.y < 0 or head.y >= number_of_cells:
            self.game_over()

    def check_collision_with_tail(self):
        if self.snake.body[0] in self.snake.body[1:]:
            self.game_over()

    def reset(self):
        self.state = "RUNNING"
        self.snake.reset()
        self.food_position = self.food.generate_random_pos(self.snake.body)
        self.score = 0

    def game_over(self):
        self.state = "STOPPED"
        if self.sound:
            self.snake.wall_hit_sound.play()
