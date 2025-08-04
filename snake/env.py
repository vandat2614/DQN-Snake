import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.math import Vector2
from pathlib import Path

from .game import Game
from .config import cell_size, number_of_cells, OFFSET, GREEN, DARK_GREEN, title_font, score_font

ASSETS = Path(__file__).parent / "assets"

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.game = Game()

        self.window_size = 2 * OFFSET + cell_size * number_of_cells
        self.surface = pygame.Surface((self.window_size, self.window_size))
        self.display_initialized = False

        self.food_image = pygame.image.load(str(ASSETS / "graphics" / "food.png"))

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.window_size, self.window_size, 3),
            dtype=np.uint8
        )

        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {'score' : 0}

    def step(self, action):

        reward = 0

        direction = self._action_to_direction(action)
        if direction + self.game.snake.direction != Vector2(0, 0):
            self.game.snake.direction = direction
        else:
            reward -= 3

        prev_score = self.game.score
        self.game.update()

        if self.game.state == "RUNNING":
            reward += 0.1
            if self.game.score == prev_score + 1:
                reward += 15
        else:
            reward -= 5

        # reward -= 10 * self.game.distance_to_food()

        terminated = self.game.state == "STOPPED"
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {'score' : self.game.score}

    def render(self):
        if not self.display_initialized and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Retro Snake")
            self.display_initialized = True

        self.surface.fill(GREEN)
        pygame.draw.rect(
            self.surface, DARK_GREEN,
            (OFFSET - 5, OFFSET - 5, cell_size * number_of_cells + 10, cell_size * number_of_cells + 10),
            width=5
        )

        # Draw food
        food_pos = self.game.food.position
        food_rect = pygame.Rect(
            OFFSET + food_pos.x * cell_size,
            OFFSET + food_pos.y * cell_size,
            cell_size, cell_size
        )
        self.surface.blit(self.food_image, food_rect)

        # Draw snake
        for segment in self.game.snake.body:
            segment_rect = pygame.Rect(
                OFFSET + segment.x * cell_size,
                OFFSET + segment.y * cell_size,
                cell_size, cell_size
            )
            pygame.draw.rect(self.surface, DARK_GREEN, segment_rect, border_radius=7)

        # Draw title and score
        title_surf = title_font.render("Retro Snake", True, DARK_GREEN)
        score_surf = score_font.render(str(self.game.score), True, DARK_GREEN)
        self.surface.blit(title_surf, (OFFSET - 5, 20))
        self.surface.blit(score_surf, (OFFSET - 5, OFFSET + cell_size * number_of_cells + 10))

        if self.render_mode == "human":
            self.window.blit(self.surface, (0, 0))
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.surface).swapaxes(0, 1)

    def close(self):
        if self.display_initialized:
            pygame.quit()
            self.display_initialized = False

    def _get_obs(self):
        # self.render()  # vẽ lại lên self.surface

        # # Lấy toàn bộ ảnh RGB từ surface
        # full_array = pygame.surfarray.array3d(self.surface).swapaxes(0, 1)

        # # Crop phần board (bỏ OFFSET, bỏ viền, bỏ chữ)
        # left = OFFSET
        # top = OFFSET
        # right = left + cell_size * number_of_cells
        # bottom = top + cell_size * number_of_cells

        # obs = full_array[top:bottom, left:right, :]
        # return obs


        snake_pos = self.game.snake.body
        food_pos = self.game.food_position

        obs = np.zeros((number_of_cells, number_of_cells), dtype=np.int32)
        for pos in snake_pos:
            if 0 <= pos.x < number_of_cells and 0 <= pos.y < number_of_cells:
                obs[int(pos.y), int(pos.x)] = 1
        obs[int(food_pos.x), int(food_pos.y)] = 1
        return obs.astype(np.uint8)


    def _draw_obs_surface(self) -> pygame.Surface:
        # Chỉ vẽ phần khung game (không có offset, border, text)
        frame = pygame.Surface((cell_size * number_of_cells, cell_size * number_of_cells))
        frame.fill((0, 0, 0))

        for segment in self.game.snake.body:
            rect = pygame.Rect(
                segment.x * cell_size,
                segment.y * cell_size,
                cell_size, cell_size
            )
            pygame.draw.rect(frame, DARK_GREEN, rect, border_radius=7)

        food_pos = self.game.food.position
        food_rect = pygame.Rect(
            food_pos.x * cell_size,
            food_pos.y * cell_size,
            cell_size, cell_size
        )
        frame.blit(self.food_image, food_rect)

        return frame

    def _action_to_direction(self, action: int) -> Vector2:
        return {
            0: Vector2(0, -1),
            1: Vector2(0, 1),
            2: Vector2(-1, 0),
            3: Vector2(1, 0),
        }.get(action, self.game.snake.direction)
