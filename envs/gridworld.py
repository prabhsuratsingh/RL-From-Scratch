import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
import os
import pickle


CELL_SIZE = 100
MARGIN = 10

def get_coords(row, col, loc='center'):
    xc = (col+1.5) * CELL_SIZE
    yc = (row+1.5) * CELL_SIZE

    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = CELL_SIZE//2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = xc - half_size, xc + half_size

        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == 'interior_triangle':
        x1, y1 = xc, yc + CELL_SIZE//3
        x2, y2 = xc + CELL_SIZE//3, yc - CELL_SIZE//3
        x3, y3 = xc - CELL_SIZE//3, yc - CELL_SIZE//3

        return [(x1,y1), (x2,y2), (x3,y3)]
    
def draw_object(screen, coords_list):
    if len(coords_list) == 1:  
        pygame.draw.circle(
            screen,
            (50, 50, 50),
            coords_list[0],
            int(0.45 * CELL_SIZE)
        )

    elif len(coords_list) == 3:  
        pygame.draw.polygon(
            screen,
            (230, 160, 50),
            coords_list
        )

    elif len(coords_list) > 3:  
        pygame.draw.polygon(
            screen,
            (100, 100, 200),
            coords_list
        )


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, num_rows=5, num_cols=6, delay=0.05, render_mode=None):
        super().__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay = delay
        self.render_mode = render_mode

        # Actions: up, right, down, left
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(num_rows * num_cols)

        self._init_actions()
        self._init_states()
        self._init_terminals()

        self.state = 0

        # Rendering
        self.window = None
        self.clock = None

    def _init_actions(self):
        self.action_defs = {
            0: lambda r, c: (max(r - 1, 0), c),
            1: lambda r, c: (r, min(c + 1, self.num_cols - 1)),
            2: lambda r, c: (min(r + 1, self.num_rows - 1), c),
            3: lambda r, c: (r, max(c - 1, 0)),
        }

    def _init_states(self):
        self.grid2state = {
            (r, c): r * self.num_cols + c
            for r in range(self.num_rows)
            for c in range(self.num_cols)
        }
        self.state2grid = {v: k for k, v in self.grid2state.items()}

    def _init_terminals(self):
        self.gold_cell = (self.num_rows // 2, self.num_cols - 2)

        self.trap_cells = [
            (self.gold_cell[0] + 1, self.gold_cell[1]),
            (self.gold_cell[0], self.gold_cell[1] - 1),
            (self.gold_cell[0] - 1, self.gold_cell[1]),
        ]

        self.gold_state = self.grid2state[self.gold_cell]
        self.trap_states = [self.grid2state[c] for c in self.trap_cells]
        self.terminal_states = [self.gold_state] + self.trap_states

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def step(self, action):
        if self.state in self.terminal_states:
            return self.state, 0.0, True, False, {}

        r, c = self.state2grid[self.state]
        nr, nc = self.action_defs[action](r, c)
        next_state = self.grid2state[(nr, nc)]

        terminated = next_state in self.terminal_states
        reward = (
            1.0 if next_state == self.gold_state
            else -1.0 if next_state in self.trap_states
            else 0.0
        )

        self.state = next_state

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                ((self.num_cols + 2) * CELL_SIZE,
                 (self.num_rows + 2) * CELL_SIZE)
            )
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.window.fill((255, 255, 255))

        # Grid
        for c in range(self.num_cols + 1):
            pygame.draw.line(
                self.window, (0, 0, 0),
                ((c + 1) * CELL_SIZE, CELL_SIZE),
                ((c + 1) * CELL_SIZE, (self.num_rows + 1) * CELL_SIZE)
            )

        for r in range(self.num_rows + 1):
            pygame.draw.line(
                self.window, (0, 0, 0),
                (CELL_SIZE, (r + 1) * CELL_SIZE),
                ((self.num_cols + 1) * CELL_SIZE, (r + 1) * CELL_SIZE)
            )

        # Traps
        for r, c in self.trap_cells:
            pygame.draw.circle(
                self.window, (50, 50, 50),
                ((c + 1.5) * CELL_SIZE, (r + 1.5) * CELL_SIZE),
                CELL_SIZE // 3
            )

        # Gold
        gr, gc = self.gold_cell
        pygame.draw.polygon(
            self.window, (230, 180, 50),
            [
                ((gc + 1.5) * CELL_SIZE, (gr + 1.2) * CELL_SIZE),
                ((gc + 1.8) * CELL_SIZE, (gr + 1.8) * CELL_SIZE),
                ((gc + 1.2) * CELL_SIZE, (gr + 1.8) * CELL_SIZE),
            ]
        )

        # Agent
        ar, ac = self.state2grid[self.state]
        pygame.draw.rect(
            self.window, (100, 100, 220),
            pygame.Rect(
                (ac + 1.2) * CELL_SIZE,
                (ar + 1.2) * CELL_SIZE,
                CELL_SIZE * 0.6,
                CELL_SIZE * 0.6
            )
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        time.sleep(self.delay)



if __name__ == "__main__":
    env = GridWorldEnv(5, 6, render_mode="human")

    for _ in range(1):
        state, _ = env.reset()
        env.render()

        while True:
            action = env.action_space.sample()

            state, reward, terminated, truncated, info = env.step(action)

            print(
                "Action",
                state,
                action,
                "-> reward:",
                reward,
                "terminated:",
                terminated
            )

            env.render()

            if terminated or truncated:
                break

    env.close()