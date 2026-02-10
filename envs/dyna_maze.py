import gymnasium as gym
from gymnasium import spaces
import pygame
import time

CELL_SIZE = 80


class DynaMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.rows = 6
        self.cols = 9
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        self._init_states()
        self._init_actions()
        self._init_maze()

        self.state = self.start_state

        self.window = None
        self.clock = None

    def _init_states(self):
        self.grid2state = {
            (r, c): r * self.cols + c
            for r in range(self.rows)
            for c in range(self.cols)
        }
        self.state2grid = {v: k for k, v in self.grid2state.items()}

    def _init_actions(self):
        self.actions = {
            0: lambda r, c: (max(r - 1, 0), c),          
            1: lambda r, c: (r, min(c + 1, self.cols - 1)),  
            2: lambda r, c: (min(r + 1, self.rows - 1), c),  
            3: lambda r, c: (r, max(c - 1, 0)),         
        }

    def _init_maze(self):
        self.start_cell = (5, 0)
        self.goal_cell = (0, 8)

        self.walls = {
            (1, 2), (2, 2), (3, 2), (4, 2),
            (0, 5), (1, 5), (2, 5), (3, 5),
        }

        self.start_state = self.grid2state[self.start_cell]
        self.goal_state = self.grid2state[self.goal_cell]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def step(self, action):
        r, c = self.state2grid[self.state]
        nr, nc = self.actions[action](r, c)

        if (nr, nc) in self.walls:
            nr, nc = r, c

        next_state = self.grid2state[(nr, nc)]
        terminated = next_state == self.goal_state

        reward = 0.0 if terminated else -1.0

        self.state = next_state

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                ((self.cols + 2) * CELL_SIZE,
                 (self.rows + 2) * CELL_SIZE)
            )
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.window.fill((255, 255, 255))

        for c in range(self.cols + 1):
            pygame.draw.line(
                self.window, (0, 0, 0),
                ((c + 1) * CELL_SIZE, CELL_SIZE),
                ((c + 1) * CELL_SIZE, (self.rows + 1) * CELL_SIZE)
            )
        for r in range(self.rows + 1):
            pygame.draw.line(
                self.window, (0, 0, 0),
                (CELL_SIZE, (r + 1) * CELL_SIZE),
                ((self.cols + 1) * CELL_SIZE, (r + 1) * CELL_SIZE)
            )

        for r, c in self.walls:
            pygame.draw.rect(
                self.window, (100, 100, 100),
                pygame.Rect(
                    (c + 1) * CELL_SIZE,
                    (r + 1) * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
            )

        gr, gc = self.goal_cell
        pygame.draw.rect(
            self.window, (50, 200, 50),
            pygame.Rect(
                (gc + 1) * CELL_SIZE,
                (gr + 1) * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
        )

        ar, ac = self.state2grid[self.state]
        pygame.draw.circle(
            self.window, (50, 50, 200),
            ((ac + 1.5) * CELL_SIZE, (ar + 1.5) * CELL_SIZE),
            CELL_SIZE // 3
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        time.sleep(0.05)

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
