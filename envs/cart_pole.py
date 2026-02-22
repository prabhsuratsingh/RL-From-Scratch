import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import math

class CartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 0.5
        self.total_mass = self.mass_cart + self.mass_pole
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0

        self.x_threshold = 2.4
        self.theta_threshold = 12 * math.pi / 180

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.state = None

        self.window = None
        self.clock = None
        self.tau = 0.02

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0

        if self.render_mode == "human":
            self.render()

        return self.state.astype(np.float32), {}
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass

        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )

        reward = 1.0  
        self.steps += 1

        truncated = False

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {}
    
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((600, 400))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.window.fill((255, 255, 255))

        x, _, theta, _ = self.state

        cart_y = 300
        scale = 100
        cart_x = 300 + x * scale

        cart_w, cart_h = 60, 30
        pygame.draw.rect(
            self.window,
            (0, 0, 0),
            pygame.Rect(
                int(cart_x - cart_w / 2),
                int(cart_y - cart_h / 2),
                int(cart_w),
                int(cart_h),
            ),
        )

        pole_len = scale * 2 * self.length
        pole_x = cart_x + pole_len * math.sin(theta)
        pole_y = cart_y - pole_len * math.cos(theta)

        pygame.draw.line(
            self.window,
            (200, 0, 0),
            (int(cart_x), int(cart_y)),
            (int(pole_x), int(pole_y)),
            6,
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None