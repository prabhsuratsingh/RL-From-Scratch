from collections import defaultdict
import numpy as np
import random

class DynaQAgent:
    """
    Dyna-Q

    """
    def __init__(
        self,
        env,
        learning_rate=0.01,
        discount_factor=0.95,
        epsilon=0.1,
        epsilon_min=0.1,
        epsilon_decay=0.99,
        planning_steps=10
    ):
        self.env = env
        self.nA = env.action_space.n

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.planning_steps = planning_steps

        self.q_table = defaultdict(lambda: np.zeros(self.nA))

        self.model = {}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        q_vals = self.q_table[state]
        return np.random.choice(
            np.flatnonzero(q_vals == q_vals.max())
        )

    def learn(self, transition):
        s, a, r, next_s, _, terminated, truncated = transition
        done = terminated or truncated

        target = r if done else r + self.gamma * np.max(self.q_table[next_s])
        self.q_table[s][a] += self.lr * (target - self.q_table[s][a])

        self.model[(s,a)] = (next_s, r)

        self._planning()

        self._adjust_epsilon()

    def _planning(self):
        if not self.model:
            return
        
        for _ in range(self.planning_steps):
            (s,a) = random.choice(list(self.model.keys()))
            next_s, r = self.model[(s,a)]

            target = r + self.gamma * np.max(self.q_table[next_s])
            self.q_table[s][a] += self.lr * (target - self.q_table[s][a])

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay