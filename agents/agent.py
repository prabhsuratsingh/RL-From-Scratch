from collections import defaultdict
import numpy as np

class Agent:
    def __init__(
        self,
        env,
        algorithm="q_learning",
        learning_rate=0.01,
        discount_factor=0.9,
        epsilon_greedy=0.9,
        epsilon_min=0.1,
        epsilon_decay=0.95,
    ):
        self.env = env
        self.algorithm = algorithm

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.nA = env.action_space.n
        self.q_table = defaultdict(lambda: np.zeros(self.nA))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        q_vals = self.q_table[state]
        return np.random.choice(
            np.flatnonzero(q_vals == q_vals.max())
        )

    def _compute_target(
        self,
        reward,
        next_state,
        terminated,
        truncated,
        next_action=None
    ):
        done = terminated or truncated

        if done:
            return reward

        if self.algorithm == "q_learning":
            return reward + self.gamma * np.max(self.q_table[next_state])

        elif self.algorithm == "sarsa":
            return reward + self.gamma * self.q_table[next_state][next_action]

        elif self.algorithm == "expected_sarsa":
            policy = np.ones(self.nA) * self.epsilon / self.nA
            # best_a = np.argmax(self.q_table[next_state])
            q_vals = self.q_table[next_state]
            best_actions = np.flatnonzero(q_vals == q_vals.max())
            best_a = np.random.choice(best_actions)

            policy[best_a] += 1.0 - self.epsilon
            return reward + self.gamma * np.dot(
                policy, self.q_table[next_state]
            )

        else:
            raise ValueError(f"Unknown algorithm {self.algorithm}")

    def learn(self, transition):
        if self.algorithm == "sarsa":
            s, a, r, next_s, next_a, terminated, truncated = transition
        else:
            s, a, r, next_s, terminated, truncated = transition
            next_a = None

        q_val = self.q_table[s][a]

        q_target = self._compute_target(
            r, next_s, terminated, truncated, next_action=next_a
        )

        self.q_table[s][a] += self.lr * (q_target - q_val)
        self._adjust_epsilon()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay