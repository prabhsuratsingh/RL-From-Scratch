from collections import defaultdict, deque
import numpy as np

class NStepTDAgent:
    """
    n-Step TD Agent

    Implemented Algorithms :-
    - **Q-Learning**
    - **SARSA**
    - **Expected SARSA**
    """
    def __init__(
        self,
        env,
        algorithm="q_learning",
        n_step=3,
        learning_rate=0.01,
        discount_factor=0.9,
        epsilon_greedy=0.9,
        epsilon_min=0.1,
        epsilon_decay=0.95,
    ):
        self.env = env
        self.algorithm = algorithm
        self.n_step = n_step

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.nA = env.action_space.n
        self.q_table = defaultdict(lambda: np.zeros(self.nA))

        self.buffer = deque()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        q_vals = self.q_table[state]
        return np.random.choice(
            np.flatnonzero(q_vals == q_vals.max())
        )

    def _compute_return(
        self,
        next_state,
        next_action,
        terminated
    ):
        G = 0.0

        for i, (_, _, r) in enumerate(self.buffer):
            G += (self.gamma ** i) * r

        if not terminated and len(self.buffer) == self.n_step:
            if self.algorithm == "q_learning":
                G += (self.gamma ** self.n_step) * np.max(self.q_table[next_state])

            elif self.algorithm == "sarsa":
                G += (self.gamma ** self.n_step) * self.q_table[next_state][next_action]

            elif self.algorithm == "expected_sarsa":
                policy = np.ones(self.nA) * self.epsilon / self.nA
                best_a = np.argmax(self.q_table[next_state])

                policy[best_a] += 1.0 - self.epsilon
                G += (self.gamma ** self.n_step) * np.dot(
                    policy, self.q_table[next_state]
                )

            else:
                raise ValueError(f"Unknown algorithm : {self.algorithm}")
        
        return G

    def learn(self, transition):
        s, a, r, next_s, next_a, terminated, truncated = transition
        done = terminated or truncated

        self.buffer.append((s, a, r))

        if len(self.buffer) == self.n_step or done:
            s0, a0, _ = self.buffer[0]
            G = self._compute_return(next_s, next_a, terminated)

            self.q_table[s0][a0] += self.lr * (G - self.q_table[s0][a0])
            self.buffer.popleft()
        
        if done:
            self._flush_buffer()

        self._adjust_epsilon()

    def _flush_buffer(self):
        while self.buffer:
            s0, a0, _ = self.buffer[0]
            G = 0.0
            for i, (_, _, r) in enumerate(self.buffer):
                G += (self.gamma ** i) * r

            self.q_table[s0][a0] += self.lr * (G - self.q_table[s0][a0])
            self.buffer.popleft()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay