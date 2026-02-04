from tkinter import NO
from turtle import st
from matplotlib.pylab import rand
import numpy as np
from collections import defaultdict


class DoubleTDAgent:
    """
    Double TD Agent

    Implemented Algorithms :-
    - **Double Q-Learning**
    - **Double SARSA**
    - **Double Expected SARSA**
    """
    def __init__(
            self,
            env,
            algorithm="double_q_learning",
            learning_rate=0.01,
            discount_factor=0.9,
            epsilon_greedy=0.9,
            epsilon_min=0.1,
            epsilon_decay=0.95
    ):
        self.env = env
        self.algorithm = algorithm

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.nA = env.action_space.n

        self.q1 = defaultdict(lambda: np.zeros(self.nA))
        self.q2 = defaultdict(lambda: np.zeros(self.nA))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        
        q_vals = self.q1[state] + self.q2[state]

        return np.random.choice(
            np.flatnonzero(q_vals == q_vals.max())
        )
    
    def _bootstrap(self, next_state, next_action, terminated, truncated, use_q1):
        done = terminated or truncated
        if done:
            return 0.0
        
        q_select = self.q1 if use_q1 else self.q2
        q_eval = self.q2 if use_q1 else self.q1

        if self.algorithm == 'double_q_learning':
            a_star = np.argmax(q_select[next_state])
            return self.gamma * q_eval[next_state][a_star]
        elif self.algorithm == "double_sarsa":
            return self.gamma * q_eval[next_state][next_action]
        elif self.algorithm == "double_expected_sarsa":
            q_vals = q_select[next_state]
            policy = np.ones(self.nA) * self.epsilon / self.nA
            best = np.argmax(q_vals)
            policy[best] += 1.0 - self.epsilon

            return self.gamma * np.dot(policy * q_vals)
        else:
            raise ValueError(f"Unknown double learning algorithm : {self.algorithm}")
        
    def learn(self, transition):
        if self.algorithm == "double_sarsa":
            s, a, r, next_s, next_a, terminated, truncated = transition
        else:
            s, a, r, next_s, terminated, truncated = transition
            next_a = None

        use_q1 = np.random,rand() < 0.5

        bootstrap = self._bootstrap(
            next_s, next_a, terminated, truncated, use_q1
        )

        Q = self.q1 if use_q1 else self.q2
        Q[s][a] += self.lr * (r - bootstrap - Q[s][a])

        self._adjust_epsilon()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
                    