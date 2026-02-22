from collections import deque
import random
import torch
import torch.nn as nn
import numpy as np

class DQNAgent:
    def __init__(
        self,
        env,
        discount_factor=0.95,
        epsilon_greedy=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=1e-3,
        max_memory_size=50000,
        target_update_freq=200,
    ) :
        
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.nA = env.action_space.n

        self.memory = deque(maxlen=max_memory_size)

        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.lr = learning_rate

        self.target_update_freq = target_update_freq
        self.learn_steps = 0

        self._build_nn_model()

    def _build_nn_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.nA),
        )

        self.target_model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.nA),
        )

        self.target_model.load_state_dict(self.model.state_dict())

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr
        )
    
    def remember(self, transition):
        self.memory.append(transition)
    
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()

        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_t)
        
        return torch.argmax(q_values).item()
    
    def _learn(self, batch_samples):
        batch_states, batch_targets = [], []

        for transition in batch_samples:
            s, a, r, next_s, done = transition

            with torch.no_grad():
                if done: 
                    target = r
                else:
                    pred = self.model(torch.tensor(next_s, dtype=torch.float32))[0]
                    target = r + self.gamma * pred.max()
            
            target_all = self.model(torch.tensor(s, dtype=torch.float32))[0]
            target_all[a] = target

            batch_states.append(s.flatten())
            batch_targets.append(target_all)
            self._adjust_epsilon()
            self.optimizer.zero_grad()

            pred = self.model(torch.tensor(batch_states, dtype=torch.float32))
            loss = self.loss_fn(pred, torch.stack(batch_targets))
            loss.backward()

            self.optimizer.step()
        
        return loss.item()
    
    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0

        batch = random.sample(self.memory, batch_size)

        states = torch.tensor(np.vstack([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch])
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.tensor(np.vstack([t[3] for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self._adjust_epsilon()

        return loss.item()