import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNet(nn.Module):
    def __init__(self, state_dim, nA):
        super(PolicyNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, nA),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):

        return self.net(x)


class ReinforceAgent:
    def __init__(
            self,
            env,
            lr=1e-3,
            discount_factor=0.99
    ):
        self.env = env
        self.gamma = discount_factor

        self.state_dim = env.observation_space.shape[0]
        self.nA = env.action_space.n

        self.policy = PolicyNet(self.state_dim, self.nA)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state_t)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def _compute_returns(self):
        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0,G)
        
        return torch.tensor(returns, dtype=torch.float32)

    def update(self):
        returns = self._compute_returns()

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss += -log_prob * G
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()

        return loss.item()