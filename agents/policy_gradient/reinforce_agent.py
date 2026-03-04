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

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        return self.net(x).squeeze(-1)
        

class ReinforceAgent:
    def __init__(
            self,
            env,
            lr=1e-3,
            discount_factor=0.99,
            use_baseline=False,
            use_entropy=False,
            entropy_coeff=0.01
    ):
        self.env = env
        self.gamma = discount_factor

        self.state_dim = env.observation_space.shape[0]
        self.nA = env.action_space.n

        self.use_baseline = use_baseline
        self.use_entropy = use_entropy
        self.entropy_coeff = entropy_coeff

        self.policy = PolicyNet(self.state_dim, self.nA)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        if self.use_baseline:
            self.value_net = ValueNet(self.state_dim)
            self.value_optim = optim.Adam(self.value_net.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []
        self.states = []
        self.entropies = []

    def choose_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)

        probs = self.policy(state_t)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.states.append(state_t)
        self.entropies.append(dist.entropy())

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

        states = torch.stack(self.states)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        if self.use_baseline:
            values = self.value_net(states)
            advantages = returns - values.detach()

            value_loss = nn.MSELoss()(values, returns)

            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
        else:
            advantages = returns

        policy_loss = -(log_probs * advantages).mean()

        if self.use_entropy:
            entropy_loss = -self.entropy_coeff * entropies.mean()
            policy_loss += entropy_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.log_probs.clear()
        self.rewards.clear()
        self.states.clear()
        self.entropies.clear()

        return policy_loss.item()