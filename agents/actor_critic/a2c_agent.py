import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim, nA):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, nA),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
class A2CAgent:
    def __init__(
            self,
            env,
            lr=1e-3,
            gamma=0.99,
            value_coeff=0.5,
            entropy_coeff=0.01
    ):
        self.env = env
        self.gamma = gamma
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        state_dim = env.observation_space.shape[0]
        nA = env.action_space.n

        self.actor = Actor(state_dim, nA)
        self.critic = Critic(state_dim)

        self.optim = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

    def choose_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)

        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy()

    def learn(
            self,
            state,
            action_log_prob,
            entropy,
            reward,
            next_state,
            done
    ) :
        state_t = torch.tensor(state, dtype=torch.float32)
        next_state_t = torch.tensor(next_state, dtype=torch.float32)

        value = self.critic(state_t)
        next_value = self.critic(next_state_t)

        target = reward + self.gamma * next_value * (1 - done)
        advantage = target - value

        policy_loss = -(action_log_prob * advantage.detach())
        value_loss = advantage.pow(2)
        entropy_loss = -self.entropy_coeff * entropy

        loss = policy_loss + self.value_coeff * value_loss + entropy_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()