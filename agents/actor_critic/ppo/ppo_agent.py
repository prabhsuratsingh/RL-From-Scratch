import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, nA):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(128, nA),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.shared(x)

        probs = self.policy_head(x)
        value = self.value_head(x)

        return probs, value.squeeze(-1)
    
class PPOAgent:
    def __init__(
            self,
            env,
            gamma=0.99,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
            lr=3e-4
    ):
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        state_dim = env.observation_space.shape[0]
        nA = env.action_space.n

        self.model = ActorCritic(state_dim, nA)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

    def choose_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)

        probs, value = self.model(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action).detach(),
            value.detach(),
            dist.entropy().detach()
        )
    
    def compute_returns(self, rewards, dones, values):
        returns = []
        G = 0

        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0
            
            G = r + self.gamma * G

            returns.insert(0, G)
        
        return torch.tensor(returns)
    
    def update(
            self,
            states,
            actions,
            old_log_probs,
            returns,
            advantages
    ):
        probs, values = self.model(states)

        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.clip_eps,
            1 + self.clip_eps
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - values).pow(2).mean()
        entropy_loss = -dist.entropy().mean()

        loss = (
            policy_loss
            + self.value_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()