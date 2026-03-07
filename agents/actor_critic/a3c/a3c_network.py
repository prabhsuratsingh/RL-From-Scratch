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
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value.unsqueeze(-1)