import torch
import torch.nn as nn
from torch.distributions import Normal

class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Policy head
        self.policy_mean = nn.Linear(hidden_dim, output_dim)
        self.policy_std = nn.Parameter(torch.zeros(output_dim))
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.shared_net(x)
        
        # Policy
        mean = self.policy_mean(features)
        std = torch.exp(self.policy_std).expand_as(mean)
        dist = Normal(mean, std)
        
        # Value
        value = self.value_head(features)
        
        return dist, value