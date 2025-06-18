import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal

class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Policy head
        self.policy_mean = nn.Linear(hidden_dim, output_dim)
        self.policy_logstd = nn.Parameter(torch.full((output_dim,), -0.5))
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module is self.policy_mean:
                nn.init.orthogonal_(module.weight, gain=0.01)
            else:
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        features = self.shared_net(x)
        
        # Policy
        mean = torch.tanh(self.policy_mean(features))  # Tanh для ограничения действий
        logstd = self.policy_logstd.expand_as(mean)
        dist = Normal(mean, torch.exp(logstd))
        
        # Value
        value = self.value_head(features)
        
        return dist, value.squeeze(-1)