import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ActionCorrectionModel(nn.Module):
    """Нейронная сеть для коррекции действий"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Дополнительная сеть для аппроксимации якобиана (влияния действий на состояния)
        self.jacobian_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim * action_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        delta_action = self.action_net(x)
        
        # Аппроксимируем якобиан: ∂state/∂action
        jacobian = self.jacobian_net(x).view(-1, state.shape[1], action.shape[1])
        return delta_action, jacobian