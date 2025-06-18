import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Sim2SimDataset(Dataset):
    """Датасет для обучения корректора"""
    def __init__(self, states, actions, delta_states):
        self.states = states
        self.actions = actions
        self.delta_states = delta_states
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'delta_state': self.delta_states[idx]
        }

class ActionCorrector:
    """Обертка для коррекции действий в реальном времени"""
    def __init__(self, model_path, device='cuda'):
        self.model = torch.load(model_path).to(device)
        self.device = device
        self.model.eval()
        
    def correct(self, state, action):
        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.FloatTensor(action).to(self.device)
        
        with torch.no_grad():
            delta_action = self.model(state_t.unsqueeze(0), action_t.unsqueeze(0))
        
        return action + delta_action.cpu().numpy().squeeze(0)