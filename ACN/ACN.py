import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .model import *
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
    def __init__(self, model_path, state_dim, action_dim, device='cuda'):
        self.device = device
        
        # Загружаем сохраненные веса
        state_dict = torch.load(model_path, map_location=device)
        
        # Создаем экземпляр модели с правильными размерами
        self.model = ActionCorrectionModel(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(device)
        
        # Загружаем веса
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
    def correct(self, state, action):
        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.FloatTensor(action).to(self.device)
        
        with torch.no_grad():
            # Модель возвращает кортеж (delta_action, jacobian)
            delta_action, _ = self.model(state_t.unsqueeze(0), action_t.unsqueeze(0))
        
        return action + delta_action.cpu().numpy().squeeze(0)