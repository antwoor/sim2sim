import os
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from abc import ABC, abstractmethod

class BaseRobotInterface(ABC):
    """Абстрактный класс интерфейса робота"""
    
    @abstractmethod
    def reset(self):
        """Сброс состояния робота"""
        pass
    
    @abstractmethod
    def get_observation(self):
        """Получение наблюдения"""
        pass
    
    @abstractmethod
    def apply_action(self, action):
        """Применение действия"""
        pass
    
    @abstractmethod
    def get_reward_components(self):
        """Получение компонентов для расчета награды"""
        pass
    
    @abstractmethod
    def is_done(self):
        """Проверка завершения эпизода"""
        pass

    @abstractmethod
    def step(self, action, control_mode =None):
        """
        Cтепает симуляцию, принимает на вход вектор действия. 
        Это действие затем обрабатывается.
        Вызывает под капотом ApplyAction и ReceiveObservation
        """
        pass

class RobotEnvWrapper(gym.Env):
    """Универсальная обертка для разных роботов и симуляторов"""
    
    def __init__(self, robot: BaseRobotInterface, max_episode_steps=1000):
        """
        Args:
            robot: Объект робота, реализующий BaseRobotInterface
            max_episode_steps: Максимальное количество шагов в эпизоде
        """
        super().__init__()
        self.robot = robot
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Определение пространства действий и состояний
        self.action_space = robot.get_action_space()
        self.observation_space = robot.get_observation_space()
        
        # Инициализация TensorBoard
        self.writer = SummaryWriter()
        self.episode_reward = 0
        self.episode_count = 0

    def reset(self):
        """Сброс среды"""
        self.robot.reset()
        self.current_step = 0
        self.episode_reward = 0
        return self.robot.get_observation()

    def step(self, action):
        """Выполнение одного шага"""
        # Применяем действие
        self.robot.step(action)
        
        # Получаем следующее состояние
        next_state = self.robot.get_observation()
        
        # Получаем компоненты для расчета награды
        reward_components = self.robot.get_reward_components()
        
        # Вычисляем награду
        reward = self.calculate_reward(**reward_components)
        self.episode_reward += reward
        
        # Проверяем завершение эпизода
        done = self.robot.is_done() or (self.current_step >= self.max_episode_steps)
        self.current_step += 1
        
        # Логируем награду в TensorBoard
        if done:
            self.writer.add_scalar('Reward/Episode', self.episode_reward, self.episode_count)
            self.episode_count += 1
        
        return next_state, reward, done, {}

    def calculate_reward(self, v_x, y, roll, pitch, yaw, contacts, joint_torques=None, fall=False, **kwargs):
        """
        Улучшенная функция вознаграждения для шагающего робота
        """
        target_height = 0.25  # Целевая высота центра масс
        
        # Базовые компоненты
        velocity_reward = 25.0 * np.clip(v_x, -2.0, 2.0)
        height_penalty = -80.0 * ((y - target_height) ** 2)

        # Стабильность и ориентация
        orientation_penalty = -15.0 * (roll**2 + pitch**2) - 5.0 * yaw**2
        
        # Контакты с поверхностью
        contact_bonus = 2.0 * np.sum(contacts)
        flight_penalty = -50.0 if np.sum(contacts) == 0 else 0.0

        # Энергоэффективность
        energy_penalty = -0.01 * np.sum(np.square(joint_torques)) if joint_torques is not None else 0.0
        
        # Временные компоненты
        time_progress = 25.0 * (self.current_step / self.max_episode_steps)
        survival_bonus = 10.0

        # Терминальные условия
        termination_penalty = -700.0 if fall else 0.0  # Большой штраф за падение
        
        total_reward = (
            velocity_reward
            + height_penalty
            + orientation_penalty
            + contact_bonus
            + flight_penalty
            + energy_penalty
            + time_progress
            + survival_bonus
            + termination_penalty
        )

        return total_reward

    def get_metrics(self):
        """Возвращает словарь с текущими метриками"""
        return self.robot.get_metrics()
    
    def render(self, mode='human'):
        """Визуализация"""
        pass

    def close(self):
        """Завершение работы"""
        self.writer.close()
        self.robot.close()
