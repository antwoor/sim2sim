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
        Сбалансированная функция вознаграждения с гарантией положительных значений
        при удовлетворительном поведении. Ключевые изменения:
        - Уменьшены доминирующие штрафы
        - Введены достижимые бонусы
        - Добавлены пороговые условия
        - Нормализованы масштабы компонент
        """
        # Целевые параметры (настраиваемые)
        target_velocity = 1.5  # м/с
        target_height = 0.25   # м
        
        # 1. Вознаграждение за скорость (основной драйвер)
        velocity_reward = 8.0 * np.exp(-0.8 * abs(v_x - target_velocity))
        
        # 2. Бонус за высоту (вместо штрафа!)
        height_error = abs(y - target_height)
        height_reward = 4.0 * np.exp(-15.0 * height_error)
        
        # 3. Ориентация (с пороговой функцией)
        orientation_error = roll**2 + pitch**2
        orientation_reward = (
            3.0 * np.exp(-8.0 * orientation_error) 
            - 5.0 * (abs(roll) > 0.6) 
            - 5.0 * (abs(pitch) > 0.6)
        )
        
        # 4. Контакты (баланс стабильности и движения)
        contact_reward = 1.5 * np.mean(contacts)
        
        # 5. Энергоэффективность (незначительный штраф)
        energy_penalty = 0
        if joint_torques is not None:
            energy_penalty = -0.002 * np.mean(np.square(joint_torques))
        
        # 6. Прогресс эпизода (поощрение выживания)
        progress_reward = 2.0 * (self.current_step / self.max_episode_steps)
        
        # 7. Терминальные условия
        if fall:
            # Динамический штраф в зависимости от времени выживания
            termination_penalty = -50.0 - 30.0 * (self.current_step / self.max_episode_steps)
        else:
            termination_penalty = 0.0
        
        # Собираем общую награду (все компоненты сравнимого масштаба)
        total_reward = (
            velocity_reward
            + height_reward
            + orientation_reward
            + contact_reward
            + energy_penalty
            + progress_reward
            + termination_penalty
        )
        
        # Гарантируем минимальное вознаграждение
        baseline_bonus = 1.5  # Базовый бонус за "не падение"
        return total_reward + baseline_bonus

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
