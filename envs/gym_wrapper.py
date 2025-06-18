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
        Улучшенная функция вознаграждения для шагающего робота.
        Основные изменения:
        - Сбалансированные коэффициенты
        - Динамическая компенсация доминирующих компонент
        - Плавные штрафы за ориентацию
        - Адаптивное поощрение скорости
        - Физически обоснованные ограничения
        """
        # Целевые параметры
        target_velocity = 1.5  # м/с
        target_height = 0.25   # м
        
        # 1. Вознаграждение за скорость (логарифмическое насыщение)
        velocity_error = min(abs(v_x - target_velocity), 2.0)
        velocity_reward = 8.0 * np.exp(-0.5 * velocity_error)
        
        # 2. Штраф за высоту (кубическая функция для мягких границ)
        height_error = abs(y - target_height)
        height_penalty = -40.0 * (height_error + 0.2 * height_error**3)
        
        # 3. Штраф за ориентацию (с разделением осей и порогом чувствительности)
        orientation_penalty = (
            -12.0 * (roll**2 + pitch**2) 
            - 3.0 * np.where(abs(yaw) > 0.3, yaw**2, 0)
        )
        
        # 4. Контактный баланс (поощрение правильных фаз)
        contact_bonus = 1.2 * np.mean(contacts)
        flight_penalty = -10.0 * np.where(np.sum(contacts) == 0, 1.0, 0.0)
        
        # 5. Энергоэффективность (нормировка на количество суставов)
        energy_penalty = 0
        if joint_torques is not None:
            torque_norm = np.mean(np.square(joint_torques))
            energy_penalty = -0.025 * torque_norm
        
        # 6. Прогресс эпизода (линейное поощрение выживания)
        survival_bonus = 20.0 * (self.current_step / self.max_episode_steps)
        
        # 7. Терминальные условия (компенсированный штраф)
        termination_penalty = -500.0 if fall else 0.0
        
        # Динамическое масштабирование (предотвращает доминирование компонент)
        total_reward = (
            0.3 * velocity_reward
            + 0.2 * height_penalty
            + 0.25 * orientation_penalty
            + 0.1 * (contact_bonus + flight_penalty)
            + 0.05 * energy_penalty
            + 0.1 * survival_bonus
            + termination_penalty
        )
        
        return np.clip(total_reward, -10.0, 15.0)

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
