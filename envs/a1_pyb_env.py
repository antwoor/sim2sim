from gym_wrapper import BaseRobotInterface, gym
import numpy as np
import os, inspect

class PyBulletA1(BaseRobotInterface):
    def __init__(self, render_mode = 'human'):
        import pybullet as pyb
        import pybullet_data as pd
        
        # Инициализация пути
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        os.sys.path.insert(0, parentdir)
        
        # Инициализация PyBullet
        self.pyb = pyb
        self.pd = pd
        
        # Подключение к симулятору (только один раз)
        if not pyb.isConnected():
            self.pyb_client = pyb.connect(pyb.GUI if render_mode == 'human' else pyb.DIRECT)
            pyb.setAdditionalSearchPath(pd.getDataPath())
            pyb.setGravity(0, 0, -9.8)
            pyb.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])
            pyb.setRealTimeSimulation(0)
        else:
            self.pyb_client = pyb.getConnectionInfo()['connectionId']
        
        # Импорт и инициализация робота
        from motion_imitation.robots import a1_original as a1_pyb
        from motion_imitation.robots import robot_config
        self.robot = a1_pyb.A1(pybullet_client=self.pyb,
                               motor_control_mode=robot_config.MotorControlMode.POSITION)
        self.robot.Reset(reload_urdf=False)
        
    def reset(self):
        self.robot.Reset(reload_urdf=False)

    def get_observation(self):
        self.robot.ReceiveObservation()
        return np.concatenate([self.robot.GetTrueObservation(), self.robot.GetFootContacts()])
    
    def apply_action(self, action):
        # Конвертируем action в numpy array если это необходимо
        if isinstance(action, tuple):
            action = action[0]  # Берем только массив действий, игнорируем log_prob
        
        # Преобразуем в numpy array и выравниваем
        action = np.asarray(action, dtype=np.float32).flatten()
        self.robot.ApplyAction(action)

    def step(self, action, control_mode=None):
        # Извлекаем только массив действий, если передается кортеж
        if isinstance(action, tuple):
            action = action[0]  # Берем только массив действий, игнорируем log_prob

        # Преобразуем в numpy array и выравниваем
        action = np.asarray(action, dtype=np.float32).flatten()

        # Проверка размера действия
        if action.shape != (12,):
            raise ValueError(f"Action must have shape (12,), got {action.shape}")

        # Применяем действие
        self.robot.Step(action, control_mode)

        # Получаем новое состояние
        return self.get_observation()

    def get_reward_components(self):
        return {
            'v_x': self.robot.GetBaseVelocity()[0],
            'y': self.robot.GetBasePosition()[2],
            'roll': self.robot.GetBaseRollPitchYaw()[0],
            'pitch': self.robot.GetBaseRollPitchYaw()[1],
            'yaw': self.robot.GetBaseRollPitchYaw()[2],
            'contacts': self.robot.GetFootContacts(),
            'joint_torques': self.robot.GetMotorTorques()
        }
    
    def is_done(self):
        return (self.robot.GetBasePosition()[2] < 0.18 or 
                np.sum(self.robot.GetBaseRollPitchYaw()**2) >= 0.73)
    
    def get_action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)

    def get_observation_space(self):
        obs_size = len(self.robot.GetTrueObservation()) + len(self.robot.GetFootContacts())
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def close(self):
        self.pyb.disconnect()

# Пример использования
if __name__ == '__main__':
    from gym_wrapper import RobotEnvWrapper
    # Создаем конкретного робота
    a1_robot = PyBulletA1(render_mode='human')
    
    # Создаем обертку среды
    env = RobotEnvWrapper(a1_robot, max_episode_steps=2000)
    
    # Теперь можно использовать env с любым RL-алгоритмом
    from ppo.ppo_agent import PPOAgent
    agent = PPOAgent(env, lr=3e-4, hidden_dim=128, epochs=200, batch_size=100)
    
    # Запуск обучения
    rewards = []
    for episode in range(2000):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    env.close()