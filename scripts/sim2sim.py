import sys
sys.path.append("/home/root/rl_ws/ppo_sim2sim")  # Если пути нет в sys.path
from ACN import *
from ACN.utils import *
from envs.gym_wrapper import *
from ppo.ppo_agent import *
from envs import a1_pyb_env, a1_mj_env

# 1. Инициализация сред
pb_env = RobotEnvWrapper(a1_pyb_env.PyBulletA1())
print("PyBullet environment has been initialized")
mj_env = RobotEnvWrapper(a1_mj_env.MujocoA1())
print("mujoco environment has been initialized")

# 2. Инициализация PPO агента
agent = PPOAgent(pb_env)
print("PPO agent has been initialized")
# 3. Первоначальный сбор данных
print("\nstarting collecting si2sim data")
sim_data = collect_sim2sim_data(pb_env, mj_env, agent, num_episodes=20)
print("\ndata has been collected, \nstarting preparing dataset")
dataset = prepare_dataset(sim_data)
print("dataset has been created")
# 4. Обучение корректора
print("starting training ACN")
corrector = train_corrector(dataset)
print("ACN has been trained")
# 5. Обучение с корректором
print("starting training PPO with corrector")
rewards = train_ppo_with_corrector(pb_env, mj_env, agent, corrector)
