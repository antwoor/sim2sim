import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
from .model import PPONetwork

class PPOAgent:
    def __init__(self, env, gamma=0.99, gae_lambda=0.95, lr=3e-4, clip_epsilon=0.2, 
                 epochs=10, batch_size=8192, mini_batch_size=512, hidden_dim=256, 
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, load_path=None):
        # Гиперпараметры
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # Инициализация сетей
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.policy = PPONetwork(obs_dim, act_dim, hidden_dim)
        self.old_policy = PPONetwork(obs_dim, act_dim, hidden_dim)
        
        if load_path and os.path.exists(load_path):
            self.load_model(load_path)
        else:
            self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: max(1.0 - step/5e6, 0.1)
        )
        
        # Буферы данных
        self.reset_buffer()
        
        # Статистика
        self.obs_mean = torch.zeros(obs_dim)
        self.obs_var = torch.ones(obs_dim)
        self.obs_count = 1e-4
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = 1e-4

    def reset_buffer(self):
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint)
        self.old_policy.load_state_dict(checkpoint)
        print(f"Loaded model weights from {path}")

    def act(self, state):
        # Нормализация состояния
        state_tensor = torch.FloatTensor(self.normalize_obs(state)).unsqueeze(0)
        
        with torch.no_grad():
            dist, value = self.old_policy(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)
            
        return action.squeeze(0).numpy(), log_prob.item(), value.item()

    def normalize_obs(self, obs):
        obs = torch.FloatTensor(obs)
        self.obs_mean = self.obs_mean * self.obs_count + obs
        self.obs_count += 1
        self.obs_mean /= self.obs_count
        
        self.obs_var = self.obs_var * (self.obs_count - 1) + (obs - self.obs_mean)**2
        self.obs_var /= self.obs_count
        
        return (obs - self.obs_mean) / (torch.sqrt(self.obs_var) + 1e-8)

    def store_transition(self, state, action, reward, done, log_prob, value):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)

    def compute_gae(self, last_value=0):
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'] + [last_value])
        dones = np.array(self.buffer['dones'])
        
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        # Вычисление GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        returns = advantages + values[:-1]
        
        # Нормализация возвратов
        returns = torch.FloatTensor(returns)
        self.ret_mean = self.ret_mean * self.ret_count + returns.mean()
        self.ret_count += len(returns)
        self.ret_mean /= self.ret_count
        
        self.ret_var = self.ret_var * (self.ret_count - len(returns)) + ((returns - self.ret_mean)**2).sum()
        self.ret_var /= self.ret_count
        
        returns = (returns - self.ret_mean) / (torch.sqrt(self.ret_var) + 1e-8)
        return advantages, returns

    def update(self, last_value=0):
        # Преобразование данных в тензоры
        states = torch.FloatTensor(np.array(self.buffer['states']))
        actions = torch.FloatTensor(np.array(self.buffer['actions']))
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).unsqueeze(1)
        
        # Вычисление GAE и возвратов
        advantages, returns = self.compute_gae(last_value)
        
        # Преобразуем advantages в тензор перед нормализацией
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Нормализация advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Оптимизация политики
        for _ in range(self.epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Forward pass
                dist, values = self.policy(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=1, keepdim=True)
                entropy = dist.entropy().mean()
                
                # Отношение вероятностей
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                # Policy loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Обновление старой политики
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.reset_buffer()
        self.lr_scheduler.step()

    def train(self, env, episodes=5000, max_steps=1000, update_freq=8192, 
              save_interval=200, log_dir="runs/ppo_a1"):
        # Подготовка директорий
        os.makedirs("a1_weights", exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        
        episode_rewards = []
        best_mean_reward = -np.inf
        global_step = 0

        for episode in range(1, episodes + 1):
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            # Статистика за эпизод
            ep_metrics = {
                'velocity': 0,
                'height': 0,
                'orientation': 0,
                'contacts': 0,
                'energy': 0,
                'survival': 0
            }

            while not done and step_count < max_steps:
                # Выбор действия
                action, log_prob, value = self.act(state)
                
                # Шаг в среде
                next_state, reward, done, info = env.step(action)
                
                # Сохранение перехода
                self.store_transition(state, action, reward, done, log_prob, value)
                
                # Обновление статистики
                episode_reward += reward
                step_count += 1
                global_step += 1
                state = next_state
                
                # Обновление политики
                if len(self.buffer['states']) >= update_freq:
                    _, last_value = self.old_policy(torch.FloatTensor(
                        self.normalize_obs(next_state)).unsqueeze(0))
                    self.update(last_value.item())
            
            # Сбор метрик
            if hasattr(env, 'get_metrics'):
                metrics = env.get_metrics()
                for k in ep_metrics:
                    ep_metrics[k] = metrics.get(k, 0)
            
            # Сохранение результатов
            episode_rewards.append(episode_reward)
            current_mean = np.mean(episode_rewards[-10:])
            
            # Логирование
            writer.add_scalar('Reward/Episode', episode_reward, episode)
            writer.add_scalar('Reward/Average_10', current_mean, episode)
            for metric, value in ep_metrics.items():
                writer.add_scalar(f'Metrics/{metric}', value, episode)
            writer.add_scalar('LearningRate', self.lr_scheduler.get_last_lr()[0], episode)
            writer.add_scalar('Metrics/Survival', step_count, episode)

            
            # Сохранение модели
            if episode % save_interval == 0 or current_mean > best_mean_reward:
                if current_mean > best_mean_reward:
                    best_mean_reward = current_mean
                    torch.save(self.policy.state_dict(), "a1_weights/best_ppo.pth")
                torch.save(self.policy.state_dict(), f"a1_weights/ppo_{episode}.pth")
            
            # Вывод прогресса
            if episode % 10 == 0:
                print(f"Episode {episode:5d} | Reward: {episode_reward:7.2f} | "
                      f"Avg: {current_mean:7.2f} | Steps: {step_count:4d} | "
                      f"Vel: {ep_metrics['velocity']:.3f} | Height: {ep_metrics['height']:.3f}")
        
        writer.close()
        return episode_rewards

# Пример использования
if __name__ == '__main__':
    from a1_pyb_env import PyBulletA1
    from gym_wrapper import RobotEnvWrapper
    
    # Создание среды
    robot = PyBulletA1(render_mode='human')
    env = RobotEnvWrapper(robot, max_episode_steps=1000)
    
    # Инициализация агента
    agent = PPOAgent(env, 
                     lr=3e-4,
                     batch_size=8192,
                     mini_batch_size=512,
                     epochs=10,
                     hidden_dim=256,
                     vf_coef=0.5,
                     ent_coef=0.01)
    
    # Обучение
    rewards = agent.train(env, 
                         episodes=5000,
                         max_steps=1000,
                         update_freq=8192,
                         save_interval=200,
                         log_dir="runs/ppo_a1_train")