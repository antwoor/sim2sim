import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import time
from .model import PPONetwork, torch
from torch.utils.tensorboard import SummaryWriter
import os

class PPOAgent:
    def __init__(self, env, gamma=0.99, lr=3e-4, clip_epsilon=0.2, 
                 epochs=10, batch_size=64, hidden_dim=64, load_path=None):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Initialize networks
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]  # Автоматическое определение размерности действий
        print("obs_dim", obs_dim)
        print("act_dim", act_dim)
        self.policy = PPONetwork(obs_dim, act_dim, hidden_dim)
        self.old_policy = PPONetwork(obs_dim, act_dim, hidden_dim)
        if load_path is not None:
            self.load_model(load_path)
        else:
            self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def load_model(self, path):
        """Загружает сохранённые веса модели"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint)
        self.old_policy.load_state_dict(checkpoint)
        print(f"Loaded model weights from {path}")

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.old_policy(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)
            
        return action.squeeze(0).numpy(), log_prob.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def compute_returns(self):
        returns = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self):
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).unsqueeze(1)
        returns = self.compute_returns()
        
        # Calculate advantages
        with torch.no_grad():
            _, values = self.policy(states)
            advantages = returns - values.squeeze()
        
        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Shuffle the data
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get new policy
                dist, values = self.policy(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratio (pi_theta / pi_theta_old)
                ratio = (new_log_probs - batch_old_log_probs).exp()
                
                # Policy loss
                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
    
    @staticmethod
    def train_ppo(env, agent, episodes=1000, max_steps=1000, update_freq=2048, 
                  save_interval=500, log_dir="runs/ppo_a1"):
        """
        Обучение PPO для шагающего робота Unitree A1

        Args:
            env: Среда с роботом (должна реализовывать get_metrics() для дополнительных метрик)
            agent: PPO агент
            episodes: Количество эпизодов обучения
            max_steps: Максимальное количество шагов в эпизоде
            update_freq: Частота обновления политики (в шагах)
            save_interval: Частота сохранения модели (в эпизодах)
            log_dir: Директория для логов TensorBoard
        """
        # Инициализация директорий
        os.makedirs("a1_weights", exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        episode_rewards = []
        best_mean_reward = -np.inf

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
                # Получаем действие
                action, log_prob = agent.act(state)

                # Шаг в среде
                next_state, reward, done, info = env.step(action)

                # Сохраняем переход
                agent.store_transition(state, action, reward, next_state, done, log_prob)

                # Обновляем статистику
                episode_reward += reward
                step_count += 1
                state = next_state

                # Логируем дополнительные метрики из среды
                if hasattr(env, 'get_metrics'):
                    metrics = env.get_metrics()
                    for k in ep_metrics:
                        ep_metrics[k] += metrics.get(k, 0)

                # Обновляем политику при накоплении достаточного числа примеров
                if len(agent.states) >= update_freq:
                    agent.update()

            # Нормализуем метрики по количеству шагов
            for k in ep_metrics:
                ep_metrics[k] /= max(1, step_count)

            # Сохраняем результаты
            episode_rewards.append(episode_reward)
            current_mean = np.mean(episode_rewards[-10:])

            # Логирование в TensorBoard
            writer.add_scalar('Reward/Episode', episode_reward, episode)
            writer.add_scalar('Reward/Average_10', current_mean, episode)

            # Логирование специфичных метрик
            writer.add_scalar('Metrics/Velocity', ep_metrics['velocity'], episode)
            writer.add_scalar('Metrics/Height', ep_metrics['height'], episode)
            writer.add_scalar('Metrics/Orientation', ep_metrics['orientation'], episode)
            writer.add_scalar('Metrics/Foot_Contacts', ep_metrics['contacts'], episode)
            writer.add_scalar('Metrics/Energy', ep_metrics['energy'], episode)
            writer.add_scalar('Metrics/Survival_Steps', step_count, episode)

            # Сохранение модели
            if episode % save_interval == 0 or current_mean > best_mean_reward:
                if current_mean > best_mean_reward:
                    best_mean_reward = current_mean
                    torch.save(agent.policy.state_dict(), f"a1_weights/best_ppo.pth")
                torch.save(agent.policy.state_dict(), f"a1_weights/ppo_{episode}.pth")

            # Вывод прогресса
            if episode % 10 == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg 10: {current_mean:7.2f} | "
                      f"Steps: {step_count:4d} | "
                      f"Vel: {ep_metrics['velocity']:5.3f} | "
                      f"Height: {ep_metrics['height']:5.3f}")

        writer.close()
        return episode_rewards

# Initialize environment and agent
if __name__ =='__main__':
    from go1_env import Go1Env  # Импорт вашего окружения
    
    env = Go1Env()
    agent = PPOAgent(env, lr=3e-4, hidden_dim=256)

    # Train the agent
    rewards = agent.train_ppo(env, agent, episodes=500, dynamic=False)