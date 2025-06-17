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

    def train_ppo(self, env, agent, episodes=1000, max_steps=500, update_freq=2048, dynamic=False, freq=None, act_k=0, pos_k=0):
        episode_rewards = []
        if not dynamic:
            base_dir = "static_weights"
            weights_dir = base_dir
            counter = 1
            while os.path.exists(weights_dir):
                weights_dir = f"{base_dir}{counter}"
                counter += 1
            os.makedirs(weights_dir, exist_ok=True)

            # Инициализация TensorBoard
            tb_dir = f"runs/ppo_{os.path.basename(weights_dir)}"
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
            
            for episode in range(episodes):
                state = env.reset()
                episode_reward = 0
                
                for step in range(max_steps):
                    # Get action
                    action, log_prob = agent.act(state)
                    
                    # Take step
                    next_state, reward, done, _ = env.step(action)
                    
                    # Store transition
                    agent.store_transition(state, action, reward, next_state, done, log_prob)
                    
                    state = next_state
                    episode_reward += reward
                    
                    # Update if we have enough samples
                    if len(agent.states) >= update_freq:
                        agent.update()
                    
                    if done:
                        break
                
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else episode_reward
                writer.add_scalar('Reward/Episode', episode_reward, episode)
                writer.add_scalar('Reward/Avg_10', avg_reward, episode)
                episode_rewards.append(episode_reward)
                
                # Print progress
                if episode % 10 == 0:
                    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
                    if episode % 500 == 0:
                        torch.save(self.policy.state_dict(), f"{weights_dir}/ppo_{episode}.pth")
        else:
            act_k = 1
            pos_k = 1
            base_dir = "dynamic_weights"
            weights_dir = base_dir
            counter = 1
            while os.path.exists(weights_dir):
                weights_dir = f"{base_dir}{counter}"
                counter += 1
            os.makedirs(weights_dir, exist_ok=True)

            # Инициализация TensorBoard
            tb_dir = f"runs/ppo_{os.path.basename(weights_dir)}"
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)

            print(f"Сохранение весов в: {weights_dir}")
            print(f"Логи TensorBoard в: {tb_dir}")

            for episode in range(episodes):
                state = env.reset()
                episode_reward = 0
                target_update_freq = 100 if freq is None else freq

                # Генерируем новую цель
                if episode % target_update_freq == 0:
                    target_pos = [np.random.rand() - 0.5, 0, 0.6]
                    env.set_target(target_pos)  # Предполагается метод в среде

                # Переменные для агрегации метрик за эпизод
                total_angle_reward = 0
                total_distance_reward = 0
                total_action_penalty = 0
                total_distance = 0
                steps = 0

                for step in range(max_steps):
                    action, log_prob = agent.act(state)
                    next_state, reward, done, info = env.step(action)

                    # Агрегируем метрики
                    total_angle_reward += info.get('angle_reward', 0)
                    total_distance_reward += info.get('distance_reward', 0)
                    total_action_penalty += info.get('action_penalty', 0)
                    total_distance += info.get('distance_to_target', 0)
                    steps += 1

                    agent.store_transition(state, action, reward, next_state, done, log_prob)
                    state = next_state
                    episode_reward += reward

                    if len(agent.states) >= update_freq:
                        agent.update()

                    if done:
                        break
                    
                # Логирование в TensorBoard
                writer.add_scalar('Reward/Total', episode_reward, episode)
                writer.add_scalar('Reward/Angle', total_angle_reward/steps, episode)
                writer.add_scalar('Reward/Distance', total_distance_reward/steps, episode)
                writer.add_scalar('Penalty/Action', total_action_penalty/steps, episode)
                writer.add_scalar('Metrics/Distance_to_target', total_distance/steps, episode)
                writer.add_scalar('Params/Position_k', pos_k, episode)
                writer.add_scalar('Params/Action_k', act_k, episode)

                episode_rewards.append(episode_reward)

                # Консольное логирование
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
                    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

                if episode % 500 == 0:
                    torch.save(agent.policy.state_dict(), f"{weights_dir}/ppo_{episode}.pth")

        writer.close()
        return episode_rewards

# Initialize environment and agent
if __name__ =='__main__':
    from go1_env import Go1Env  # Импорт вашего окружения
    
    env = Go1Env()
    agent = PPOAgent(env, lr=3e-4, hidden_dim=256)

    # Train the agent
    rewards = agent.train_ppo(env, agent, episodes=500, dynamic=False)