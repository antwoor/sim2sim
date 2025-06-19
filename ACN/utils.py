from .ACN import *
from .model import *
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Any
from torch.utils.tensorboard import SummaryWriter

def collect_sim2sim_data(pb_env, mj_env, policy, num_episodes=100, max_steps=500):
    """Сбор параллельных данных из двух симуляторов"""
    pb_states, pb_actions, pb_next_states = [], [], []
    mj_states, mj_actions, mj_next_states = [], [], []
    
    for ep in range(num_episodes):
        # Синхронизированная инициализация
        pb_state = pb_env.reset()
        mj_state = mj_env.reset()
        
        # Установка идентичных начальных условий
        #mj_env.set_state(pb_state)
        
        for _ in range(max_steps):
            # Генерация действия
            action = policy.act(pb_state)[0]
            
            # Шаг в PyBullet
            pb_next_state, _, _, _ = pb_env.step(action)
            
            # Шаг в MuJoCo
            mj_next_state, _, _, _ = mj_env.step(action)
            
            # Сохранение данных
            pb_states.append(pb_state)
            pb_actions.append(action)
            pb_next_states.append(pb_next_state)
            
            mj_states.append(mj_state)
            mj_actions.append(action)
            mj_next_states.append(mj_next_state)
            
            # Обновление состояний
            pb_state = pb_next_state
            mj_state = mj_next_state

        if ep % num_episodes ==5:
            print("episode number", ep, "|", num_episodes-ep, "remaining")
    
    return {
        'pb': (np.array(pb_states), np.array(pb_actions), np.array(pb_next_states)),
        'mj': (np.array(mj_states), np.array(mj_actions), np.array(mj_next_states))
    }

def prepare_dataset(sim_data):
    """Подготовка датасета для обучения"""
    pb_states, pb_actions, pb_next_states = sim_data['pb']
    mj_states, mj_actions, mj_next_states = sim_data['mj']
    
    # Вычисление разницы в изменениях состояний
    pb_delta = pb_next_states - pb_states
    mj_delta = mj_next_states - mj_states
    delta_states = pb_delta - mj_delta
    
    # Фильтрация данных по схожести состояний
    state_diffs = np.linalg.norm(pb_states - mj_states, axis=1)
    threshold = np.percentile(state_diffs, 70)  # Берем 70% наиболее схожих состояний
    mask = state_diffs < threshold
    
    return Sim2SimDataset(
        pb_states[mask],
        pb_actions[mask],
        delta_states[mask]
    )

def train_corrector(dataset, epochs=100, batch_size=256, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Получаем размерности из первого элемента датасета
    state_dim = dataset[0]['state'].shape[0]
    action_dim = dataset[0]['action'].shape[0]
    
    model = ActionCorrectionModel(
        state_dim=state_dim,
        action_dim=action_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            s = batch['state'].float().to(device)
            a = batch['action'].float().to(device)
            target_delta_state = batch['delta_state'].float().to(device)
            
            # Предсказываем коррекцию и якобиан
            delta_a, jacobian = model(s, a)
            
            # Аппроксимируем изменение состояния
            # Δstate ≈ J · Δaction
            predicted_delta_state = torch.bmm(jacobian, delta_a.unsqueeze(-1)).squeeze(-1)
            
            # Сравниваем с целевым изменением состояния
            loss = torch.mean(torch.norm(predicted_delta_state - target_delta_state, dim=1)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    return model, state_dim, action_dim  # Возвращаем также размерности!

def train_ppo_with_corrector(pb_env, mj_env, agent, episodes=5000, max_steps=1000, update_freq=500):
    """Обучение PPO с корректором действий"""
    # Создаем директорию для сохранения результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = f"eval_runs/run_{timestamp}"
    correctors_dir = f"correctors/run_{timestamp}"
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(correctors_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=eval_dir)
    episode_rewards = []
    mj_rewards = []
    pyb_rewards = []
    
    # Инициализация переменных для размерностей
    state_dim = None
    action_dim = None
    
    for episode in range(episodes):
        # Сбор данных для коррекции (каждые 50 эпизодов)
        if episode % 50 == 0:
            sim_data = collect_sim2sim_data(pb_env, mj_env, agent, num_episodes=6)
            dataset = prepare_dataset(sim_data)
            
            # Обучаем корректор и получаем размерности
            corrector_model, state_dim, action_dim = train_corrector(dataset)
            
            # Сохраняем веса модели
            model_path = f"{correctors_dir}/ACN_{episode}.pth"
            torch.save(corrector_model.state_dict(), model_path)
        
        # Обучение в PyBullet
        state = pb_env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action, log_prob, value = agent.act(state)
            
            # Шаг в среде
            next_state, reward, done, info = pb_env.step(action)

            # Сохранение перехода
            agent.store_transition(state, action, reward, done, log_prob, value)
            
            # Обновление статистики
            episode_reward += reward
            state = next_state
            
            # Обновление политики
            if len(agent.buffer['states']) >= update_freq:
                _, last_value = agent.old_policy(torch.FloatTensor(
                    agent.normalize_obs(next_state)).unsqueeze(0))
                agent.update(last_value.item())
            
            # Сохранение результатов
            episode_rewards.append(episode_reward)
            
            if done:
                break
        
        # Валидация в MuJoCo с корректором
        if state_dim is not None and action_dim is not None:
            corrector = ActionCorrector(
                model_path=model_path,
                state_dim=state_dim,
                action_dim=action_dim
            )
            pyb_reward = evaluate_in_pyb(pb_env, agent, max_steps, corrector)
            mj_reward = evaluate_in_mujoco(mj_env, agent, max_steps, corrector)
            naked_pyb_reward = evaluate_in_pyb(pb_env, agent, max_steps, corrector=None)
            naked_mj_reward = evaluate_in_mujoco(mj_env, agent, max_steps, corrector=None)
        else:
            print("MUJOCO REWARD IS 0")
            mj_reward = 0  # На первых итерациях, пока нет корректора
            pyb_reward = 0
        # Логирование
        writer.add_scalar('Reward/PyBullet', pyb_reward, episode)
        writer.add_scalar('Reward/MuJoCo', mj_reward, episode)
        writer.add_scalar('Naked_Reward/PyBullet', naked_pyb_reward, episode)
        writer.add_scalar('Naked_Reward/MuJoCo', naked_mj_reward, episode)
        episode_rewards.append(episode_reward)
        mj_rewards.append(mj_reward)
        pyb_rewards.append(pyb_reward)
        
        # Сохранение графиков каждые 100 эпизодов
        if episode % 100 == 0 or episode == episodes - 1:
            plt.figure(figsize=(12, 6))
            
            # График наград PyBullet
            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards, label='PyBullet Reward')
            plt.title('Training in PyBullet')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            
            # График наград MuJoCo
            plt.subplot(1, 2, 2)
            plt.plot(mj_rewards, label='MuJoCo Reward', color='orange')
            plt.title('Evaluation in MuJoCo')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{eval_dir}/rewards_episode_{episode}.png")
            plt.close()
    
    # Сохранение финальных результатов
    save_training_results(eval_dir, episode_rewards, mj_rewards)
    writer.close()
    return episode_rewards

def save_training_results(eval_dir, pb_rewards, mj_rewards):
    """Сохраняет финальные результаты обучения"""
    # Сохраняем сырые данные
    np.savez(f"{eval_dir}/final_rewards.npz", 
             pb_rewards=pb_rewards, 
             mj_rewards=mj_rewards)
    
    # Создаем финальный график
    plt.figure(figsize=(10, 5))
    plt.plot(pb_rewards, label='PyBullet Training')
    plt.plot(mj_rewards, label='MuJoCo Evaluation', alpha=0.7)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{eval_dir}/final_rewards.png")
    plt.close()

def evaluate_in_mujoco(env, agent, max_steps, corrector: Any | None = None):
    """Оценка производительности в MuJoCo с коррекцией"""
    state = env.reset()
    total_reward = 0
    if corrector is None:
        for _ in range(max_steps):
        # Генерация и коррекция действия
            action, log_prob, value = agent.act(state)
            # Шаг в среде
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                break
    else:   
        for _ in range(max_steps):
            # Генерация и коррекция действия
            action, log_prob, value = agent.act(state)
            corrected_action = corrector.correct(state, action)
            
            # Шаг в среде
            next_state, reward, done, _ = env.step(corrected_action)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
    
    return total_reward

def evaluate_in_pyb(env, agent,  max_steps, corrector: Any | None = None):
    """Оценка производительности в MuJoCo с коррекцией"""
    state = env.reset()
    total_reward = 0
    if corrector is None:
        for _ in range(max_steps):
        # Генерация и коррекция действия
            action, log_prob, value = agent.act(state)

            # Шаг в среде
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                break
    else:   
        for _ in range(max_steps):
            # Генерация и коррекция действия
            action, log_prob, value = agent.act(state)
            corrected_action = corrector.correct(state, action)

            # Шаг в среде
            next_state, reward, done, _ = env.step(corrected_action)

            total_reward += reward
            state = next_state

            if done:
                break
    
    return total_reward