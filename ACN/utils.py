from .ACN import *
from .model import *
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

        if ep % num_episodes ==10:
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
    model = ActionCorrectionModel(
        state_dim=dataset[0]['state'].shape[0],
        action_dim=dataset[0]['action'].shape[0]
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
    
    return model

def train_ppo_with_corrector(pb_env, mj_env, agent, corrector, episodes=1000, max_steps=1000):
    """Обучение PPO с корректором действий"""
    writer = SummaryWriter()
    episode_rewards = []
    
    for episode in range(episodes):
        # Сбор данных для коррекции (каждые 50 эпизодов)
        if episode % 50 == 0:
            sim_data = collect_sim2sim_data(pb_env, mj_env, agent, num_episodes=5)
            dataset = prepare_dataset(sim_data)
            corrector = train_corrector(dataset)
        
        # Обучение в PyBullet
        state = pb_env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            # Генерация действия
            action, log_prob = agent.act(state)
            
            # Применение действия
            next_state, reward, done, _ = pb_env.step(action)
            
            # Обновление агента
            agent.store_transition(state, action, reward, next_state, done, log_prob)
            if len(agent.states) >= agent.update_freq:
                agent.update()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Валидация в MuJoCo с корректором
        mj_reward = evaluate_in_mujoco(mj_env, agent, corrector, max_steps)
        
        # Логирование
        writer.add_scalar('Reward/PyBullet', episode_reward, episode)
        writer.add_scalar('Reward/MuJoCo', mj_reward, episode)
        episode_rewards.append(episode_reward)
    
    writer.close()
    return episode_rewards

def evaluate_in_mujoco(env, agent, corrector, max_steps):
    """Оценка производительности в MuJoCo с коррекцией"""
    state = env.reset()
    total_reward = 0
    
    for _ in range(max_steps):
        # Генерация и коррекция действия
        action = agent.act(state)[0]
        corrected_action = corrector.correct(state, action)
        
        # Шаг в среде
        next_state, reward, done, _ = env.step(corrected_action)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return total_reward