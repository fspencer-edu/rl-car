import numpy as np
from tqdm import tqdm


def epsilon_greedy(q_table, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(q_table.shape[1])
    return int(np.argmax(q_table[state]))


def run_greedy_episode(env, q_table):
    state = env.reset()
    path = [env.pos.copy()]
    total_reward = 0.0
    success = False

    for _ in range(env.max_steps):
        action = int(np.argmax(q_table[state]))
        state, reward, done = env.step(action)
        total_reward += reward
        path.append(env.pos.copy())

        if env.reached_goal():
            success = True

        if done:
            break

    return path, success, total_reward


def train_q_learning(
    env,
    episodes=600,
    alpha=0.18,
    gamma=0.96,
    epsilon=1.0,
    epsilon_decay=0.992,
    epsilon_min=0.05,
):
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float32)

    rewards = []
    steps_history = []
    success_history = []

    snapshot_eps = [0, 9, 29, 79, 199, 399, 599]
    snapshot_paths = {}
    snapshot_meta = {}

    for ep in tqdm(range(episodes), desc="Training car"):
        state = env.reset()
        total_reward = 0.0
        path = [env.pos.copy()]
        success = False

        for _ in range(env.max_steps):
            action = epsilon_greedy(q_table, state, epsilon)
            next_state, reward, done = env.step(action)

            best_next = np.max(q_table[next_state])
            td_target = reward + gamma * best_next * (1 - int(done))
            q_table[state, action] += alpha * (td_target - q_table[state, action])

            state = next_state
            total_reward += reward
            path.append(env.pos.copy())

            if env.reached_goal():
                success = True

            if done:
                break

        rewards.append(total_reward)
        steps_history.append(len(path) - 1)
        success_history.append(int(success))

        if ep in snapshot_eps:
            demo_path, demo_success, demo_reward = run_greedy_episode(env, q_table)
            snapshot_paths[ep] = demo_path
            snapshot_meta[ep] = {
                "success": demo_success,
                "reward": demo_reward,
                "steps": len(demo_path) - 1,
            }

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, rewards, steps_history, success_history, snapshot_paths, snapshot_meta


def moving_average(data, window=20):
    data = np.asarray(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")
