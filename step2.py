import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

# --- Setup ---
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0)

# For plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.set_title("Reward per Episode")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
reward_line, = ax1.plot([], [], label="Reward")
ax1.legend()

ax2.set_title("Performance (Moving Average)")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Avg Reward (last 10)")
perf_line, = ax2.plot([], [], label="Moving Avg (10)", color='orange')
ax2.legend()

# Data storage
episode_rewards = []
moving_avg_rewards = []

n_episodes = 100

for episode in range(n_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Train a little after each episode
    model.learn(total_timesteps=1024, reset_num_timesteps=False)

    # Store & plot
    episode_rewards.append(total_reward)
    moving_avg = np.mean(episode_rewards[-10:])
    moving_avg_rewards.append(moving_avg)

    # Update plots
    reward_line.set_data(np.arange(len(episode_rewards)), episode_rewards)
    perf_line.set_data(np.arange(len(moving_avg_rewards)), moving_avg_rewards)

    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()

    plt.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()
env.close()
