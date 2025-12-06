import numpy as np
import matplotlib.pyplot as plt
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# === Dummy Brain-Command Environment (Simulates different risk zones) ===
class BrainCommandEnv(gym.Env):
    def __init__(self, zone="underwater"):
        super(BrainCommandEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)  # Stop, Left, Right, Forward
        self.zone = zone
        self.state = np.random.rand(4)
        self.steps = 0
        self.max_steps = 100

    def step(self, action):
        reward = self._compute_reward(action)
        self.state = np.random.rand(4)
        self.steps += 1
        done = self.steps >= self.max_steps
        return self.state, reward, done, {}

    def reset(self):
        self.steps = 0
        self.state = np.random.rand(4)
        return self.state

    def _compute_reward(self, action):
        zone_weights = {
            "underwater":    [1.0, 0.6, 0.6, 1.2],
            "nuclear":       [1.2, 0.5, 0.5, 1.0],
            "earthquake":    [1.5, 0.7, 0.7, 0.9]
        }
        weights = zone_weights.get(self.zone, [1.0]*4)
        return weights[action]

# === DQN Agent ===
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.max(self.model.predict(next_state[np.newaxis], verbose=0)[0])
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            q_values[0][action] = target
            self.model.fit(state[np.newaxis], q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# === Training Function ===
def train_rl_agent(zone="underwater", episodes=50):
    env = BrainCommandEnv(zone)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    reward_history = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        reward_history.append(total_reward)
        print(f"Episode {e+1}/{episodes} - Zone: {zone:<10} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    return agent, reward_history

# === Evaluation Function ===
def evaluate_agent(agent, zone):
    env = BrainCommandEnv(zone)
    state = env.reset()
    total_reward = 0
    for _ in range(env.max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    return total_reward

# === Enhanced 3-in-1 Plot Function ===
def plot_all_rewards_with_metrics(zone_rewards, adaptability_matrix, zones):
    plt.figure(figsize=(12, 6))
    for i, zone in enumerate(zones):
        rewards = zone_rewards[zone]
        avg_reward = np.mean(rewards[-10:])
        adaptation_score = np.sum(adaptability_matrix[i])
        label = f"{zone.capitalize()} | Avg: {avg_reward:.1f} | Adapt: {adaptation_score:.1f}"
        plt.plot(rewards, label=label)

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward (Proxy for Accuracy)")
    plt.title("RL Agent Training in Risk Zones\n(Accuracy = Avg. Reward, Adaptation = Cross-Zone Sum)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    zones = ["underwater", "nuclear", "earthquake"]
    zone_rewards = {}
    trained_agents = {}

    # Train agents
    for zone in zones:
        print(f"\n--- Training in {zone.upper()} zone ---")
        agent, rewards = train_rl_agent(zone=zone, episodes=50)
        zone_rewards[zone] = rewards
        trained_agents[zone] = agent

    # Evaluate cross-zone adaptability
    adaptability_matrix = np.zeros((len(zones), len(zones)))
    print("\n=== Cross-Zone Adaptability Evaluation ===")
    for i, train_zone in enumerate(zones):
        agent = trained_agents[train_zone]
        for j, eval_zone in enumerate(zones):
            reward = evaluate_agent(agent, eval_zone)
            adaptability_matrix[i][j] = reward
            print(f"Trained on {train_zone:<10} | Evaluated on {eval_zone:<10} -> Reward: {reward:.2f}")

    # Plot combined reward graph with metrics
    plot_all_rewards_with_metrics(zone_rewards, adaptability_matrix, zones)
