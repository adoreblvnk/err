import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from factory_simulator import FactoryEnv
from tqdm import tqdm

# Create vectorized environment
env = DummyVecEnv([lambda: FactoryEnv(max_material_rate=5, target_throughput=50, render_mode="human")])

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    n_steps=128,
    batch_size=64,
    gamma=0.99,
)

# Train the agent
model.learn(total_timesteps=50000)

# Save model
model.save("ppo_factory")

# Test the trained agent
obs = env.reset()
saved_info = []
for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    z =  env.step(action)
    print(action)
    env.render()

episode_rewards = []
episode_throughputs = []

for ep in range(10):
    obs = env.reset()
    total_reward = 0
    total_throughput = 0
    done = False
    i = 0
    for i in tqdm(range(1, 50000)):
        action, _ = model.predict(obs, deterministic=True)
        z = env.step(action)
        done = z[2][0]
        total_reward += z[1][0]
        total_throughput += z[-1][0]['throughput']
    episode_rewards.append(total_reward)
    episode_throughputs.append(total_throughput)

import matplotlib.pyplot as plt

plt.plot(episode_rewards, label="Reward per episode")
plt.plot(episode_throughputs, label="Throughput per episode")
plt.xlabel("Episode")
plt.ylabel("Value")
plt.legend()
plt.show()
