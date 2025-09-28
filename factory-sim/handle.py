import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from factory_simulator import FactoryEnv

# Create vectorized environment
env = DummyVecEnv([lambda: FactoryEnv(max_material_rate=5, target_throughput=50, render_mode="human")])

# Load the trained model
model = PPO.load("ppo_factory", env=env)

# Test the trained agent
obs = env.reset()
best_reward = -np.inf
best_action = None
best_step_info = None

def format_action(action):
    action = action[0]
    action = [*map(int, action)]
    return f"\nnormal machines: {action[0]}\nhigh powered machines: {action[1]}\nnormal assemblers: {action[2]}\nhigh powered assemblers: {action[3]}"

for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, info = env.step(action)
    
    # Check if this is the best configuration so far
    if reward > best_reward:
        best_reward = reward
        best_action = action
        best_step_info = info
    
    print(f"Step {step}: Action={action}, Reward={reward}, Info={info[0]}")
    env.render()
    
    if terminated:
        obs = env.reset()

print("\n=== Optimal Configuration Found ===")
print(f"Configuration: {format_action(best_action)}")