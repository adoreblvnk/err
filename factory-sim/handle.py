import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from factory_simulator import FactoryEnv

def train_and_evaluate_factory(
    total_timesteps=50000,
    learning_rate=0.001,
    n_steps=128,
    batch_size=64,
    gamma=0.99,
    max_material_rate=5,
    target_throughput=50,
    eval_steps=50,
    render=False,
):
    """
    Train a PPO agent on the factory environment and return the optimal configuration found.
    """
    # Create vectorized environment
    env = DummyVecEnv([lambda: FactoryEnv(max_material_rate=max_material_rate,
                                          target_throughput=target_throughput,
                                          render_mode="human" if render else None)])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
    )
    
    # Train the agent
    print('training model...')
    model.learn(total_timesteps=total_timesteps)
    print('done training model...')
    
    # Evaluate agent to find best configuration
    obs = env.reset()
    best_reward = -np.inf
    best_devices = -np.inf
    best_action = None
    best_step_info = None
    
    def format_action(action):
        action = action[0]  # VecEnv returns batch dimension
        action = [*map(int, action)]
        return {
            "normal_machines": action[0],
            "high_powered_machines": action[1],
            "normal_assemblers": action[2],
            "high_powered_assemblers": action[3]
        }
    
    for step in range(eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        
        if reward > best_reward:
            best_reward = reward
            best_devices = obs[0][2]
            best_action = action
            best_step_info = info
        
        if render:
            env.render()
        
        if terminated:
            obs = env.reset()
    
    print("\n=== Optimal Configuration Found ===")
    print(f"Configuration: {format_action(best_action)}")

    return [format_action(best_action), {
        'material_backlog': obs[0][0] * 20,
        'devices_made': obs[0][1] * 50,
    }]

# Example usage:
# train_and_evaluate_factory(total_timesteps=10000, render=True)

print(train_and_evaluate_factory(total_timesteps = 5000, render=True))