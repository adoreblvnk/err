import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import gymnasium as gym
from gymnasium import spaces

class FactoryEnv(gym.Env):
    """
    Factory simulation with regular and high-powered machines/assemblers.
    High-powered units process twice as fast but fail more often.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_material_rate=20,
        target_throughput=10,
        max_machines=30,
        max_high_powered_machines=10,
        max_assemblers=30,
        max_high_powered_assemblers=10,
        base_machine_speed=1.0,
        base_assembler_speed=1.0,
        maintenance_time=2,
        render_mode="human"
    ):
        super().__init__()
        self.render_mode = render_mode

        # Action space:
        # [num_machines, num_high_powered_machines, num_assemblers, num_high_powered_assemblers]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([max_machines, max_high_powered_machines, max_assemblers, max_high_powered_assemblers], dtype=np.float32)
        )

        # Observation space: [material_backlog, component_inventory, device_inventory, machines_in_maintenance, assemblers_in_maintenance]
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

        # Environment parameters
        self.max_material_rate = max_material_rate
        self.target_throughput = target_throughput
        self.base_machine_speed = base_machine_speed
        self.base_assembler_speed = base_assembler_speed
        self.maintenance_time = maintenance_time
        self.machine_fail_prob = 0.05
        self.assembler_fail_prob = 0.05

        # Machine/assembler limits
        self.max_machines = max_machines
        self.max_high_powered_machines = max_high_powered_machines
        self.max_assemblers = max_assemblers
        self.max_high_powered_assemblers = max_high_powered_assemblers

        # State
        self.current_step = 0
        self.material_backlog = 0
        self.component_inventory = 10
        self.device_inventory = 0
        self.prev_action = None
        self.past_reward = 0

        # Maintenance arrays
        self.machine_maintenance = np.zeros(self.max_machines + self.max_high_powered_machines)
        self.assembler_maintenance = np.zeros(self.max_assemblers + self.max_high_powered_assemblers)

        self.reset()

    def step(self, action):
        num_machines, num_high_machines, num_assemblers, num_high_assemblers = action
        num_machines = int(num_machines)
        num_high_machines = int(num_high_machines)
        num_assemblers = int(num_assemblers)
        num_high_assemblers = int(num_high_assemblers)


        # Update maintenance
        self.machine_maintenance = np.maximum(0, self.machine_maintenance - 1)
        self.assembler_maintenance = np.maximum(0, self.assembler_maintenance - 1)

        # Count active machines/assemblers
        active_machines = num_machines - np.sum(self.machine_maintenance[:num_machines] > 0)
        active_high_machines = num_high_machines - np.sum(self.machine_maintenance[num_machines:num_machines+num_high_machines] > 0)
        active_assemblers = num_assemblers - np.sum(self.assembler_maintenance[:num_assemblers] > 0)
        active_high_assemblers = num_high_assemblers - np.sum(self.assembler_maintenance[num_assemblers:num_assemblers+num_high_assemblers] > 0)

        # Random failures
        for i in range(num_machines):
            if self.machine_maintenance[i] == 0 and np.random.rand() < self.machine_fail_prob:
                self.machine_maintenance[i] = self.maintenance_time
        for i in range(num_machines, num_machines+num_high_machines):
            if self.machine_maintenance[i] == 0 and np.random.rand() < self.machine_fail_prob*2:
                self.machine_maintenance[i] = self.maintenance_time

        for i in range(num_assemblers):
            if self.assembler_maintenance[i] == 0 and np.random.rand() < self.assembler_fail_prob:
                self.assembler_maintenance[i] = self.maintenance_time
        for i in range(num_assemblers, num_assemblers+num_high_assemblers):
            if self.assembler_maintenance[i] == 0 and np.random.rand() < self.assembler_fail_prob*2:
                self.assembler_maintenance[i] = self.maintenance_time

        # Material arrival
        materials_arriving = self.max_material_rate

        # Process materials into components
        possible_components = active_machines * self.base_machine_speed + active_high_machines * (2*self.base_machine_speed)
        components_made = min(materials_arriving + self.material_backlog, possible_components)
        self.material_backlog = max(0, materials_arriving + self.material_backlog - components_made)

        # Process components into devices
        possible_devices = active_assemblers * self.base_assembler_speed + active_high_assemblers * (2*self.base_assembler_speed)
        devices_made = min(self.component_inventory + components_made, possible_devices)
        self.component_inventory = self.component_inventory + components_made - devices_made
        self.device_inventory += devices_made

        # Observation
        obs = np.array([
            self.material_backlog / max(1, self.max_material_rate),
            self.component_inventory / 50.0,
            self.device_inventory / 50.0,
            np.sum(self.machine_maintenance) / (self.max_machines + self.max_high_powered_machines),
            np.sum(self.assembler_maintenance) / (self.max_assemblers + self.max_high_powered_assemblers)
        ], dtype=np.float32)

        # Reward = throughput normalized + penalty for resource usage
        throughput_reward = devices_made / max(1.0, self.target_throughput)
        reward = throughput_reward

        self.current_step += 1
        self.prev_action = action
        self.past_reward = reward

        done = self.current_step >= 128
        info = {"throughput": devices_made,
                "active_machines": active_machines + active_high_machines,
                "active_assemblers": active_assemblers + active_high_assemblers}

        if self.current_step % 100 == 0:
                #print(f'testing this configuration:\n {num_machines = }\n {num_high_machines = }\n {num_assemblers = }\n {num_high_assemblers = }\n')
                pass
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.material_backlog = 0
        self.component_inventory = 0
        self.device_inventory = 0
        self.num_machines = 10
        self.num_high_powered_machines = 5
        self.num_assemblers = 10
        self.num_high_powered_assembers = 5
        self.machine_maintenance = np.zeros(self.max_machines + self.max_high_powered_machines)
        self.assembler_maintenance = np.zeros(self.max_assemblers + self.max_high_powered_assemblers)

        obs = np.array([
            self.material_backlog / max(1, self.max_material_rate),
            self.component_inventory / 50.0,
            self.device_inventory / 50.0,
            np.sum(self.machine_maintenance) / (self.max_machines + self.max_high_powered_machines),
            np.sum(self.assembler_maintenance) / (self.max_assemblers + self.max_high_powered_assemblers)
        ], dtype=np.float32)
        info = {}
        return obs, info

    def render(self, mode="human"):
        if self.render_mode != "human":
            return
        print(f"Step {self.current_step}")
        print(f"Material backlog: {self.material_backlog:.1f}, Components: {self.component_inventory:.1f}, Devices: {self.device_inventory:.1f}")
        print(f"Machines in maintenance: {np.sum(self.machine_maintenance>0)}, Assemblers in maintenance: {np.sum(self.assembler_maintenance>0)}")




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
        'material_backlog': int(obs[0][0] * 20),
        'devices_made': int(obs[0][1] * 50),
    }]

# Example usage:
# train_and_evaluate_factory(total_timesteps=10000, render=True)

#print(train_and_evaluate_factory(total_timesteps = 5000, render=True))
