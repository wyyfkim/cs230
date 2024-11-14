import gymnasium as gym
import ale_py
import torch
from stable_baselines3 import PPO

total_timesteps = 100000 # Total number of samples (env steps) to train on
max_steps_per_episode = 1000  # Maximum number of steps per episode

# Create the environment
env = gym.make('ALE/DemonAttack-v5', render_mode='human')

# Create the PPO model with a CnnPolicy
model = PPO("CnnPolicy", env, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
model.learn(total_timesteps)

# Save the model
model.save("ppo_demon_attack")

# Load the model (optional)
# model = PPO.load("ppo_demon_attack")

# Run a test episode
obs, _ = env.reset()  # Unpack the observation from the tuple

for _ in range(max_steps_per_episode):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)  # Handle both `done` and `truncated`
    env.render()  # Render without unpacking
    
    if done or truncated:  # Reset when the episode ends
        obs, _ = env.reset()

# Close the environment
env.close()