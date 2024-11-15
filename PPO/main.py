import gymnasium as gym
import ale_py
import torch
from stable_baselines3 import PPO

total_timesteps = 100000 # Total number of samples (env steps) to train on
max_steps_per_episode = 100000  # Maximum number of steps per episode
test_episode_times = 10

# Create the environment
env = gym.make('ALE/DemonAttack-v5')

# Create the PPO model with a CnnPolicy
model = PPO("CnnPolicy", env, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
model.learn(total_timesteps)

# Save the model
model.save("ppo_demon_attack")

# Load the model (optional)
# model = PPO.load("ppo_demon_attack")

# Run a test episode
total_reward = 0
for i in range(test_episode_times):
    episode_reward = 0  # Initialize the episode reward counter
    obs, _ = env.reset()
    step_cnt = 0
    for _ in range(max_steps_per_episode):
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action)  # Handle both `done` and `truncated`
        #env.render()  # Render without unpacking
        
        episode_reward += rewards  # Accumulate the rewards

        # Check if the game is over by looking at the 'lives' in the info dictionary
        if info.get('lives', 1) == 0:  # The default is 1 if 'lives' is not present in `info`
            print(f"Game over: agent ran out of lives. Score: {episode_reward}")
            break
        
        if done or truncated:  # Print score and reset when the episode ends(should be impossible)
            print(f"Episode finished. Score: {episode_reward}")

        step_cnt += 1
    print(f"The {i}th episode take {step_cnt} steps")
    total_reward += episode_reward
print(f"Played {test_episode_times} eposides and total score is {total_reward} and average each episode is {total_reward / test_episode_times}")

# Close the environment
env.close()