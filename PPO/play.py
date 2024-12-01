import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.distributions.categorical import Categorical

# Load the saved model
model_path = "./models/step_5000000_ppo_agent.pt"  # Replace with your saved model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same Agent class as in the training script
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)  # Include the critic

    def get_action(self, x):
        with torch.no_grad():
            hidden = self.network(x / 255.0)
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            action = probs.sample()
        return action

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Define the environment setup (must match the training environment)
def make_env(env_id):
    env = gym.make(env_id, render_mode="human")  # Enable render mode
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


# Load the environment and agent
env_id = "ALE/DemonAttack-v5"  # Replace with your environment ID
env = make_env(env_id)
obs_space = env.observation_space.shape
action_space = env.action_space.n

envs = gym.vector.SyncVectorEnv([lambda: env])
agent = Agent(envs)
agent.load_state_dict(torch.load(model_path, map_location=device))
agent.eval()

# Main loop for playing the game
obs, _ = env.reset(seed=42)
while True:
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
    action = agent.get_action(obs_tensor).item()  # Convert tensor to Python integer
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment
    # env.render()
    
    # Exit the loop if the episode ends
    if terminated or truncated:
        if info['lives'] == 0:
            break

env.close()