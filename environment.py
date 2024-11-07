import numpy as np
import gym
from gym.spaces import Box
from collections import deque
from stable_baselines3.common import atari_wrappers

def create_atari_environment(
    env_name: str,
    frame_stack: int = 4,
    noop_max: int = 30,
) -> gym.Env:
    name =f'{env_name}NoFrameskip-v4'
    env = gym.make(name)

    env = atari_wrappers.AtariWrapper(env, clip_reward=True, noop_max=noop_max, terminal_on_life_loss=False)

    return env
