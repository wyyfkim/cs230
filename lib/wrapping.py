#pytorch agentnet
import ptan.common.wrappers as ptanwrap
# import gym
# import gym.spaces as spaces
# import numpy as np
# #opencv
# import cv2

def wrap_demon_attack(env, stack_frames=4, episodic_life=True):
  """Apply a common set of wrappers for Atari games."""
  assert 'NoFrameskip' in env.spec.id
  if episodic_life:
    env = ptanwrap.EpisodicLifeEnv(env)
  env = ptanwrap.NoopResetEnv(env, noop_max=30)
  env = ptanwrap.MaxAndSkipEnv(env, skip=4)
  if 'FIRE' in env.unwrapped.get_action_meanings():
    env = ptanwrap.FireResetEnv(env)
  env = ptanwrap.ProcessFrame84(env)
  env = ptanwrap.ImageToPyTorch(env)
  env = ptanwrap.FrameStack(env, stack_frames)
  env = ptanwrap.ClippedRewardsWrapper(env)
  return env

def wrap_demon_attack_test(env, stack_frames=4):
  """Apply a common set of wrappers for Atari games."""
  #!!!!!!No frameskip package
  # assert 'NoFrameskip' in env.spec.id

  # !!!!!!ptanwrap package updated
  #if episodic_life:
    #env = ptanwrap.EpisodicLifeEnv(env)
  # env = ptanwrap.NoopResetEnv(env, noop_max=30)
  # env = ptanwrap.MaxAndSkipEnv(env, skip=4)
  # if 'FIRE' in env.unwrapped.get_action_meanings():
  #   env = ptanwrap.FireResetEnv(env)
  # env = ptanwrap.ProcessFrame84(env)
  env = ptanwrap.ImageToPyTorch(env)
  # env = ptanwrap.FrameStack(env, stack_frames)
  #env = ptanwrap.ClippedRewardsWrapper(env)
  return env