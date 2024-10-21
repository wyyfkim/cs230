import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make('ALE/DemonAttack-v5', render_mode="human")

EPISODES = 1000
ACTION_NOTHING = 0
ACTION_FIRE = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3
ACTION_RIGHT_AND_FIRE = 4
ACTION_LEFT_AND_FIRE = 5

state = env.reset()
for e in range(EPISODES):
     state = env.step(ACTION_FIRE)
     env.render()
     print(e, state[1])