from typing import Iterable
import math
import gym
from absl import app
import numpy as np
from typing import NamedTuple, Tuple
import torch
from torch import nn
from absl import logging

from DeepQNetwork import agent as agent_lib
from DeepQNetwork import environment as gym_env
from DeepQNetwork import replay_lib as replay_lib
from DeepQNetwork import util

def calc_conv2d_output(h_w: Tuple, kernel_size: int = 1, stride: int = 1, pad: int = 0, dilation: int = 1) -> Tuple[int, int]:
    if not isinstance(kernel_size, Tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return (h, w)

class ConvNet(nn.Module):
    """DQN Nature paper conv2d layers backbone, returns feature representation vector."""

    def __init__(self, state_dim: tuple) -> None:
        super().__init__()

        # Compute the output shape of final conv2d layer
        c, h, w = state_dim
        h, w = calc_conv2d_output((h, w), 8, 4)
        h, w = calc_conv2d_output((h, w), 4, 2)
        h, w = calc_conv2d_output((h, w), 3, 1)

        self.out_features = 64 * h * w

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given raw state images, returns feature representation vector"""
        return self.net(x)

class DqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor

class DqnNet(nn.Module):
    def __init__(self, state_dim: tuple, action_dim: int):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output liner layer
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')

        super().__init__()
        self.action_dim = action_dim

        #network backbone
        self.body = ConvNet(state_dim)

        self.value_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

        # Initialize weights.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> DqnNetworkOutputs:
        x = x.float() / 255.0
        x = self.body(x)
        q_values = self.value_head(x)  # [batch_size, action_dim]
        return DqnNetworkOutputs(q_values=q_values)

class LinearSchedule:
    def __init__(self, begin_value, end_value, begin_t, end_t=None, decay_steps=None):
        if (end_t is None) == (decay_steps is None):
            raise ValueError('Exactly one of end_t, decay_steps must be provided.')
        self._decay_steps = decay_steps if end_t is None else end_t - begin_t
        self._begin_t = begin_t
        self._begin_value = begin_value
        self._end_value = end_value
    def __call__(self, t):
        frac = min(max(t - self._begin_t, 0), self._decay_steps) / self._decay_steps
        return (1 - frac) * self._begin_value + frac * self._end_value

def main(argv):
    logging.info("start")
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs DQN agent on {runtime_device}')
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exploration_epsilon_begin_value = 1.0 #Begin value of the exploration rate in e-greedy policy.
    exploration_epsilon_end_value = 0.01 #End (decayed) value of the exploration rate in e-greedy policy.
    exploration_epsilon_decay_step = int(1e6) # Total steps (after frame skip) to decay value of the exploration rate.
    eval_exploration_epsilon = 0.01 #Fixed exploration rate in e-greedy policy for evaluation.
    min_reply_size = 50000 #Minimum replay size before learning starts.

    random_state = np.random.RandomState(1)  # seed = 1

    # Create environment.
    print("Create environment")
    train_env = gym_env.create_atari_environment(
            env_name='DemonAttack',
            frame_stack=4,
            noop_max=30,
        )

    state_dim = train_env.observation_space.shape
    action_dim = train_env.action_space.n

    # Test environment and state shape.
    obs = train_env.reset()
    print("Create DqnNetwork")
    network = DqnNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.00025)

    # Create e-greedy exploration epsilon schedule
    print("Create  e-greedy exploration epsilon schedule")
    exploration_epsilon_schedule = LinearSchedule(
        begin_t=int(min_reply_size),
        decay_steps=int(exploration_epsilon_decay_step),
        begin_value=exploration_epsilon_begin_value,
        end_value=exploration_epsilon_end_value
    )

    print("create replay buffer")
    replay = replay_lib.UniformReplay(
        capacity=int(1e6), #Maximum replay size.
        structure=replay_lib.TransitionStructure,
        random_state=random_state,
        encoder=lambda transition: transition._replace(
            s_tm1=replay_lib.compress(transition.s_tm1),
            s_t=replay_lib.compress(transition.s_t),
        ),
        decoder=lambda transition: transition._replace(
        s_tm1=replay_lib.uncompress(transition.s_tm1),
        s_t=replay_lib.uncompress(transition.s_t),
        )
    )

    # Create DQN agent instance
    print("create train agent")
    train_agent = agent_lib.Dqn(
        network=network,
        optimizer=optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        replay=replay,
        exploration_epsilon=exploration_epsilon_schedule,
        batch_size=32, #Sample batch size when updating the neural network.
        min_replay_size=  min_reply_size,
        learn_interval=4, #The frequency (measured in agent steps) to update parameters.
        target_net_update_interval=2500, #The frequency (measured in number of Q network parameter updates) to update target networks.
        discount=0.99, #Discount rate.
        action_dim=action_dim,
        random_state=random_state,
        device=runtime_device,
    )

    # Run the training for N iterations.
    print("start iteration and training")
    num_iterations = 100
    for iteration in range(1, num_iterations + 1):
        print("Iteration: ", iteration)

        while True:  # For each episode.
            print("For each episode.")
            train_agent.reset()
            # Think of reset as a special 'action' the agent takes, thus given us a reward 'zero', and a new state 's_t'.
            observation = train_env.reset()
            reward = 0.0
            done = loss_life = False
            first_step = True
            info = {}

            while True:  # For each step in the current episode.
                # print("For each step in the current episode. Reward is ", reward)
                timestep_t = util.TimeStep(
                    observation=observation,
                    reward=reward,
                    done=done or loss_life,
                    first=first_step,
                    info=info,
                )
                #Given current timestep, do an action selection, add it to the replay, and sample from replay buffer and learn.
                a_t = train_agent.step(timestep_t)
                # yield train_env, timestep_t, train_agent, a_t

                a_tm1 = a_t
                observation, reward, done, info = train_env.step(a_tm1)

                first_step = False

                # For Atari games, check if should treat loss a life as a soft-terminal state
                loss_life = False
                if 'loss_life' in info and info['loss_life']:
                    loss_life = info['loss_life']

                if done:  # Actual end of an episode
                    # This final agent.step() will ensure the done state and final reward will be seen by the agent and the trackers
                    timestep_t = util.TimeStep(
                        observation=observation,
                        reward=reward,
                        done=True,
                        first=False,
                        info=info,
                    )
                    unused_a = train_agent.step(timestep_t)  # noqa: F841
                    # Save checkpoint
                    filename = "./checkpoints/DQN_" + str(iteration) + ".ckpt"
                    torch.save(network.state_dict(), filename)
                    break

if __name__ == '__main__':
    app.run(main)
