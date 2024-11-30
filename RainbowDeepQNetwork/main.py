import math
from absl import app
import numpy as np
from typing import NamedTuple, Tuple
import torch
from torch import nn
from torch.nn import functional
from absl import logging
from itertools import chain

from RainbowDeepQNetwork import agent
from lib import replay_lib as replay_lib, util, environment as gym_env


def calc_conv2d_output(h_w: Tuple, kernel_size: int = 1, stride: int = 1, pad: int = 0, dilation: int = 1) -> Tuple[
    int, int]:
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

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Only call this during initialization"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Should call this after doing backpropagation"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return functional.linear(x, weight, bias)

class C51NetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    q_logits: torch.Tensor  # use logits and log_softmax() when calculate loss to avoid log() on zero cause NaN

class RainbowDqnNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, atoms: torch.Tensor):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output liner layer
        """
        if action_dim < 1:
            raise ValueError(f'Expect action_dim to be a positive integer, got {action_dim}')
        if len(state_dim) != 3:
            raise ValueError(f'Expect state_dim to be a tuple with [C, H, W], got {state_dim}')
        if len(atoms.shape) != 1:
            raise ValueError(f'Expect atoms to be a 1D tensor, got {atoms.shape}')

        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.size(0)

        # network backbone
        self.body = ConvNet(state_dim)


        self.advantage_head = nn.Sequential(
            NoisyLinear(self.body.out_features, 512),
            nn.ReLU(),
            NoisyLinear(512, action_dim * self.num_atoms),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.body.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * self.num_atoms),
        )

        # Initialize weights.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> C51NetworkOutputs:
        x = x.float() / 255.0
        x = self.body(x)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        advantages = advantages.view(-1, self.action_dim, self.num_atoms)
        values = values.view(-1, 1, self.num_atoms)

        q_logits = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))

        q_logits = q_logits.view(-1, self.action_dim, self.num_atoms)  # [batch_size, action_dim, num_atoms]

        q_dist = F.softmax(q_logits, dim=-1)
        atoms = self.atoms[None, None, :].to(device=x.device)
        q_values = torch.sum(q_dist * atoms, dim=-1)

        return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

    def reset_noise(self) -> None:
        """Reset noisy layer"""
        # combine two lists into one: list(chain(*zip(a, b)))
        for module in list(chain(*zip(self.advantage_head.modules(), self.value_head.modules()))):
            if isinstance(module, NoisyLinear):
                module.reset_noise()

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
    exploration_epsilon_begin_value = 1.0  # Begin value of the exploration rate in e-greedy policy.
    exploration_epsilon_end_value = 0.01  # End (decayed) value of the exploration rate in e-greedy policy.
    exploration_epsilon_decay_step = int(1e6)  # Total steps (after frame skip) to decay value of the exploration rate.
    eval_exploration_epsilon = 0.01  # Fixed exploration rate in e-greedy policy for evaluation.
    min_reply_size = 50000  # Minimum replay size before learning starts.
    min_replay_size = 50000  # Minimum replay size before learning starts
    num_iterations = 100
    num_train_steps = int(5e5)  # Number of training steps (environment steps or frames) to run per iteration
    importance_sampling_exponent_begin_value = 0.4  # Importance sampling exponent begin value
    importance_sampling_exponent_end_value = 1.0  # Importance sampling exponent end value after decay
    learning_rate = 0.00025
    num_atoms= 51 #Number of elements in the support of the categorical DQN
    v_min= -10.0 #Minimum elements value in the support of the categorical DQN
    v_max=10.0#Maximum elements value in the support of the categorical DQN
    n_step=5#TD n-step bootstrap

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
    atoms = torch.linspace(v_min, v_max, num_atoms).to(device=runtime_device, dtype=torch.float32)

    # Test environment and state shape.
    obs = train_env.reset()
    print("Create C51-DQN Network")
    network = RainbowDqnNet(state_dim=state_dim, action_dim=action_dim, atoms=atoms)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.00025)

    print("create replay buffer")
    # Create prioritized transition replay
    # Note the t in the replay is not exactly aligned with the agent t.
    importance_sampling_exponent_schedule = LinearSchedule(
        begin_t=int(min_replay_size),
        end_t=(num_iterations * int(num_train_steps)),
        begin_value=importance_sampling_exponent_begin_value,
        end_value=importance_sampling_exponent_end_value,
    )
    replay = replay_lib.PrioritizedReplay(
        capacity=int(1e6),  # Maximum replay size.
        structure=replay_lib.TransitionStructure,
        priority_exponent=0.6,
        importance_sampling_exponent=importance_sampling_exponent_schedule,
        normalize_weights=True,
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

    # Create C51-DQN agent instance
    print("create train agent")
    train_agent = agent.RainbowDqn(
        network=network,
        optimizer=optimizer,
        atoms=atoms,
        # transition_accumulator=replay_lib.TransitionAccumulator(),
        transition_accumulator=replay_lib.NStepTransitionAccumulator(n=n_step, discount=0.99),
        replay=replay,
        batch_size=32,  # Sample batch size when updating the neural network.
        min_replay_size=min_reply_size,
        learn_interval=4,  # The frequency (measured in agent steps) to update parameters.
        target_net_update_interval=2500,
        # The frequency (measured in number of Q network parameter updates) to update target networks.
        n_step=n_step,
        discount=0.99,  # Discount rate.
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
                # Given current timestep, do an action selection, add it to the replay, and sample from replay buffer and learn.
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
