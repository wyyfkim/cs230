import copy
from typing import Callable
import numpy as np
import torch
from torch import nn
import base
import abc
from typing import NamedTuple, Text, Mapping, Iterable, Optional, Any
import numpy as np
from DeepQNetwork import replay_lib as replay_lib
from DeepQNetwork import util

torch.autograd.set_detect_anomaly(True)
Action = int

class Agent(abc.ABC):
    """Agent interface."""
    agent_name: str  # agent name
    step_t: int  # runtime steps
    @abc.abstractmethod
    def step(self, timestep: util.TimeStep) -> Action:
        """Selects action given timestep and potentially learns."""
    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """
    @property
    @abc.abstractmethod
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""

class QExtra(NamedTuple):
    target: Optional[torch.Tensor]
    td_error: Optional[torch.Tensor]


def qlearning(
    q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    q_t: torch.Tensor,
) -> base.LossOutput:
    r"""Implements the Q-learning loss.

    The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
    the target `r_t + discount_t * max q_t`.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/book/ebook/node65.html).

    Args:
      q_tm1: Tensor holding Q-values for first timestep in a batch of
        transitions, shape `[B x action_dim]`.
      a_tm1: Tensor holding action indices, shape `[B]`.
      r_t: Tensor holding rewards, shape `[B]`.
      discount_t: Tensor holding discount values, shape `[B]`.
      q_t: Tensor holding Q-values for second timestep in a batch of
        transitions, shape `[B x action_dim]`.

    Returns:
      A namedtuple with fields:

      * `loss`: a tensor containing the batch of losses, shape `[B]`.
      * `extra`: a namedtuple with fields:
          * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
          * `td_error`: batch of temporal difference errors, shape `[B]`.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(q_tm1, 2, torch.float32)
    base.assert_rank_and_dtype(a_tm1, 1, torch.long)
    base.assert_rank_and_dtype(r_t, 1, torch.float32)
    base.assert_rank_and_dtype(discount_t, 1, torch.float32)
    base.assert_rank_and_dtype(q_t, 2, torch.float32)

    base.assert_batch_dimension(a_tm1, q_tm1.shape[0])
    base.assert_batch_dimension(r_t, q_tm1.shape[0])
    base.assert_batch_dimension(discount_t, q_tm1.shape[0])
    base.assert_batch_dimension(q_t, q_tm1.shape[0])

    # Q-learning op.
    # Build target and select head to update.
    with torch.no_grad():
        target_tm1 = r_t + discount_t * torch.max(q_t, dim=1)[0]
    qa_tm1 = base.batched_index(q_tm1, a_tm1)
    # B = q_tm1.shape[0]
    # qa_tm1 = q_tm1[torch.arange(0, B), a_tm1]

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(loss, QExtra(target_tm1, td_error))

class Dqn(Agent):
    """DQN agent"""

    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        replay: replay_lib.UniformReplay,
        transition_accumulator: replay_lib.TransitionAccumulator,
        exploration_epsilon: Callable[[int], float],
        learn_interval: int,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        action_dim: int,
        discount: float,
        device: torch.device,
    ):
        """
        Args:
            network: the Q network we want to optimize.
            optimizer: the optimizer for Q network.
            random_state: used to sample random actions for e-greedy policy.
            replay: experience replay.
            transition_accumulator: external helper class to build n-step transition.
            exploration_epsilon: external schedule of e in e-greedy exploration rate.
            learn_interval: the frequency (measured in agent steps) to do learning.
            target_net_update_interval: the frequency (measured in number of online Q network parameter updates)
                 to Update target network parameters.
            min_replay_size: Minimum replay size before start to do learning.
            batch_size: sample batch size.
            action_dim: number of valid actions in the environment.
            discount: gamma discount for future rewards.
            device: PyTorch runtime device.
        """

        self.agent_name = 'DQN'

        self._device = device
        self._random_state = random_state
        self._action_dim = action_dim

        # Online Q network
        self._online_network = network.to(device=self._device)
        self._optimizer = optimizer

        # Target Q network
        self._target_network = copy.deepcopy(self._online_network).to(device=self._device)
        # Disable autograd for target network
        for p in self._target_network.parameters():
            p.requires_grad = False

        # Experience replay parameters
        self._transition_accumulator = transition_accumulator
        self._batch_size = batch_size
        self._replay = replay

        # Learning related parameters
        self._discount = discount
        self._exploration_epsilon = exploration_epsilon
        self._min_replay_size = min_replay_size
        self._learn_interval = learn_interval
        self._target_net_update_interval = target_net_update_interval

        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan

    def step(self, timestep: util.TimeStep) -> Action:
        """Given current timestep, do a action selection and a series of learn related activities"""
        self._step_t += 1

        a_t = self.act(timestep)

        # Try build transition and add into replay
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._replay.add(transition)

        # Return if replay is not ready
        if self._replay.size < self._min_replay_size:
            return a_t

        # Start to learn
        if self._step_t % self._learn_interval == 0:
            self._learn()

        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

    def act(self, timestep: util.TimeStep) -> Action:
        'Given timestep, return an action.'
        a_t = self._choose_action(timestep, self.exploration_epsilon)
        return a_t

    @torch.no_grad()
    def _choose_action(self, timestep: util.TimeStep, epsilon: float) -> Action:
        if self._random_state.rand() <= epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(0, self._action_dim)
            return a_t

        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        q_values = self._online_network(s_t).q_values
        a_t = torch.argmax(q_values, dim=-1)
        return a_t.cpu().item()

    def _learn(self) -> None:
        """Sample a batch of transitions and learn."""
        transitions = self._replay.sample(self._batch_size)
        self._update(transitions)

        # Update target network parameters
        if self._update_t > 1 and self._update_t % self._target_net_update_interval == 0:
            self._update_target_network()

    def _update(self, transitions: replay_lib.Transition) -> None:
        self._optimizer.zero_grad()
        loss = self._calc_loss(transitions)
        loss.backward()
        self._optimizer.step()
        self._update_t += 1

        # For logging only.
        self._loss_t = loss.detach().cpu().item()

    def _calc_loss(self, transitions: replay_lib.Transition) -> torch.Tensor:
        """Calculate loss for a given batch of transitions."""
        s_tm1 = torch.from_numpy(transitions.s_tm1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [batch_size]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [batch_size]
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [batch_size]

        # Rank and dtype checks, note states may be images, which is rank 4.
        base.assert_rank_and_dtype(s_tm1, (2, 4), torch.float32)
        base.assert_rank_and_dtype(s_t, (2, 4), torch.float32)
        base.assert_rank_and_dtype(a_tm1, 1, torch.long)
        base.assert_rank_and_dtype(r_t, 1, torch.float32)
        base.assert_rank_and_dtype(done, 1, torch.bool)

        discount_t = (~done).float() * self._discount

        # Compute predicted q values for s_tm1, using online Q network
        q_tm1 = self._online_network(s_tm1).q_values  # [batch_size, action_dim]

        # Compute predicted q values for s_t, using target Q network
        with torch.no_grad():
            target_q_t = self._target_network(s_t).q_values  # [batch_size, action_dim]

        # Compute loss which is 0.5 * square(td_errors)
        loss = qlearning(q_tm1, a_tm1, r_t, discount_t, target_q_t).loss
        # Averaging over batch dimension
        loss = torch.mean(loss, dim=0)
        return loss

    def _update_target_network(self):
        """Copy online network parameters to target network."""
        self._target_network.load_state_dict(self._online_network.state_dict())
        self._target_update_t += 1

    @property
    def exploration_epsilon(self):
        """Call external schedule function"""
        return self._exploration_epsilon(self._step_t)

    @property
    def statistics(self):
        """Returns current agent statistics as a dictionary."""
        return {
            # 'learning_rate': self._optimizer.param_groups[0]['lr'],
            'loss': self._loss_t,
            # 'discount': self._discount,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
            'exploration_epsilon': self.exploration_epsilon,
        }
