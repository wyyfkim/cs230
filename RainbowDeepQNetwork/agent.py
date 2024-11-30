import copy
from typing import Callable, Tuple
import torch
from torch import nn
from typing import NamedTuple, Text, Mapping, Optional
import numpy as np
from lib import replay_lib as replay_lib, util
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
Action = int

class Extra(NamedTuple):
    target: Optional[torch.Tensor]

def slice_with_actions(embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    batch_size, action_dim = embeddings.shape[:2]

    # Values are the 'values' in a sparse tensor we will be setting
    act_idx = actions[:, None]

    values = torch.reshape(torch.ones(actions.shape, dtype=torch.int8, device=actions.device), [-1])

    # Create a range for each index into the batch
    act_range = torch.arange(0, batch_size, dtype=torch.int64)[:, None].to(device=actions.device)
    # Combine this into coordinates with the action indices
    indices = torch.concat([act_range, act_idx], 1)

    # Needs transpose indices before adding to torch.sparse_coo_tensor.
    actions_mask = torch.sparse_coo_tensor(indices.t(), values, [batch_size, action_dim])
    with torch.no_grad():
        actions_mask = actions_mask.to_dense().bool()

    sliced_emb = torch.masked_select(embeddings, actions_mask[..., None])
    # Make sure shape is the same as embeddings
    sliced_emb = sliced_emb.reshape(embeddings.shape[0], -1)
    return sliced_emb


def l2_project(z_p: torch.Tensor, p: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    # Extract vmin and vmax and construct helper tensors from z_q
    vmin, vmax = z_q[0], z_q[-1]
    d_pos = torch.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = torch.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # Clip z_p to be in new support range (vmin, vmax).
    z_p = torch.clamp(z_p, min=vmin, max=vmax)[:, None, :]  # B x 1 x Kp

    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = torch.where(d_neg > 0, 1.0 / d_neg, torch.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = torch.where(d_pos > 0, 1.0 / d_pos, torch.zeros_like(d_pos))  # 1 x Kq x 1

    delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = (delta_qp >= 0.0).to(dtype=p.dtype)  # B x Kq x Kp

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1.0 - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return torch.sum(torch.clamp(1.0 - delta_hat, min=0.0, max=1.0) * p, 2)

def categorical_dist_double_qlearning(
    atoms_tm1: torch.Tensor,
    logits_q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    atoms_t: torch.Tensor,
    logits_q_t: torch.Tensor,
    q_t_selector: torch.Tensor,
) -> util.LossOutput:
    # Categorical distributional double Q-learning op.
    # Scale and shift time-t distribution atoms by discount and reward.
    target_z = r_t[:, None] + discount_t[:, None] * atoms_t[None, :]

    # Convert logits to distribution, then find greedy policy action in
    # state s_t.
    q_t_probs = F.softmax(logits_q_t, dim=-1)
    pi_t = torch.argmax(q_t_selector, dim=1)
    # Compute distribution for greedy action.
    p_target_z = slice_with_actions(q_t_probs, pi_t)

    # Project using the Cramer distance
    with torch.no_grad():
        target_tm1 = l2_project(target_z, p_target_z, atoms_tm1)

    logit_qa_tm1 = slice_with_actions(logits_q_tm1, a_tm1)

    loss = F.cross_entropy(input=logit_qa_tm1, target=target_tm1, reduction='none')

    return util.LossOutput(loss, Extra(target_tm1))

class RainbowDqn(util.Agent):
    """RainbowDQN agent"""

    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        atoms: torch.Tensor,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        replay: replay_lib.PrioritizedReplay,
        transition_accumulator: replay_lib.TransitionAccumulator,
        exploration_epsilon: Callable[[int], float],
        learn_interval: int,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        n_step: int,
        discount: float,
        device: torch.device,
    ):
        self.agent_name = 'RainbowDQN'
        self._device = device
        # Online Q network
        self._online_network = network.to(device=self._device) #the Q network we want to optimize.
        self._optimizer = optimizer
        # Target Q network
        self._target_network = copy.deepcopy(self._online_network).to(device=self._device)
        # Disable autograd for target network
        for p in self._target_network.parameters():
            p.requires_grad = False
        # Experience replay parameters
        self._transition_accumulator = transition_accumulator #external helper class to build n-step transition.
        self._batch_size = batch_size #sample batch size.
        self._replay = replay # experience replay buffer
        self._max_seen_priority = 1.0

        # Learning related parameters
        self._discount = discount #gamma discount for future rewards.
        self._n_step = n_step
        self._min_replay_size = min_replay_size
        self._learn_interval = learn_interval
        self._target_net_update_interval = target_net_update_interval
        self._clip_grad = True
        self._max_grad_norm = 10.0 #Max gradients norm when do gradients clip.
        # Categorical DQN parameters
        self._num_atoms = atoms.size(0)
        self._atoms = atoms.to(device=self._device)  # the support vector for probability distribution
        self._v_min = self._atoms[0]
        self._v_max = self._atoms[-1]
        self._delta_z = (self._v_max - self._v_min) / float(self._num_atoms - 1)
        # Counters and stats
        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._loss_t = np.nan

    def step(self, timestep: util.TimeStep) -> Action:
        """Given current timestep, do a action selection and a series of learn related activities"""
        self._step_t += 1
        a_t = self._choose_action(timestep)
        # a_t = self._choose_action(timestep, self._exploration_epsilon(self._step_t))
        # Try build transition and add into replay
        for transition in self._transition_accumulator.step(timestep, a_t):
            self._replay.add(transition, priority=self._max_seen_priority)
        if self._replay.size < self._min_replay_size:
            return a_t
        if self._step_t % self._learn_interval == 0:
            self._learn()
        return a_t

    def reset(self):
        """This method should be called at the beginning of every episode."""
        self._transition_accumulator.reset()

    @torch.no_grad()
    def _choose_action(self, timestep: util.TimeStep) -> Action:
        s_t = torch.from_numpy(timestep.observation[None, ...]).to(device=self._device, dtype=torch.float32)
        q_values = self._online_network(s_t).q_values
        a_t = torch.argmax(q_values, dim=-1)
        return a_t.cpu().item()

    def _learn(self) -> None:
        transitions, indices, weights = self._replay.sample(self._batch_size)
        weights = torch.from_numpy(weights).to(device=self._device, dtype=torch.float32)  # [batch_size]
        self._optimizer.zero_grad()
        loss, priorities = self._calc_loss(transitions)
        # Multiply loss by sampling weights, averaging over batch dimension
        loss = torch.mean(loss * weights.detach())
        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(self._online_network.parameters(), self._max_grad_norm,
                                           error_if_nonfinite=True)
        self._optimizer.step()
        self._update_t += 1

        # For logging only.
        self._loss_t = loss.detach().cpu().item()

        # Update target network parameters
        if self._update_t > 1 and self._update_t % self._target_net_update_interval == 0:
            self._target_network.load_state_dict(self._online_network.state_dict())
            self._target_update_t += 1

        if priorities.shape != (self._batch_size,):
            raise RuntimeError(f'Expect priorities has shape {self._batch_size}, got {priorities.shape}')
        priorities = np.clip(np.abs(priorities), 0.0, 100.0)  # np.abs(priorities)
        self._max_seen_priority = np.max([self._max_seen_priority, np.max(priorities)])
        self._replay.update_priorities(indices, priorities)
        #reset noise
        self._online_network.reset_noise()
        self._target_network.reset_noise()

    def _calc_loss(self, transitions: replay_lib.Transition) -> Tuple[torch.Tensor, np.ndarray]:
        """Calculate loss for a given batch of transitions."""
        s_tm1 = torch.from_numpy(transitions.s_tm1).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        a_tm1 = torch.from_numpy(transitions.a_tm1).to(device=self._device, dtype=torch.int64)  # [batch_size]
        r_t = torch.from_numpy(transitions.r_t).to(device=self._device, dtype=torch.float32)  # [batch_size]
        s_t = torch.from_numpy(transitions.s_t).to(device=self._device, dtype=torch.float32)  # [batch_size, state_shape]
        done = torch.from_numpy(transitions.done).to(device=self._device, dtype=torch.bool)  # [batch_size]

        discount_t = (~done).float() * self._discount

        # Compute predicted q values distribution for s_tm1, using online Q network
        logits_q_tm1 = self._online_network(s_tm1).q_logits  # [batch_size, action_dim, num_atoms]

        # Compute predicted q values distribution for s_t, using target Q network and double Q
        with torch.no_grad():
            q_t_selector = self._online_network(s_t).q_values  # [batch_size, action_dim]
            target_logits_q_t = self._target_network(s_t).q_logits  # [batch_size, action_dim, num_atoms]

        # Calculate categorical distribution q loss.
        loss = categorical_dist_double_qlearning(
            self._atoms,
            logits_q_tm1,
            a_tm1,
            r_t,
            discount_t,
            self._atoms,
            target_logits_q_t,
            q_t_selector,
        ).loss
        # Use loss as priorities.
        priorities = loss.detach().cpu().numpy()
        return loss, priorities
