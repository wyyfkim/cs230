from typing import NamedTuple, Text, Mapping, Iterable, Optional, Any
import numpy as np
import abc
import torch

class TimeStep(NamedTuple):
    """Environment timestep"""

    observation: Optional[np.ndarray]
    reward: Optional[float]  # reward from the environment, could be clipped or scaled
    done: Optional[bool]  # termination mark of a episode, could also be loss-of-life for Atari
    first: Optional[bool]  # first step of an episode
    info: Optional[
        Mapping[Text, Any]
    ]  # Info dictionary which contains non-clipped/unscaled reward and other information, only used by the trackers

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

class LossOutput(NamedTuple):
    loss: torch.Tensor
    extra: Optional[NamedTuple]
