from typing import NamedTuple, Text, Mapping, Iterable, Optional, Any
import numpy as np

class TimeStep(NamedTuple):
    """Environment timestep"""

    observation: Optional[np.ndarray]
    reward: Optional[float]  # reward from the environment, could be clipped or scaled
    done: Optional[bool]  # termination mark of a episode, could also be loss-of-life for Atari
    first: Optional[bool]  # first step of an episode
    info: Optional[
        Mapping[Text, Any]
    ]  # Info dictionary which contains non-clipped/unscaled reward and other information, only used by the trackers
