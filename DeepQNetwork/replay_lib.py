from typing import Any, NamedTuple, Callable, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
import numpy as np
import snappy
from DeepQNetwork import util

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


class Transition(NamedTuple):
    """A full transition for general use case"""

    s_tm1: Optional[np.ndarray]
    a_tm1: Optional[int]
    r_t: Optional[float]
    s_t: Optional[np.ndarray]
    done: Optional[bool]


TransitionStructure = Transition(s_tm1=None, a_tm1=None, r_t=None, s_t=None, done=None)


class UniformReplay(Generic[ReplayStructure]):
    """Uniform replay, with circular buffer storage for flat named tuples."""

    def __init__(
        self,
        capacity: int,
        structure: ReplayStructure,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        encoder: Optional[Callable[[ReplayStructure], Any]] = None,
        decoder: Optional[Callable[[Any], ReplayStructure]] = None,
    ):
        self.structure = structure
        self._capacity = capacity
        self._random_state = random_state
        self._storage = [None] * capacity
        self._num_added = 0
        self._encoder = encoder or (lambda s: s)
        self._decoder = decoder or (lambda s: s)

    def add(self, item: ReplayStructure) -> None:
        """Adds single item to replay."""
        self._storage[self._num_added % self._capacity] = self._encoder(item)
        self._num_added += 1

    def get(self, indices: Sequence[int]) -> List[ReplayStructure]:
        """Retrieves items by indices."""
        return [self._decoder(self._storage[i]) for i in indices]

    def sample(self, batch_size: int) -> ReplayStructure:
        """Samples batch of items from replay uniformly, with replacement."""
        if self.size < batch_size:
            raise RuntimeError(f'Replay only have {self.size} samples, got sample batch size {batch_size}')

        indices = self._random_state.randint(self.size, size=batch_size)
        samples = self.get(indices)
        return np_stack_list_of_transitions(samples, self.structure)

    @property
    def size(self) -> int:
        """Number of items currently contained in replay."""
        return min(self._num_added, self._capacity)

    def reset(self) -> None:
        """Reset the state of replay, should be called at the beginning of every episode"""
        self._num_added = 0


class TransitionAccumulator:
    """Accumulates timesteps to form transitions."""

    def __init__(self):
        self._timestep_tm1 = None
        self._a_tm1 = None

    def step(self, timestep_t: util.TimeStep, a_t: int) -> Iterable[Transition]:
        """Accumulates timestep and resulting action, maybe yield a transition.

        We only need the s_t, r_t, and done flag for a given timestep_t
        the first timestep yield nothing since we don't have a full transition

        if the given timestep_t transition is terminal state, we need to reset the state of the accumulator,
        so the next timestep which is the start of a new episode yields nothing
        """
        if timestep_t.first:
            self.reset()

        if self._timestep_tm1 is None:
            if not timestep_t.first:
                raise ValueError(f'Expected first timestep, got {str(timestep_t)}')
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            return  # Empty iterable.

        transition = Transition(
            s_tm1=self._timestep_tm1.observation,
            a_tm1=self._a_tm1,
            r_t=timestep_t.reward,
            s_t=timestep_t.observation,
            done=timestep_t.done,
        )
        self._timestep_tm1 = timestep_t
        self._a_tm1 = a_t
        yield transition

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be `FIRST`."""
        self._timestep_tm1 = None
        self._a_tm1 = None


def np_stack_list_of_transitions(transitions, structure, axis=0):
    """
    Stack list of transition objects into one transition object with lists of tensors
    on a given dimension (default 0)
    """

    transposed = zip(*transitions)
    stacked = [np.stack(xs, axis=axis) for xs in transposed]
    return type(structure)(*stacked)
