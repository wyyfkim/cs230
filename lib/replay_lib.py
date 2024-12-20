from typing import Any, NamedTuple, Callable, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
import numpy as np
import snappy
import collections
from lib import util

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
def compress(array: np.ndarray) -> CompressedArray:
    return snappy.compress(array), array.shape, array.dtype
def uncompress(compressed: CompressedArray) -> np.ndarray:
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compressed_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)
def _build_n_step_transition(transitions: Iterable[Transition], discount: float) -> Transition:
    r_t = 0.0
    discount_t = 1.0
    for transition in transitions:
        r_t += discount_t * transition.r_t
        discount_t *= discount

    return Transition(
        s_tm1=transitions[0].s_tm1,
        a_tm1=transitions[0].a_tm1,
        r_t=r_t,
        s_t=transitions[-1].s_t,
        done=transitions[-1].done,
    )

class PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        structure: ReplayStructure,
        priority_exponent: float,
        importance_sampling_exponent: float,
        random_state: np.random.RandomState,
        normalize_weights: bool = True,
        time_major: bool = False,
        encoder: Optional[Callable[[ReplayStructure], Any]] = None,
        decoder: Optional[Callable[[Any], ReplayStructure]] = None,
    ):
        if capacity <= 0:
            raise ValueError(f'Expect capacity to be a positive integer, got {capacity}')
        self.structure = structure
        self._capacity = capacity
        self._random_state = random_state
        self._encoder = encoder or (lambda s: s)
        self._decoder = decoder or (lambda s: s)

        self._storage = [None] * capacity
        self._num_added = 0

        self._time_major = time_major

        self._priorities = np.ones((capacity,), dtype=np.float32)
        self._priority_exponent = priority_exponent
        self._importance_sampling_exponent = importance_sampling_exponent

        self._normalize_weights = normalize_weights

    def add(self, item: ReplayStructure, priority: float) -> None:
        """Adds a single item with a given priority to the replay buffer."""
        if not np.isfinite(priority) or priority < 0.0:
            raise ValueError('priority must be finite and positive.')

        index = self._num_added % self._capacity
        self._priorities[index] = priority
        self._storage[index] = self._encoder(item)
        self._num_added += 1

    def get(self, indices: Sequence[int]) -> Iterable[ReplayStructure]:
        """Retrieves transitions by indices."""
        return [self._decoder(self._storage[i]) for i in indices]

    def sample(self, size: int) -> Tuple[ReplayStructure, np.ndarray, np.ndarray]:
        """Samples a batch of transitions."""
        if self.size < size:
            raise RuntimeError(f'Replay only have {self.size} samples, got sample size {size}')

        if self._priority_exponent == 0:
            indices = self._random_state.uniform(0, self.size, size=size).astype(np.int64)
            weights = np.ones_like(indices, dtype=np.float32)
        else:
            # code copied from seed_rl
            priorities = self._priorities[: self.size] ** self._priority_exponent

            probs = priorities / np.sum(priorities)
            indices = self._random_state.choice(np.arange(probs.shape[0]), size=size, replace=True, p=probs)

            # Importance weights.
            weights = ((1.0 / self.size) / np.take(probs, indices)) ** self.importance_sampling_exponent

            if self._normalize_weights:
                weights /= np.max(weights) + 1e-8  # Normalize.

        samples = self.get(indices)
        stacked = np_stack_list_of_transitions(samples, self.structure, self.stack_dim)
        return stacked, indices, weights

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        """Updates indices with given priorities."""
        priorities = np.asarray(priorities)
        if not np.isfinite(priorities).all() or (priorities < 0.0).any():
            raise ValueError('priorities must be finite and positive.')
        for index, priority in zip(indices, priorities):
            self._priorities[index] = priority

    @property
    def stack_dim(self) -> int:
        """Stack dimension, for RNN we may need to make the tensor time major by stacking on second dimension as [T, B, ...]."""
        if self._time_major:
            return 1
        else:
            return 0

    @property
    def size(self) -> None:
        """Number of elements currently contained in replay."""
        return min(self._num_added, self._capacity)

    @property
    def capacity(self) -> None:
        """Total capacity of replay (maximum number of items that can be stored)."""
        return self._capacity

    @property
    def importance_sampling_exponent(self):
        """Importance sampling exponent at current step."""
        return self._importance_sampling_exponent(self._num_added)

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

class NStepTransitionAccumulator:
    def __init__(self, n, discount):
        self._discount = discount
        self._transitions = collections.deque(maxlen=n)  # Store 1-step transitions.
        self._timestep_tm1 = None
        self._a_tm1 = None

    def step(self, timestep_t: util.TimeStep, a_t: int) -> Iterable[Transition]:
        """Accumulates timestep and resulting action, yields transitions."""
        if timestep_t.first:
            self.reset()

        # There are no transitions on the first timestep.
        if self._timestep_tm1 is None:
            assert self._a_tm1 is None
            if not timestep_t.first:
                raise ValueError(f'Expected first timestep, got {str(timestep_t)}')
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            return  # Empty iterable.

        self._transitions.append(
            Transition(
                s_tm1=self._timestep_tm1.observation,
                a_tm1=self._a_tm1,
                r_t=timestep_t.reward,
                s_t=timestep_t.observation,
                done=timestep_t.done,
            )
        )

        self._timestep_tm1 = timestep_t
        self._a_tm1 = a_t

        if timestep_t.done:
            # Yield any remaining n, n-1, ..., 1-step transitions at episode end.
            while self._transitions:
                yield _build_n_step_transition(self._transitions, self._discount)
                self._transitions.popleft()
        else:
            # Wait for n transitions before yielding anything.
            if len(self._transitions) < self._transitions.maxlen:
                return  # Empty iterable.

            assert len(self._transitions) == self._transitions.maxlen

            # This is the typical case, yield a single n-step transition.
            yield _build_n_step_transition(self._transitions, self._discount)

    def reset(self) -> None:
        """Resets the accumulator. Following timestep is expected to be FIRST."""
        self._transitions.clear()
        self._timestep_tm1 = None
        self._a_tm1 = None

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

