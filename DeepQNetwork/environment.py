import cv2
import numpy as np
import gym
from gym.spaces import Box
from collections import deque
from stable_baselines3.common import atari_wrappers
from gym.wrappers import AtariPreprocessing

class LazyFrames(object):
    """This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was."""

    def __init__(self, frames):
        self.dtype = frames[0].dtype
        self.shape = (frames[0].shape[0], frames[0].shape[1], len(frames))
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(shape[:-1] + (shape[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ObservationChannelFirst(gym.ObservationWrapper):
    """Make observation image channel first, this is for PyTorch only."""

    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        _low, _high = (0.0, 255)
        new_dtype = env.observation_space.dtype
        self.observation_space = Box(low=_low, high=_high, shape=new_shape, dtype=new_dtype)

    def observation(self, obs):
        # permute [H, W, C] array to in the range [C, H, W]
        # return np.transpose(observation, axes=(2, 0, 1)).astype(self.observation_space.dtype)
        obs = np.asarray(obs, dtype=self.observation_space.dtype).transpose(2, 0, 1)
        # make sure it's C-contiguous for compress state
        return np.ascontiguousarray(obs, dtype=self.observation_space.dtype)

class ResizeAndGrayscaleFrame(gym.ObservationWrapper):
    """
    Resize frames to 84x84, and grascale image as done in the Nature paper.
    """

    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)

        assert self.observation_space.dtype == np.uint8 and len(self.observation_space.shape) == 3

        self.frame_width = width
        self.frame_height = height
        self.grayscale = grayscale
        num_channels = 1 if self.grayscale else 3

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.frame_height, self.frame_width, num_channels),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # pylint: disable=no-member

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
        # pylint: disable=no-member

        if self.grayscale:
            obs = np.expand_dims(obs, -1)

        return obs

def create_atari_environment(
    env_name: str,
    frame_stack: int = 4,
    noop_max: int = 30,
) -> gym.Env:
    # name =f'{env_name}NoFrameskip-v4'
    # env = gym.make(name)
    env = gym.make('ALE/DemonAttack-v5')
    env = ResizeAndGrayscaleFrame(env, width=84, height=84)

    print(env)
    # env = atari_wrappers.AtariWrapper(env)
    #
    # # env = atari_wrappers.AtariWrapper(env, clip_reward=True, noop_max=noop_max, terminal_on_life_loss=False)
    #
    # #stack n last frames.
    env = FrameStack(env, frame_stack)
    # # change observation image from shape [H, W, C] to in the range [C, H, W]
    env = ObservationChannelFirst(env)

    return env
