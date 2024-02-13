from skimage import transform
import gym
import numpy as np
from gym.spaces import Box

class SkipFrame(gym.Wrapper):
    """
    Wrapper that skips a fixed number of frames for each action. This can speed up training
    and can also make the problem easier for the agent by reducing the frequency of decision making.
    """
    def __init__(self, env, skip):
        # Return only every 'skip' frame
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, _, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, None, info

class ResizeObservation(gym.ObservationWrapper):
    """
    Wrapper that resizes the observation space of the environment to a given shape.
    """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        # Update the observation space to the new shape
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        # Resize and normalize observation
        resize_obs = transform.resize(observation, self.shape, anti_aliasing=True)
        return resize_obs

class GrayPermuteObservation(gym.ObservationWrapper):
    """
    Wrapper that converts the observation space of the environment to grayscale (one channel).
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        # Update the observation space to reflect the new shape (grayscale)
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):        
        # Select first channel from the observation
        observation = observation[:, :, 0]
        return observation