"""
Gymnasium wrappers for the pricing environment.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class NormalizedObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to normalize observations.
    """

    def __init__(self, env: gym.Env, mean: np.ndarray = None, std: np.ndarray = None):
        super().__init__(env)
        self.mean = mean
        self.std = std

        if self.mean is None or self.std is None:
            logger.warning("No normalization parameters provided, using identity normalization")

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize the observation."""
        if self.mean is not None and self.std is not None:
            return (observation - self.mean) / (self.std + 1e-8)
        return observation


class RewardScalingWrapper(gym.RewardWrapper):
    """
    Wrapper to scale rewards.
    """

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        """Scale the reward."""
        return reward * self.scale


class FrameStackWrapper(gym.Wrapper):
    """
    Wrapper to stack multiple observations.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack

        # Update observation space
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(obs_shape[0] * num_stack,),
            dtype=env.observation_space.dtype
        )

        self.frames = []

    def reset(self, **kwargs):
        """Reset the environment and frame stack."""
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.num_stack
        return self._get_stacked_obs(), info

    def step(self, action):
        """Step the environment and update frame stack."""
        obs, reward, done, truncated, info = self.env.step(action)

        self.frames.pop(0)
        self.frames.append(obs)

        return self._get_stacked_obs(), reward, done, truncated, info

    def _get_stacked_obs(self) -> np.ndarray:
        """Get the stacked observation."""
        return np.concatenate(self.frames, axis=0)


class ActionNoiseWrapper(gym.Wrapper):
    """
    Wrapper to add noise to actions.
    """

    def __init__(self, env: gym.Env, noise_std: float = 0.1):
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        """Add noise to the action."""
        noisy_action = action + np.random.normal(0, self.noise_std, size=action.shape)
        # Clip to action space bounds
        noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)

        return self.env.step(noisy_action)


class MonitoringWrapper(gym.Wrapper):
    """
    Wrapper for monitoring and logging.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def reset(self, **kwargs):
        """Reset and log episode statistics."""
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

        self.current_episode_reward = 0
        self.current_episode_length = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        """Step and track episode statistics."""
        obs, reward, done, truncated, info = self.env.step(action)

        self.current_episode_reward += reward
        self.current_episode_length += 1

        # Add episode statistics to info
        info['episode'] = {
            'reward': self.current_episode_reward,
            'length': self.current_episode_length,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        }

        return obs, reward, done, truncated, info


def create_wrapped_env(env: gym.Env, config: Dict[str, Any]) -> gym.Env:
    """
    Create a wrapped environment with common wrappers.
    """
    # Apply wrappers based on config
    if config.get('wrappers', {}).get('normalize_observations', False):
        env = NormalizedObservationWrapper(env)

    if config.get('wrappers', {}).get('reward_scaling'):
        scale = config['wrappers']['reward_scaling']
        env = RewardScalingWrapper(env, scale=scale)

    if config.get('wrappers', {}).get('frame_stack'):
        num_stack = config['wrappers']['frame_stack']
        env = FrameStackWrapper(env, num_stack=num_stack)

    if config.get('wrappers', {}).get('action_noise'):
        noise_std = config['wrappers']['action_noise']
        env = ActionNoiseWrapper(env, noise_std=noise_std)

    # Always add monitoring
    env = MonitoringWrapper(env)

    return env