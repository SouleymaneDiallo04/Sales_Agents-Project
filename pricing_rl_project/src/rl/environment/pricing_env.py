"""
Pricing environment for reinforcement learning.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple
import logging

from ..models.state_processor import StateProcessor
from ..models.reward_calculator import RewardCalculator
from ...simulation.market_simulator import MarketSimulator

logger = logging.getLogger(__name__)


class PricingEnv(gym.Env):
    """
    Gymnasium environment for dynamic pricing using reinforcement learning.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        self.config = config or {}
        self.market_simulator = MarketSimulator(config=self.config)
        self.state_processor = StateProcessor()
        self.reward_calculator = RewardCalculator()

        # Define action and observation spaces
        # Action: price adjustment (-1.0 to 1.0, scaled to actual price range)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation: state vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_processor.state_dim,),
            dtype=np.float32
        )

        # Environment state
        self.current_step = 0
        self.max_steps = self.config.get('simulation', {}).get('steps', 1000)
        self.current_state = None
        self.last_action = None
        self.episode_reward = 0.0

        logger.info("Pricing environment initialized")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset market simulator
        self.market_simulator.reset()

        # Reset environment state
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_action = None

        # Get initial state
        market_state = self.market_simulator.get_current_state()
        self.current_state = self.state_processor.process_state(market_state)

        logger.debug("Environment reset")

        return self.current_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        self.last_action = action[0]

        # Convert action to price decision
        price_decision = self._action_to_price(action[0])

        # Apply decision to market
        market_response = self.market_simulator.step(price_decision)

        # Process new state
        new_state = self.state_processor.process_state(market_response['state'])

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            action=price_decision,
            market_response=market_response,
            previous_state=self.current_state
        )

        # Update episode reward
        self.episode_reward += reward

        # Check if episode is done
        done = self.current_step >= self.max_steps
        truncated = False  # Could add truncation logic

        # Update current state
        self.current_state = new_state

        info = {
            'step': self.current_step,
            'price_decision': price_decision,
            'market_response': market_response,
            'episode_reward': self.episode_reward
        }

        return new_state, reward, done, truncated, info

    def _action_to_price(self, action: float) -> float:
        """Convert RL action to actual price."""
        # Action is in [-1, 1], convert to price adjustment
        # This is a simple linear mapping - could be more sophisticated
        base_price = self.config.get('pricing', {}).get('base_price', 100.0)
        max_adjustment = self.config.get('pricing', {}).get('max_adjustment', 0.5)

        adjustment = action * max_adjustment
        new_price = base_price * (1 + adjustment)

        # Ensure price stays within reasonable bounds
        min_price = base_price * 0.1
        max_price = base_price * 3.0
        new_price = np.clip(new_price, min_price, max_price)

        return new_price

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Current Price: {self.market_simulator.current_price}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"Market Demand: {self.market_simulator.current_demand}")
        return None

    def close(self):
        """Clean up environment resources."""
        self.market_simulator.close()
        logger.info("Pricing environment closed")