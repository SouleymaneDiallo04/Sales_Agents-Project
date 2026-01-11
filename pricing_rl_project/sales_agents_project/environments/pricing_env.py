"""
Environnement de pricing pour fine-tuning
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

class PricingEnvironment(gym.Env):
    """Environnement spécialisé pour fine-tuning pricing"""
    
    def __init__(self):
        super().__init__()
        
        # Configuration
        self.max_steps = 30
        self.current_step = 0
        self.base_price = 100.0
        self.cost_price = 50.0
        self.current_price = self.base_price
        self.stock = 100
        
        # Espaces
        self.action_space = gym.spaces.Discrete(5)  # -10%, -5%, 0%, +5%, +10%
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_price = self.base_price
        self.stock = 100
        self.total_profit = 0
        
        state = self._get_state()
        return state, {}
    
    def step(self, action: int):
        # Appliquer changement prix
        price_change = self._action_to_change(action)
        self.current_price *= (1 + price_change)
        
        # Simuler demande
        demand = self._simulate_demand()
        sales = min(demand, self.stock)
        self.stock -= sales
        
        # Calculer profit
        revenue = sales * self.current_price
        cost = sales * self.cost_price
        profit = revenue - cost
        self.total_profit += profit
        
        # Récompense
        reward = self._calculate_reward(profit, price_change, sales)
        
        # Prochain état
        next_state = self._get_state()
        
        # Terminaison
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'price': self.current_price,
            'sales': sales,
            'profit': profit,
            'stock': self.stock,
            'price_change': price_change
        }
        
        return next_state, reward, terminated, False, info
    
    def _get_state(self):
        """État du marché"""
        state = np.zeros(8, dtype=np.float32)
        
        # Features
        state[0] = (self.stock / 100.0) * 2 - 1
        state[1] = (self.current_price / 200.0) * 2 - 1
        state[2] = np.sin(2 * np.pi * self.current_step / 30)  # Saisonnalité
        state[3] = 1.0 if self.current_step % 7 >= 5 else -1.0  # Weekend
        state[4] = np.random.uniform(-1, 1)  # Concurrence
        state[5] = np.random.uniform(-1, 1)  # Économie
        state[6] = self.total_profit / 1000.0
        state[7] = self.current_step / self.max_steps * 2 - 1
        
        return state
    
    def _action_to_change(self, action: int) -> float:
        changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
        return changes[action]
    
    def _simulate_demand(self) -> float:
        """Simuler demande avec élasticité"""
        price_ratio = self.current_price / self.base_price
        elasticity = -1.5
        price_effect = price_ratio ** elasticity
        
        base_demand = 50
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * self.current_step / 30)
        noise = np.random.normal(1.0, 0.2)
        
        return max(0, base_demand * price_effect * seasonal * noise)
    
    def _calculate_reward(self, profit: float, price_change: float, sales: int) -> float:
        """Récompense pour fine-tuning"""
        profit_reward = profit / 100.0
        change_penalty = abs(price_change) * 3.0
        sales_bonus = sales / 20.0
        
        return profit_reward - change_penalty + sales_bonus