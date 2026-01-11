"""
Environnement de négociation client
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple

class NegotiationEnvironment(gym.Env):
    """Environnement pour négociation prix"""
    
    def __init__(self):
        super().__init__()
        
        self.max_steps = 10
        self.current_step = 0
        
        # Espaces
        self.action_space = gym.spaces.Discrete(3)  # Conceder, Maintenir, Insister
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.initial_price = 100.0
        self.current_price = self.initial_price
        self.customer_willingness = np.random.uniform(80, 120)
        self.customer_patience = np.random.randint(3, 8)
        
        state = self._get_state()
        return state, {}
    
    def step(self, action: int):
        # Action de l'agent
        price_change = self._action_to_change(action)
        self.current_price *= (1 + price_change)
        
        # Réaction client
        customer_accepts = self._customer_response()
        
        # Calculer résultat
        if customer_accepts:
            sale_completed = True
            revenue = self.current_price
        else:
            sale_completed = False
            revenue = 0
        
        # Récompense
        reward = self._calculate_reward(sale_completed, revenue)
        
        # Prochain état
        next_state = self._get_state()
        
        # Terminaison
        self.current_step += 1
        terminated = self.current_step >= self.max_steps or sale_completed
        
        info = {
            'customer_accepts': customer_accepts,
            'final_price': self.current_price if sale_completed else None,
            'revenue': revenue
        }
        
        return next_state, reward, terminated, False, info
    
    def _get_state(self):
        state = np.zeros(6, dtype=np.float32)
        
        state[0] = (self.current_price / 150.0) * 2 - 1
        state[1] = (self.customer_willingness / 150.0) * 2 - 1
        state[2] = self.current_step / self.max_steps * 2 - 1
        state[3] = self.customer_patience / 10.0 * 2 - 1
        state[4] = np.random.uniform(-1, 1)  # Client humeur
        state[5] = np.random.uniform(-1, 1)  # Contexte
        
        return state
    
    def _action_to_change(self, action: int) -> float:
        changes = [-0.05, 0.0, 0.02]  # Conceder 5%, Maintenir, Insister +2%
        return changes[action]
    
    def _customer_response(self) -> bool:
        """Le client accepte-t-il ?"""
        price_ratio = self.current_price / self.customer_willingness
        
        # Probabilité d'acceptation
        if price_ratio <= 0.95:
            prob = 0.9  # Bon prix
        elif price_ratio <= 1.05:
            prob = 0.6  # Prix acceptable
        else:
            prob = 0.3  # Trop cher
        
        # Patience diminue
        patience_factor = max(0, 1 - (self.current_step / self.customer_patience))
        prob *= patience_factor
        
        return np.random.random() < prob
    
    def _calculate_reward(self, sale_completed: bool, revenue: float) -> float:
        if sale_completed:
            # Récompense basée sur marge
            cost = 50.0
            profit = revenue - cost
            return profit / 10.0
        else:
            return -1.0  # Pénalité échec