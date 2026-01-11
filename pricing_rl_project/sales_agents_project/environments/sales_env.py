"""
Environnement principal de vente pour RL
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime

class SalesEnvironment(gym.Env):
    """Environnement de vente e-commerce"""
    
    def __init__(self, product_id: str = None):
        super().__init__()
        
        self.product_id = product_id or "PROD_001"
        self.max_steps = 100
        self.current_step = 0
        
        # Espaces
        self.action_space = gym.spaces.Discrete(7)  # Stratégies de vente
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # Initialiser
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_sales = 0
        self.total_revenue = 0
        
        # État initial
        state = self._generate_initial_state()
        return state, {}
    
    def step(self, action: int):
        # Simuler vente
        strategy = self._action_to_strategy(action)
        sale_success, revenue = self._simulate_sale(strategy)
        
        # Mettre à jour
        if sale_success:
            self.total_sales += 1
            self.total_revenue += revenue
        
        # Récompense
        reward = self._calculate_reward(sale_success, revenue, strategy)
        
        # Nouvel état
        next_state = self._get_next_state(sale_success)
        
        # Terminaison
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Info
        info = {
            'step': self.current_step,
            'sale_success': sale_success,
            'revenue': revenue,
            'strategy': strategy,
            'total_sales': self.total_sales,
            'total_revenue': self.total_revenue
        }
        
        return next_state, reward, terminated, False, info
    
    def _generate_initial_state(self):
        """Générer état initial"""
        return np.random.uniform(-1, 1, size=(10,)).astype(np.float32)
    
    def _action_to_strategy(self, action: int) -> str:
        strategies = [
            'low_price', 'medium_price', 'high_price',
            'bundle', 'cross_sell', 'up_sell', 'discount'
        ]
        return strategies[action]
    
    def _simulate_sale(self, strategy: str) -> Tuple[bool, float]:
        """Simuler résultat vente"""
        # Probabilité succès par stratégie
        success_rates = {
            'low_price': 0.8,
            'medium_price': 0.6,
            'high_price': 0.4,
            'bundle': 0.7,
            'cross_sell': 0.5,
            'up_sell': 0.3,
            'discount': 0.9
        }
        
        # Revenu par stratégie
        revenues = {
            'low_price': 50,
            'medium_price': 100,
            'high_price': 200,
            'bundle': 150,
            'cross_sell': 80,
            'up_sell': 120,
            'discount': 40
        }
        
        success_rate = success_rates.get(strategy, 0.5)
        success = np.random.random() < success_rate
        
        return success, revenues[strategy] if success else 0
    
    def _calculate_reward(self, success: bool, revenue: float, strategy: str) -> float:
        """Calculer récompense"""
        if success:
            base_reward = revenue / 10.0
            
            # Bonus stratégie
            strategy_bonus = {
                'up_sell': 2.0,
                'bundle': 1.5,
                'cross_sell': 1.0
            }.get(strategy, 0.0)
            
            return base_reward + strategy_bonus
        else:
            return -0.5  # Pénalité échec
    
    def _get_next_state(self, sale_success: bool):
        """Générer prochain état"""
        state = np.random.uniform(-1, 1, size=(10,)).astype(np.float32)
        
        # Ajouter info succès/échec
        state[0] = 1.0 if sale_success else -1.0
        
        return state