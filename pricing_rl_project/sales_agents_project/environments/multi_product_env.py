"""
Environnement multi-produits pour cross-selling
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, List

class MultiProductEnvironment(gym.Env):
    """Environnement pour vente multi-produits"""
    
    def __init__(self, n_products=3):
        super().__init__()
        
        self.n_products = n_products
        self.max_steps = 50
        self.current_step = 0
        
        # Espaces
        self.action_space = gym.spaces.Discrete(3)  # Produit A, B, C
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_products * 3,), dtype=np.float32
        )
        
        # Produits
        self.products = []
        self._initialize_products()
        
        self.reset()
    
    def _initialize_products(self):
        """Initialiser produits"""
        for i in range(self.n_products):
            self.products.append({
                'id': f'PROD_{i+1}',
                'base_price': np.random.uniform(50, 200),
                'cost': np.random.uniform(20, 100),
                'stock': np.random.randint(50, 200),
                'popularity': np.random.uniform(0.3, 0.9),
                'compatibility': np.random.uniform(0, 1, size=self.n_products)
            })
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.sales_history = []
        
        state = self._get_state()
        return state, {}
    
    def step(self, action: int):
        # Sélectionner produit
        product_idx = action % self.n_products
        product = self.products[product_idx]
        
        # Simuler vente
        sale_success, quantity = self._simulate_sale(product_idx)
        
        if sale_success:
            # Mettre à jour stock
            product['stock'] -= quantity
            
            # Calculer revenue
            revenue = quantity * product['base_price']
            
            # Cross-sell possible ?
            cross_sell_revenue = self._try_cross_sell(product_idx)
            revenue += cross_sell_revenue
        else:
            revenue = 0
        
        # Récompense
        reward = self._calculate_reward(sale_success, revenue, product_idx)
        
        # Prochain état
        next_state = self._get_state()
        
        # Terminaison
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'product_sold': product['id'] if sale_success else None,
            'quantity': quantity if sale_success else 0,
            'revenue': revenue,
            'cross_sell': cross_sell_revenue > 0
        }
        
        return next_state, reward, terminated, False, info
    
    def _get_state(self):
        """État multi-produits"""
        state = []
        
        for product in self.products:
            # Features par produit
            stock_ratio = product['stock'] / 200.0
            price_ratio = product['base_price'] / 200.0
            
            state.extend([
                stock_ratio * 2 - 1,
                price_ratio * 2 - 1,
                product['popularity'] * 2 - 1
            ])
        
        return np.array(state, dtype=np.float32)
    
    def _simulate_sale(self, product_idx: int) -> Tuple[bool, int]:
        """Simuler vente produit"""
        product = self.products[product_idx]
        
        # Probabilité vente basée sur popularité
        success_prob = product['popularity'] * 0.8
        success = np.random.random() < success_prob
        
        if success:
            quantity = np.random.randint(1, 3)
            return True, quantity
        else:
            return False, 0
    
    def _try_cross_sell(self, main_product_idx: int) -> float:
        """Tenter cross-sell"""
        main_product = self.products[main_product_idx]
        revenue = 0
        
        # Vérifier compatibilité avec autres produits
        for i, compat in enumerate(main_product['compatibility']):
            if i != main_product_idx and compat > 0.5:
                # Probabilité cross-sell
                if np.random.random() < compat * 0.5:
                    product = self.products[i]
                    revenue += product['base_price'] * 0.7  # Discount cross-sell
        
        return revenue
    
    def _calculate_reward(self, success: bool, revenue: float, product_idx: int) -> float:
        if success:
            product = self.products[product_idx]
            profit = revenue - (product['cost'] * (revenue / product['base_price']))
            return profit / 10.0
        else:
            return -0.2