"""
Agent pour produits maison (stables, fidélité)
"""

import numpy as np
from typing import Dict

class HomeAgent:
    """Agent pour produits maison/décoration"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.strategy_type = "home"
    
    def decide_price(self, product_id: str) -> Dict:
        """Stratégie produits maison"""
        product = self.db.get_product(product_id)
        current_price = product['current_price']
        
        # Produits maison = stabilité
        # Petits ajustements seulement
        change = np.random.uniform(-0.03, 0.03)  # Max 3% changement
        
        new_price = current_price * (1 + change)
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': change * 100,
            'confidence': 0.82,
            'strategy': 'stable_pricing',
            'explanation': 'Ajustement léger pour produits maison (stabilité)'
        }
    
    def _validate_price(self, product_id: str, price: float) -> float:
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)