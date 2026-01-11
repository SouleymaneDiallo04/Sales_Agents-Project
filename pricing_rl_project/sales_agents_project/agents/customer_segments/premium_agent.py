"""
Agent pour clients premium (moins sensibles au prix)
"""

import numpy as np
from typing import Dict

class PremiumAgent:
    """Agent pour clients recherchant qualité et service"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.strategy_type = "premium"
    
    def decide_price(self, product_id: str, customer_profile: Dict = None) -> Dict:
        """Stratégie premium: prix plus élevés, valeur ajoutée"""
        product = self.db.get_product(product_id)
        current_price = product['current_price']
        
        # Clients premium acceptent prix plus élevés
        if customer_profile and customer_profile.get('loyalty', 0) > 0.6:
            price_multiplier = 1.1  + np.random.uniform(0, 0.05)
        else:
            price_multiplier = 1.05
        
        new_price = current_price * price_multiplier
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': ((new_price - current_price) / current_price) * 100,
            'confidence': 0.75,
            'strategy': 'value_based_pricing',
            'explanation': 'Positionnement premium avec valeur ajoutée'
        }
    
    def _validate_price(self, product_id: str, price: float) -> float:
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)