"""
Agent pour clients avec besoin urgent
"""

import numpy as np
from typing import Dict

class UrgentAgent:
    """Agent pour clients qui ont besoin rapide"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.strategy_type = "urgent"
    
    def decide_price(self, product_id: str, customer_profile: Dict = None) -> Dict:
        """Stratégie pour clients urgents"""
        product = self.db.get_product(product_id)
        current_price = product['current_price']
        
        # Clients urgents moins sensibles au prix
        urgency_bonus = 0.0
        if customer_profile and customer_profile.get('urgency', 0) > 0.7:
            urgency_bonus = 0.08  # 8% supplémentaire
        
        # Stock bas = opportunité
        stock_ratio = product['current_stock'] / 100.0
        if stock_ratio < 0.3:
            stock_premium = 0.05  # Rare = plus cher
        else:
            stock_premium = 0.0
        
        new_price = current_price * (1 + urgency_bonus + stock_premium)
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': ((new_price - current_price) / current_price) * 100,
            'confidence': 0.70,
            'strategy': 'urgency_pricing',
            'explanation': f'Prix ajusté pour besoin urgent (stock: {product["current_stock"]} unités)'
        }
    
    def _validate_price(self, product_id: str, price: float) -> float:
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)