"""
Agent pour produits mode (saisonnalité forte)
"""

import numpy as np
from datetime import datetime
from typing import Dict

class FashionAgent:
    """Agent pour produits mode"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.strategy_type = "fashion"
    
    def decide_price(self, product_id: str) -> Dict:
        """Stratégie mode avec saisonnalité"""
        product = self.db.get_product(product_id)
        current_price = product['current_price']
        
        # Saisonnalité
        month = datetime.now().month
        season_factor = self._get_season_factor(month)
        
        # Stock - mode périmée vite
        stock_ratio = product['current_stock'] / 100.0
        if stock_ratio > 0.7:
            clearance_discount = 0.15  # Liquidation
        else:
            clearance_discount = 0.0
        
        new_price = current_price * season_factor * (1 - clearance_discount)
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': ((new_price - current_price) / current_price) * 100,
            'confidence': 0.78,
            'strategy': 'seasonal_pricing',
            'explanation': f'Prix ajusté saisonnalité (mois: {month}, stock: {product["current_stock"]})'
        }
    
    def _get_season_factor(self, month: int) -> float:
        """Facteur saisonnier"""
        # Ex: été = plus cher pour vêtements été
        if month in [6, 7, 8]:  # Été
            return 1.1
        elif month in [12, 1, 2]:  # Hiver
            return 1.15
        else:
            return 1.0
    
    def _validate_price(self, product_id: str, price: float) -> float:
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)