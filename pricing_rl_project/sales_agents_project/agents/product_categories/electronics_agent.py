"""
Agent spécialisé pour produits électroniques
"""

import numpy as np
from typing import Dict

class ElectronicsAgent:
    """Agent pour produits électroniques (obsolescence rapide)"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.strategy_type = "electronics"
    
    def decide_price(self, product_id: str) -> Dict:
        """Stratégie spécifique électronique"""
        product = self.db.get_product(product_id)
        current_price = product['current_price']
        
        # Vérifier ancienneté produit
        days_since_update = self._days_since_update(product_id)
        
        # Baisser prix si ancien
        if days_since_update > 180:  # > 6 mois
            price_multiplier = 0.85
            strategy = "clearance"
        elif days_since_update > 90:
            price_multiplier = 0.92
            strategy = "promotion"
        else:
            price_multiplier = 1.05  # Nouveau = premium
            strategy = "new_product"
        
        new_price = current_price * price_multiplier
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': ((new_price - current_price) / current_price) * 100,
            'confidence': 0.80,
            'strategy': strategy,
            'explanation': f'Produit électronique (ancienneté: {days_since_update} jours)'
        }
    
    def _days_since_update(self, product_id: str) -> int:
        """Calculer jours depuis dernière mise à jour"""
        query = """
            SELECT DATEDIFF(NOW(), MAX(updated_at)) as days 
            FROM products 
            WHERE product_id = %s
        """
        result = self.db._execute_query(query, (product_id,), fetch_one=True)
        return result['days'] if result else 0
    
    def _validate_price(self, product_id: str, price: float) -> float:
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)