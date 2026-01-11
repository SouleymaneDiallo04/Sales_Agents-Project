"""
Stratégie basée sur la valeur perçue
"""

import numpy as np
from typing import Dict

class ValueBasedAgent:
    """Agent pour pricing basé sur la valeur"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.strategy_type = "value_based"
    
    def decide_price(self, product_id: str) -> Dict:
        """Pricing basé sur valeur perçue"""
        product = self.db.get_product(product_id)
        current_price = product['current_price']
        
        # Analyse valeur
        popularity = product.get('popularity_score', 0.5)
        uniqueness = self._calculate_uniqueness(product_id)
        
        # Prix basé sur valeur
        value_multiplier = 0.8 + (popularity * 0.4) + (uniqueness * 0.3)
        
        new_price = current_price * value_multiplier
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': ((new_price - current_price) / current_price) * 100,
            'confidence': 0.77,
            'strategy': 'value_pricing',
            'explanation': f'Pricing basé valeur (popularité: {popularity:.2f}, unicité: {uniqueness:.2f})'
        }
    
    def _calculate_uniqueness(self, product_id: str) -> float:
        """Calculer unicité produit"""
        # Simuler analyse compétition
        query = """
            SELECT COUNT(*) as similar_products 
            FROM products 
            WHERE category = (
                SELECT category FROM products WHERE product_id = %s
            )
        """
        result = self.db._execute_query(query, (product_id,), fetch_one=True)
        similar = result['similar_products'] if result else 10
        
        # Moins il y a de similaires, plus c'est unique
        uniqueness = max(0, 1 - (similar / 50))
        return min(1.0, uniqueness)
    
    def _validate_price(self, product_id: str, price: float) -> float:
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)