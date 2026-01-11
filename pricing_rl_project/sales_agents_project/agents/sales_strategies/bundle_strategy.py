"""
Stratégie de bundling (packs produits)
"""

import numpy as np
from typing import Dict, List

class BundleStrategyAgent:
    """Agent pour stratégie de bundles"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.strategy_type = "bundle"
    
    def decide_price(self, product_id: str, bundle_products: List[str] = None) -> Dict:
        """Stratégie bundles"""
        product = self.db.get_product(product_id)
        current_price = product['current_price']
        
        # Si bundle, calculer prix bundle
        if bundle_products and len(bundle_products) > 0:
            bundle_price = self._calculate_bundle_price(bundle_products)
            discount = 0.15  # 15% discount sur bundle
            
            new_price = bundle_price * (1 - discount)
            strategy = "bundle_discount"
            explanation = f"Prix bundle ({len(bundle_products)} produits) avec {discount*100}% réduction"
        else:
            # Prix normal avec incitation bundle
            new_price = current_price * 0.97  # Léger discount
            strategy = "bundle_incentive"
            explanation = "Prix incitatif pour encourager achat bundle"
        
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': ((new_price - current_price) / current_price) * 100,
            'confidence': 0.83,
            'strategy': strategy,
            'explanation': explanation
        }
    
    def _calculate_bundle_price(self, product_ids: List[str]) -> float:
        """Calculer prix total bundle"""
        total = 0
        for pid in product_ids:
            product = self.db.get_product(pid)
            if product:
                total += product['current_price']
        return total
    
    def _validate_price(self, product_id: str, price: float) -> float:
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)