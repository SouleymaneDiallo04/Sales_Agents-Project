"""
Stratégie aggressive de pricing
"""

import numpy as np
from typing import Dict, Any, Optional

class AggressivePricingAgent:
    """Agent pour stratégie aggressive (conquête marché)"""
    
    def __init__(self, db_connection=None, agent_id=None, name=None):
        # Compatibilité avec les deux interfaces
        if db_connection is not None:
            self.db = db_connection
        else:
            self.db = None  # Sera défini plus tard si nécessaire
            
        self.agent_id = agent_id or "aggressive_agent"
        self.name = name or "Aggressive Pricing Agent"
        self.strategy_type = "aggressive"
    
    def propose_price(self, product: Dict[str, Any], market_context: Dict[str, Any]) -> float:
        """Propose un prix selon stratégie aggressive"""
        current_price = product.get('current_price', product.get('price', 100.0))
        cost_price = product.get('cost_price', current_price * 0.7)
        
        # Prix concurrent minimum depuis market_context
        competitors = market_context.get('competitors', [])
        comp_prices = [c.get('price', c.get('competitor_price', current_price)) for c in competitors]
        min_competitor = min(comp_prices) if comp_prices else current_price
        
        # Sous-coter agressivement (mais pas à perte)
        target_price = min_competitor * 0.95  # -5% vs concurrent
        
        # Ne pas vendre à perte
        min_acceptable = cost_price * 1.05  # Marge 5% minimum
        
        new_price = max(target_price, min_acceptable)
        
        # Validation des limites
        min_price = product.get('min_price', cost_price)
        max_price = product.get('max_price', current_price * 2)
        new_price = np.clip(new_price, min_price, max_price)
        
        return float(new_price)
    
    def calculate_success_probability(self, product: Dict[str, Any], market_context: Dict[str, Any]) -> float:
        """Calcule la probabilité de succès pour cette stratégie"""
        # La stratégie agressive a une probabilité de succès moyenne
        # basée sur la compétitivité du marché
        competitors = market_context.get('competitors', [])
        if not competitors:
            return 0.7  # Bonne probabilité si pas de concurrents
            
        current_price = product.get('current_price', product.get('price', 100.0))
        comp_prices = [c.get('price', c.get('competitor_price', current_price)) for c in competitors]
        avg_competitor = sum(comp_prices) / len(comp_prices)
        
        # Plus le prix est bas par rapport aux concurrents, plus la probabilité est élevée
        price_ratio = current_price / avg_competitor if avg_competitor > 0 else 1.0
        base_probability = 0.6
        
        if price_ratio < 0.95:  # Prix plus bas que concurrents
            base_probability += 0.2
        elif price_ratio > 1.05:  # Prix plus élevé
            base_probability -= 0.2
            
        return np.clip(base_probability, 0.1, 0.95)
    
    def decide_price(self, product_id: str) -> Dict:
        """Stratégie aggressive: baisser prix pour gagner parts (interface legacy)"""
        if self.db is None:
            raise ValueError("Database connection required for decide_price method")
            
        product = self.db.get_product(product_id)
        competitors = self.db.get_competitor_prices(product_id)
        
        current_price = product['current_price']
        cost_price = product['cost_price']
        
        # Prix concurrent minimum
        comp_prices = [c['competitor_price'] for c in competitors]
        min_competitor = min(comp_prices) if comp_prices else current_price
        
        # Sous-coter agressivement (mais pas à perte)
        target_price = min_competitor * 0.95  # -5% vs concurrent
        
        # Ne pas vendre à perte
        min_acceptable = cost_price * 1.05  # Marge 5% minimum
        
        new_price = max(target_price, min_acceptable)
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': ((new_price - current_price) / current_price) * 100,
            'confidence': 0.88,
            'strategy': 'market_conquest',
            'explanation': f'Prix aggressive pour conquête marché (concurrent: {min_competitor}€)'
        }
    
    def _validate_price(self, product_id: str, price: float) -> float:
        if self.db is None:
            return price  # Pas de validation possible sans DB
            
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)