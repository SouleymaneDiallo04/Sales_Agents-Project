"""
Agent spécialisé pour clients sensibles au prix
"""

import numpy as np
from typing import Dict, Any
from stable_baselines3 import PPO

class PriceSensitiveAgent:
    """Agent pour clients qui cherchent les meilleurs prix"""
    
    def __init__(self, db_connection, model_path: str = None):
        self.db = db_connection
        self.model = None
        self.strategy_type = "price_sensitive"
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Charger modèle fine-tuné"""
        try:
            self.model = PPO.load(model_path)
            print(f"✅ Modèle chargé: {model_path}")
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            self.model = None
    
    def decide_price(self, product_id: str, customer_profile: Dict = None) -> Dict:
        """Prendre décision de prix pour client sensible"""
        if not self.model:
            return self._fallback_strategy(product_id)
        
        # Récupérer état actuel
        state = self._get_current_state(product_id)
        
        # Prédiction RL
        action, _ = self.model.predict(state, deterministic=True)
        
        # Calculer nouveau prix
        price_change = self._action_to_price_change(action)
        current_price = self.db.get_product_price(product_id)
        new_price = current_price * (1 + price_change)
        
        # Pour clients sensibles, privilégier les baisses
        if customer_profile and customer_profile.get('price_sensitivity', 0) > 0.7:
            # S'assurer qu'on baisse ou maintient
            if price_change > 0:
                new_price = current_price * 0.95  # Forcer baisse de 5%
        
        # Valider contraintes
        new_price = self._validate_price(product_id, new_price)
        
        return {
            'agent': self.strategy_type,
            'current_price': current_price,
            'recommended_price': new_price,
            'price_change_percent': ((new_price - current_price) / current_price) * 100,
            'confidence': 0.85,
            'strategy': 'price_competitive',
            'explanation': 'Optimisation pour clients sensibles au prix'
        }
    
    def _get_current_state(self, product_id: str) -> np.ndarray:
        """Construire état pour modèle RL"""
        product = self.db.get_product(product_id)
        competitors = self.db.get_competitor_prices(product_id)
        
        # Features
        stock_ratio = product['current_stock'] / 100.0
        price_ratio = product['current_price'] / product['max_price']
        
        # Prix concurrents
        comp_prices = [c['competitor_price'] for c in competitors]
        avg_competitor = np.mean(comp_prices) if comp_prices else product['current_price']
        
        state = np.zeros(8, dtype=np.float32)
        state[0] = np.clip(stock_ratio, 0, 1)
        state[1] = price_ratio
        state[2] = product['current_price'] / avg_competitor
        state[3] = min(comp_prices) / product['current_price'] if comp_prices else 1.0
        state[4] = 1.0  # Flag client sensible
        state[5] = 0.8  # Importance prix
        state[6] = 0.2  # Importance marge réduite
        state[7] = 0.0  # Placeholder
        
        return state
    
    def _fallback_strategy(self, product_id: str) -> Dict:
        """Stratégie de secours"""
        product = self.db.get_product(product_id)
        competitors = self.db.get_competitor_prices(product_id)
        
        # Prix minimum concurrent
        comp_prices = [c['competitor_price'] for c in competitors]
        min_competitor = min(comp_prices) if comp_prices else product['current_price']
        
        # Sous-coter légèrement
        new_price = min_competitor * 0.98
        
        return {
            'agent': self.strategy_type,
            'current_price': product['current_price'],
            'recommended_price': new_price,
            'price_change_percent': ((new_price - product['current_price']) / product['current_price']) * 100,
            'confidence': 0.65,
            'strategy': 'competition_undercut',
            'explanation': f'Sous-cotation concurrent ({min_competitor}€)'
        }
    
    def _action_to_price_change(self, action: int) -> float:
        """Mapping actions -> changements prix"""
        changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
        return changes[action % len(changes)]
    
    def _validate_price(self, product_id: str, price: float) -> float:
        """Valider prix selon contraintes"""
        product = self.db.get_product(product_id)
        min_price = product['min_price']
        max_price = product['max_price']
        return np.clip(price, min_price, max_price)