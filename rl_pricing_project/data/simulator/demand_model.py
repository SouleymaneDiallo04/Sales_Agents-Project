import numpy as np
from typing import Dict, Any

class DemandModel:
    """Modèle de demande réaliste avec élasticité-prix"""
    
    def __init__(self, base_demand: float = 15.0, price_elasticity: float = -1.8):
        self.base_demand = base_demand
        self.price_elasticity = price_elasticity
        
    def calculate_demand(self, 
                        our_price: float, 
                        competitor_prices: Dict[str, float],
                        market_conditions: Dict[str, Any],
                        seasonality: float = 1.0) -> float:
        """Calcule la demande basée sur multiples facteurs"""
        
        # Prix moyen des concurrents
        avg_competitor_price = np.mean(list(competitor_prices.values()))
        
        # Effet prix (élasticité)
        price_ratio = our_price / avg_competitor_price
        price_effect = price_ratio ** self.price_elasticity
        
        # Effet compétition
        min_competitor_price = min(competitor_prices.values())
        competition_effect = 1.5 - (0.5 * (our_price / min_competitor_price))
        
        # Effet saisonnier
        seasonal_effect = seasonality
        
        # Conditions économiques
        economic_effect = market_conditions.get('consumer_confidence', 1.0)
        
        # Bruit aléatoire réaliste
        noise = np.random.normal(1.0, 0.15)
        
        # Calcul demande finale
        demand = (self.base_demand * price_effect * competition_effect * 
                 seasonal_effect * economic_effect * noise)
        
        return max(0, demand)
    
    def calculate_price_elasticity(self, price: float, demand: float, 
                                 previous_price: float, previous_demand: float) -> float:
        """Calcule l'élasticité-prix basée sur données historiques"""
        if previous_demand == 0 or previous_price == 0:
            return self.price_elasticity
            
        price_change = (price - previous_price) / previous_price
        demand_change = (demand - previous_demand) / previous_demand
        
        if price_change == 0:
            return self.price_elasticity
            
        elasticity = demand_change / price_change
        return elasticity