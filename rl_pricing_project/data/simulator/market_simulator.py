import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

class MarketSimulator:
    """Simulateur de marché réaliste pour l'entraînement RL"""
    
    def __init__(self, product_data: Dict[str, Any]):
        self.product_data = product_data
        self.current_date = datetime.now()
        self.market_conditions = {
            'volatility': 0.1,
            'trend': 0.0,  # tendance générale du marché
            'seasonality_amplitude': 0.3
        }
    
    def simulate_competitor_prices(self, our_price: float) -> Dict[str, float]:
        """Simule les prix des concurrents"""
        base_strategies = {
            'Amazon': {'aggressiveness': 0.8, 'reaction_time': 1},
            'CDiscount': {'aggressiveness': 0.6, 'reaction_time': 2},
            'Boulanger': {'aggressiveness': 0.4, 'reaction_time': 3}
        }
        
        competitor_prices = {}
        for name, strategy in base_strategies.items():
            # Prix de base avec variation stratégique
            base_ratio = 1.0 + np.random.normal(0, 0.05)
            
            # Effet agressivité
            if strategy['aggressiveness'] > 0.7:
                base_ratio -= 0.05  # Concurrent agressif
            
            # Variation temporelle
            day_effect = 0.02 * np.sin(2 * np.pi * self.current_date.day / 30)
            base_ratio += day_effect
            
            # Bruit aléatoire
            noise = np.random.normal(0, 0.02)
            
            final_price = our_price * (base_ratio + noise)
            competitor_prices[name] = max(final_price, self.product_data['cost_price'] * 1.05)
        
        return competitor_prices
    
    def simulate_market_events(self) -> Dict[str, Any]:
        """Simule des événements de marché aléatoires"""
        events = {
            'promotion_competitor': False,
            'stock_shortage': False, 
            'demand_spike': False,
            'market_crash': False
        }
        
        # Probabilités d'événements
        if random.random() < 0.05:  # 5% chance
            events['promotion_competitor'] = True
        
        if random.random() < 0.03:  # 3% chance  
            events['demand_spike'] = True
            
        if random.random() < 0.02:  # 2% chance
            events['stock_shortage'] = True
            
        return events
    
    def calculate_seasonality(self, date: datetime) -> float:
        """Calcule l'effet saisonnier"""
        # Pic pendant les fêtes
        if date.month == 12:  # Décembre
            return 1.4
        elif date.month == 11:  # Novembre (Black Friday)
            return 1.6
        elif date.month in [6, 7]:  # Été
            return 0.8
        else:
            return 1.0
    
    def simulate_economic_conditions(self) -> Dict[str, float]:
        """Simule les conditions économiques générales"""
        return {
            'consumer_confidence': np.random.uniform(0.7, 1.3),
            'inflation_rate': np.random.uniform(0.01, 0.05),
            'purchasing_power': np.random.uniform(0.9, 1.1)
        }
    
    def advance_time(self, days: int = 1):
        """Avance le temps de simulation"""
        self.current_date += timedelta(days=days)