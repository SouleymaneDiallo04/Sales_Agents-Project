"""
Sélecteur de stratégie optimale
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class StrategySelector:
    """Sélectionne la stratégie optimale basée sur contexte"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.strategy_weights = self._load_strategy_weights()
        
    def _load_strategy_weights(self) -> Dict:
        """Charger poids des stratégies"""
        
        return {
            'market_conditions': {
                'high_competition': {'aggressive': 0.8, 'value_based': 0.2, 'premium': 0.0},
                'low_competition': {'aggressive': 0.1, 'value_based': 0.4, 'premium': 0.5},
                'volatile': {'aggressive': 0.3, 'value_based': 0.5, 'premium': 0.2}
            },
            'product_lifecycle': {
                'introduction': {'aggressive': 0.6, 'value_based': 0.4, 'premium': 0.0},
                'growth': {'aggressive': 0.3, 'value_based': 0.5, 'premium': 0.2},
                'maturity': {'aggressive': 0.2, 'value_based': 0.4, 'premium': 0.4},
                'decline': {'aggressive': 0.7, 'value_based': 0.3, 'premium': 0.0}
            },
            'customer_segment': {
                'price_sensitive': {'aggressive': 0.9, 'value_based': 0.1, 'premium': 0.0},
                'premium': {'aggressive': 0.0, 'value_based': 0.3, 'premium': 0.7},
                'urgent': {'aggressive': 0.5, 'value_based': 0.3, 'premium': 0.2}
            },
            'business_objective': {
                'market_share': {'aggressive': 0.9, 'value_based': 0.1, 'premium': 0.0},
                'profit_max': {'aggressive': 0.1, 'value_based': 0.4, 'premium': 0.5},
                'revenue_max': {'aggressive': 0.4, 'value_based': 0.5, 'premium': 0.1},
                'inventory_clearance': {'aggressive': 0.8, 'value_based': 0.2, 'premium': 0.0}
            }
        }
    
    def select_optimal_strategy(self, 
                              product_id: str,
                              context: Dict) -> Dict:
        """
        Sélectionner stratégie optimale
        
        Returns:
            Dict avec stratégie recommandée et scores
        """
        
        try:
            # Analyser contexte
            context_analysis = self._analyze_context(product_id, context)
            
            # Calculer scores pour chaque stratégie
            strategy_scores = self._calculate_strategy_scores(context_analysis)
            
            # Sélectionner meilleure stratégie
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1]['total_score'])
            
            # Générer recommandation
            recommendation = self._build_recommendation(
                best_strategy[0], 
                best_strategy[1],
                context_analysis
            )
            
            logger.info(f"🎯 Stratégie sélectionnée: {best_strategy[0]} "
                       f"(score: {best_strategy[1]['total_score']:.2f})")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Erreur sélection stratégie: {e}")
            return self._default_strategy()
    
    def _analyze_context(self, product_id: str, context: Dict) -> Dict:
        """Analyser le contexte complet"""
        
        analysis = {
            'market_conditions': self._analyze_market_conditions(product_id),
            'product_lifecycle': self._analyze_product_lifecycle(product_id),
            'customer_context': self._analyze_customer_context(context),
            'business_context': self._analyze_business_context(context),
            'temporal_factors': self._analyze_temporal_factors()
        }
        
        return analysis
    
    def _analyze_market_conditions(self, product_id: str) -> str:
        """Analyser conditions marché"""
        
        try:
            # Récupérer données concurrents
            competitors = self.db.get_competitor_prices(product_id)
            
            if not competitors:
                return 'low_competition'
            
            # Analyser compétition
            comp_prices = [c['competitor_price'] for c in competitors]
            our_product = self.db.get_product(product_id)
            our_price = float(our_product['current_price']) if our_product else 0
            
            if comp_prices:
                avg_competitor = np.mean(comp_prices)
                price_diff = abs(our_price - avg_competitor) / avg_competitor
                
                if price_diff > 0.2:
                    return 'high_competition'
                elif np.std(comp_prices) / avg_competitor > 0.15:
                    return 'volatile'
                else:
                    return 'low_competition'
            
        except Exception as e:
            logger.warning(f"Erreur analyse marché: {e}")
        
        return 'low_competition'
    
    def _analyze_product_lifecycle(self, product_id: str) -> str:
        """Analyser cycle de vie produit"""
        
        try:
            # Récupérer historique ventes
            query = """
                SELECT 
                    DATEDIFF(NOW(), MIN(timestamp)) as days_since_first_sale,
                    AVG(units_sold) as avg_sales,
                    COUNT(*) as sales_count
                FROM pricing_results pr
                JOIN pricing_decisions pd ON pr.decision_id = pd.decision_id
                WHERE pd.product_id = %s
            """
            
            result = self.db._execute_query(query, (product_id,), fetch_one=True)
            
            if not result:
                return 'introduction'
            
            days_since_first = result['days_since_first_sale'] or 0
            avg_sales = result['avg_sales'] or 0
            
            # Déterminer phase
            if days_since_first < 30:
                return 'introduction'
            elif days_since_first < 180 and avg_sales > 10:
                return 'growth'
            elif days_since_first < 365:
                return 'maturity'
            else:
                return 'decline'
                
        except Exception as e:
            logger.warning(f"Erreur analyse cycle vie: {e}")
        
        return 'maturity'
    
    def _analyze_customer_context(self, context: Dict) -> str:
        """Analyser contexte client"""
        
        segment = context.get('customer_segment')
        
        if segment in ['price_sensitive', 'premium', 'urgent']:
            return segment
        
        # Déduire du contexte
        if context.get('budget_constraint'):
            return 'price_sensitive'
        elif context.get('urgency'):
            return 'urgent'
        elif context.get('loyalty_score', 0) > 0.7:
            return 'premium'
        
        return 'price_sensitive'  # Par défaut
    
    def _analyze_business_context(self, context: Dict) -> str:
        """Analyser contexte business"""
        
        objective = context.get('objective', 'profit_max')
        
        if objective in ['market_share', 'profit_max', 'revenue_max', 'inventory_clearance']:
            return objective
        
        # Vérifier stock
        product_id = context.get('product_id')
        if product_id:
            product = self.db.get_product(product_id)
            if product and product.get('current_stock', 0) > 100:
                return 'inventory_clearance'
        
        return 'profit_max'  # Par défaut
    
    def _analyze_temporal_factors(self) -> Dict:
        """Analyser facteurs temporels"""
        
        now = datetime.now()
        
        factors = {
            'is_weekend': now.weekday() >= 5,
            'is_holiday': self._is_holiday(now),
            'season': self._get_season(now),
            'time_of_day': now.hour,
            'month': now.month
        }
        
        return factors
    
    def _is_holiday(self, date: datetime) -> bool:
        """Vérifier si jour férié (simplifié)"""
        # Liste simplifiée
        holidays = [
            (1, 1),   # Nouvel an
            (5, 1),   # Fête travail
            (7, 14),  # Fête nationale
            (12, 25)  # Noël
        ]
        
        return (date.month, date.day) in holidays
    
    def _get_season(self, date: datetime) -> str:
        """Obtenir saison"""
        month = date.month
        
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def _calculate_strategy_scores(self, context_analysis: Dict) -> Dict:
        """Calculer scores pour chaque stratégie"""
        
        strategies = ['aggressive', 'value_based', 'premium']
        scores = {s: {'total_score': 0, 'component_scores': {}} for s in strategies}
        
        # Calculer pour chaque composante
        for component, component_value in context_analysis.items():
            if component in self.strategy_weights:
                if isinstance(component_value, dict):
                    # Facteurs temporels spéciaux
                    if component == 'temporal_factors':
                        temporal_scores = self._calculate_temporal_scores(component_value)
                        for strategy in strategies:
                            scores[strategy]['component_scores']['temporal'] = temporal_scores.get(strategy, 0.5)
                            scores[strategy]['total_score'] += temporal_scores.get(strategy, 0.5) * 0.2
                else:
                    # Composantes normales
                    weights = self.strategy_weights[component].get(component_value, {})
                    
                    for strategy in strategies:
                        weight = weights.get(strategy, 0.3)  # Default
                        scores[strategy]['component_scores'][component] = weight
                        
                        # Ponderer l'importance de la composante
                        component_weight = self._get_component_weight(component)
                        scores[strategy]['total_score'] += weight * component_weight
        
        return scores
    
    def _calculate_temporal_scores(self, temporal_factors: Dict) -> Dict:
        """Calculer scores basés sur facteurs temporels"""
        
        scores = {'aggressive': 0.5, 'value_based': 0.5, 'premium': 0.5}
        
        # Weekend vs semaine
        if temporal_factors['is_weekend']:
            scores['premium'] += 0.2  # Plus de premium weekend
            scores['aggressive'] -= 0.1
        else:
            scores['aggressive'] += 0.1  # Plus aggressif en semaine
            scores['value_based'] += 0.1
        
        # Saison
        season = temporal_factors['season']
        if season == 'summer':
            scores['aggressive'] += 0.15 