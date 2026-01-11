"""
Évaluateur pour agents de vente
"""

import numpy as np
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SalesEvaluator:
    """Évalue les performances des agents de vente"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def evaluate_agent_performance(self, 
                                 agent_type: str, 
                                 product_id: str,
                                 period_days: int = 30) -> Dict:
        """Évaluer performance agent sur période"""
        
        logger.info(f"📊 Évaluation agent {agent_type} sur {product_id}")
        
        # Récupérer données
        decisions = self._get_agent_decisions(agent_type, product_id, period_days)
        results = self._get_pricing_results(product_id, period_days)
        
        if not decisions or not results:
            return {'error': 'No data available'}
        
        # Calculer métriques
        metrics = {
            'total_decisions': len(decisions),
            'total_revenue': 0,
            'total_profit': 0,
            'avg_profit_per_decision': 0,
            'conversion_rate': 0,
            'price_stability': 0,
            'confidence_avg': 0,
            'best_decision': None,
            'worst_decision': None
        }
        
        # Analyser décisions
        profits = []
        confidences = []
        price_changes = []
        
        for decision in decisions:
            # Trouver résultat correspondant
            result = next((r for r in results 
                          if r['decision_id'] == decision['decision_id']), None)
            
            if result:
                profit = result.get('profit', 0)
                metrics['total_revenue'] += result.get('revenue', 0)
                metrics['total_profit'] += profit
                profits.append(profit)
            
            confidences.append(decision.get('confidence_score', 0))
            price_changes.append(abs(decision.get('price_change', 0)))
        
        # Calculer métriques agrégées
        if profits:
            metrics['avg_profit_per_decision'] = np.mean(profits)
            metrics['best_decision'] = max(profits) if profits else 0
            metrics['worst_decision'] = min(profits) if profits else 0
        
        if confidences:
            metrics['confidence_avg'] = np.mean(confidences)
        
        if price_changes:
            metrics['price_stability'] = 1.0 / (1.0 + np.mean(price_changes))
        
        # Calculer taux conversion
        successful_decisions = len([p for p in profits if p > 0])
        metrics['conversion_rate'] = successful_decisions / len(decisions) if decisions else 0
        
        # Score global
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        logger.info(f"✅ Évaluation terminée - Score: {metrics['overall_score']:.2f}")
        
        return metrics
    
    def _get_agent_decisions(self, agent_type: str, product_id: str, days: int) -> List[Dict]:
        """Récupérer décisions agent"""
        
        query = """
            SELECT * FROM pricing_decisions 
            WHERE product_id = %s 
            AND model_id LIKE %s
            AND timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY timestamp DESC
        """
        
        return self.db._execute_query(query, (
            product_id,
            f"%{agent_type}%",
            days
        ))
    
    def _get_pricing_results(self, product_id: str, days: int) -> List[Dict]:
        """Récupérer résultats pricing"""
        
        query = """
            SELECT pr.* FROM pricing_results pr
            JOIN pricing_decisions pd ON pr.decision_id = pd.decision_id
            WHERE pd.product_id = %s
            AND pr.created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        """
        
        return self.db._execute_query(query, (product_id, days))
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculer score global"""
        
        weights = {
            'avg_profit_per_decision': 0.3,
            'conversion_rate': 0.25,
            'price_stability': 0.2,
            'confidence_avg': 0.15,
            'total_decisions': 0.1
        }
        
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                # Normaliser la métrique
                normalized = self._normalize_metric(metric, metrics[metric])
                score += normalized * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0
    
    def _normalize_metric(self, metric: str, value: Any) -> float:
        """Normaliser une métrique entre 0 et 1"""
        
        normalization_rules = {
            'avg_profit_per_decision': lambda x: min(x / 1000, 1.0),  # Max 1000€
            'conversion_rate': lambda x: x,  # Déjà entre 0 et 1
            'price_stability': lambda x: x,  # Déjà entre 0 et 1
            'confidence_avg': lambda x: x,  # Déjà entre 0 et 1
            'total_decisions': lambda x: min(x / 100, 1.0)  # Max 100 décisions
        }
        
        if metric in normalization_rules:
            return max(0, min(1, normalization_rules[metric](value)))
        
        return 0.0
    
    def compare_agents(self, 
                      product_id: str, 
                      agent_types: List[str],
                      period_days: int = 30) -> Dict:
        """Comparer plusieurs agents"""
        
        comparison = {}
        
        for agent_type in agent_types:
            metrics = self.evaluate_agent_performance(
                agent_type, product_id, period_days
            )
            comparison[agent_type] = metrics
        
        # Déterminer meilleur agent
        if comparison:
            best_agent = max(comparison.items(), 
                           key=lambda x: x[1].get('overall_score', 0))
            
            comparison['best_agent'] = {
                'agent_type': best_agent[0],
                'score': best_agent[1].get('overall_score', 0)
            }
        
        return comparison
    
    def generate_performance_report(self, 
                                  product_id: str, 
                                  start_date: datetime, 
                                  end_date: datetime) -> Dict:
        """Générer rapport de performance détaillé"""
        
        report = {
            'period': {
                'start': start_date,
                'end': end_date,
                'days': (end_date - start_date).days
            },
            'product_id': product_id,
            'agents_performance': {},
            'market_analysis': {},
            'recommendations': []
        }
        
        # Analyser chaque agent
        agent_types = ['aggressive', 'value_based', 'premium', 'bundle']
        
        for agent in agent_types:
            performance = self.evaluate_agent_performance(
                agent, product_id, report['period']['days']
            )
            report['agents_performance'][agent] = performance
        
        # Analyse marché
        report['market_analysis'] = self._analyze_market_trends(
            product_id, start_date, end_date
        )
        
        # Recommandations
        report['recommendations'] = self._generate_recommendations(
            report['agents_performance'],
            report['market_analysis']
        )
        
        return report
    
    def _analyze_market_trends(self, product_id: str, start: datetime, end: datetime) -> Dict:
        """Analyser tendances marché"""
        
        query = """
            SELECT 
                DATE(timestamp) as date,
                AVG(current_price) as avg_price,
                AVG(competitor_1_price) as avg_competitor_1,
                AVG(competitor_2_price) as avg_competitor_2,
                COUNT(*) as observations
            FROM pricing_states 
            WHERE product_id = %s
            AND timestamp BETWEEN %s AND %s
            GROUP BY DATE(timestamp)
            ORDER BY date
        """
        
        data = self.db._execute_query(query, (product_id, start, end))
        
        if not data:
            return {}
        
        # Calculer tendances
        prices = [float(d['avg_price']) for d in data]
        comp1_prices = [float(d['avg_competitor_1']) for d in data if d['avg_competitor_1']]
        comp2_prices = [float(d['avg_competitor_2']) for d in data if d['avg_competitor_2']]
        
        trends = {
            'price_trend': self._calculate_trend(prices),
            'competition_trend': self._calculate_trend(comp1_prices + comp2_prices) 
                               if comp1_prices or comp2_prices else 0,
            'price_volatility': np.std(prices) / np.mean(prices) if prices else 0,
            'competition_intensity': self._calculate_competition_intensity(
                prices, comp1_prices, comp2_prices
            ),
            'days_analyzed': len(data)
        }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculer tendance linéaire"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        return slope / np.mean(values) if np.mean(values) != 0 else 0
    
    def _calculate_competition_intensity(self, 
                                       our_prices: List[float],
                                       comp1_prices: List[float],
                                       comp2_prices: List[float]) -> float:
        """Calculer intensité compétition"""
        
        if not our_prices or (not comp1_prices and not comp2_prices):
            return 0
        
        # Calculer différence moyenne
        differences = []
        
        if comp1_prices:
            min_len = min(len(our_prices), len(comp1_prices))
            diff = np.mean([abs(our_prices[i] - comp1_prices[i]) 
                          for i in range(min_len)])
            differences.append(diff)
        
        if comp2_prices:
            min_len = min(len(our_prices), len(comp2_prices))
            diff = np.mean([abs(our_prices[i] - comp2_prices[i]) 
                          for i in range(min_len)])
            differences.append(diff)
        
        avg_difference = np.mean(differences) if differences else 0
        avg_price = np.mean(our_prices)
        
        return avg_difference / avg_price if avg_price != 0 else 0
    
    def _generate_recommendations(self, 
                                agent_performance: Dict,
                                market_analysis: Dict) -> List[str]:
        """Générer recommandations basées sur performance"""
        
        recommendations = []
        
        # Trouver meilleur agent
        best_agent = max(agent_performance.items(), 
                        key=lambda x: x[1].get('overall_score', 0),
                        default=(None, {}))
        
        if best_agent[0]:
            recommendations.append(
                f"Utiliser l'agent {best_agent[0]} (score: {best_agent[1].get('overall_score', 0):.2f})"
            )
        
        # Analyse marché
        competition = market_analysis.get('competition_intensity', 0)
        if competition > 0.1:
            recommendations.append(
                "Compétition intense - Stratégie aggressive recommandée"
            )
        elif competition < 0.05:
            recommendations.append(
                "Faible compétition - Stratégie premium possible"
            )
        
        # Analyse stabilité prix
        volatility = market_analysis.get('price_volatility', 0)
        if volatility > 0.15:
            recommendations.append(
                "Haute volatilité - Augmenter fréquence d'ajustement prix"
            )
        
        return recommendations