"""
Benchmark des agents RL vs stratégies traditionnelles
"""

import numpy as np
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class Benchmark:
    """Benchmarking des stratégies de pricing"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.baseline_strategies = self._define_baseline_strategies()
    
    def _define_baseline_strategies(self) -> Dict:
        """Définir stratégies baselines"""
        
        return {
            'fixed_price': {
                'name': 'Prix Fixe',
                'description': 'Prix constant sur la période',
                'function': self._fixed_price_strategy
            },
            'cost_plus': {
                'name': 'Coût + Marge',
                'description': 'Coût + pourcentage fixe (30%)',
                'function': self._cost_plus_strategy
            },
            'competitor_match': {
                'name': 'Alignement Concurrent',
                'description': 'Moyenne des prix concurrents',
                'function': self._competitor_match_strategy
            },
            'weekend_boost': {
                'name': 'Boost Weekend',
                'description': '+10% weekend, -5% semaine',
                'function': self._weekend_boost_strategy
            },
            'clearance': {
                'name': 'Liquidation Stock',
                'description': 'Réduction progressive selon stock',
                'function': self._clearance_strategy
            }
        }
    
    def run_benchmark(self, 
                     product_id: str, 
                     period_days: int = 90,
                     n_simulations: int = 100) -> Dict:
        """Exécuter benchmark complet"""
        
        logger.info(f"🏁 Benchmark produit {product_id}")
        
        # Récupérer données historiques
        historical_data = self._get_historical_data(product_id, period_days)
        
        if not historical_data:
            return {'error': 'No historical data'}
        
        benchmark_results = {}
        
        # Tester chaque stratégie
        for strategy_id, strategy_info in self.baseline_strategies.items():
            logger.info(f"  Testing: {strategy_info['name']}")
            
            results = self._test_strategy(
                strategy_info['function'],
                historical_data,
                n_simulations
            )
            
            benchmark_results[strategy_id] = {
                'name': strategy_info['name'],
                'description': strategy_info['description'],
                'results': results
            }
        
        # Tester agent RL (si disponible)
        rl_results = self._test_rl_agent(product_id, historical_data, n_simulations)
        if rl_results:
            benchmark_results['rl_agent'] = {
                'name': 'Agent RL',
                'description': 'Reinforcement Learning fine-tuné',
                'results': rl_results
            }
        
        # Analyser résultats
        analysis = self._analyze_benchmark_results(benchmark_results)
        
        logger.info("✅ Benchmark terminé")
        
        return {
            'benchmark_results': benchmark_results,
            'analysis': analysis,
            'best_strategy': analysis.get('best_strategy', {}),
            'rl_vs_baseline': analysis.get('rl_improvement', 0)
        }
    
    def _get_historical_data(self, product_id: str, days: int) -> List[Dict]:
        """Récupérer données historiques"""
        
        query = """
            SELECT 
                ps.timestamp,
                ps.current_price,
                ps.competitor_1_price,
                ps.competitor_2_price,
                ps.current_stock,
                pr.units_sold,
                pr.revenue,
                pr.profit
            FROM pricing_states ps
            LEFT JOIN pricing_results pr ON ps.state_id = (
                SELECT state_id FROM pricing_decisions 
                WHERE state_id = ps.state_id LIMIT 1
            )
            WHERE ps.product_id = %s
            AND ps.timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY ps.timestamp
        """
        
        return self.db._execute_query(query, (product_id, days))
    
    def _test_strategy(self, strategy_func, historical_data: List[Dict], n_simulations: int) -> Dict:
        """Tester une stratégie sur données historiques"""
        
        all_profits = []
        all_revenues = []
        all_decisions = []
        
        for sim in range(n_simulations):
            sim_profits = []
            sim_revenues = []
            sim_decisions = []
            
            for i, data_point in enumerate(historical_data):
                if i == 0:
                    continue
                
                # Données pour décision
                context = {
                    'current_price': historical_data[i-1]['current_price'],
                    'competitor_prices': [
                        historical_data[i-1].get('competitor_1_price'),
                        historical_data[i-1].get('competitor_2_price')
                    ],
                    'current_stock': historical_data[i-1]['current_stock'],
                    'historical_demand': historical_data[i-1].get('units_sold', 0),
                    'day_of_week': historical_data[i-1]['timestamp'].weekday(),
                    'is_weekend': historical_data[i-1]['timestamp'].weekday() >= 5
                }
                
                # Appliquer stratégie
                decision = strategy_func(context)
                sim_decisions.append(decision)
                
                # Simuler résultat
                simulated_result = self._simulate_decision_result(
                    decision, data_point, context
                )
                
                sim_profits.append(simulated_result.get('profit', 0))
                sim_revenues.append(simulated_result.get('revenue', 0))
            
            if sim_profits:
                all_profits.append(np.sum(sim_profits))
                all_revenues.append(np.sum(sim_revenues))
                all_decisions.append(sim_decisions)
        
        # Statistiques
        if all_profits:
            results = {
                'avg_total_profit': np.mean(all_profits),
                'std_total_profit': np.std(all_profits),
                'min_total_profit': np.min(all_profits),
                'max_total_profit': np.max(all_profits),
                'avg_total_revenue': np.mean(all_revenues),
                'avg_profit_per_decision': np.mean([p/len(historical_data) for p in all_profits]),
                'n_simulations': n_simulations,
                'n_decisions_per_sim': len(historical_data) - 1,
                'decision_pattern': self._analyze_decision_pattern(all_decisions[0] if all_decisions else [])
            }
        else:
            results = {'error': 'No simulation results'}
        
        return results
    
    def _fixed_price_strategy(self, context: Dict) -> Dict:
        """Stratégie prix fixe"""
        return {
            'strategy': 'fixed_price',
            'price': context['current_price'],  # Même prix
            'confidence': 1.0,
            'explanation': 'Prix constant'
        }
    
    def _cost_plus_strategy(self, context: Dict) -> Dict:
        """Stratégie coût + marge"""
        # Récupérer coût depuis DB
        product = self.db.get_product('PROD_001')  # Placeholder
        cost = product['cost_price'] if product else 50
        
        price = cost * 1.3  # Marge 30%
        
        return {
            'strategy': 'cost_plus',
            'price': price,
            'confidence': 0.8,
            'explanation': f'Coût ({cost}€) + 30% marge'
        }
    
    def _competitor_match_strategy(self, context: Dict) -> Dict:
        """Stratégie alignement concurrent"""
        comp_prices = [p for p in context['competitor_prices'] if p is not None]
        
        if comp_prices:
            target_price = np.mean(comp_prices)
        else:
            target_price = context['current_price']
        
        return {
            'strategy': 'competitor_match',
            'price': target_price,
            'confidence': 0.7,
            'explanation': f'Moyenne concurrents: {target_price:.2f}€'
        }
    
    def _weekend_boost_strategy(self, context: Dict) -> Dict:
        """Stratégie boost weekend"""
        if context['is_weekend']:
            price = context['current_price'] * 1.10
            explanation = 'Weekend: +10%'
        else:
            price = context['current_price'] * 0.95
            explanation = 'Semaine: -5%'
        
        return {
            'strategy': 'weekend_boost',
            'price': price,
            'confidence': 0.6,
            'explanation': explanation
        }
    
    def _clearance_strategy(self, context: Dict) -> Dict:
        """Stratégie liquidation stock"""
        stock_ratio = context['current_stock'] / 100.0  # Normalisé
        
        if stock_ratio > 0.8:
            discount = 0.20  # -20%
        elif stock_ratio > 0.5:
            discount = 0.10  # -10%
        else:
            discount = 0.0
        
        price = context['current_price'] * (1 - discount)
        
        return {
            'strategy': 'clearance',
            'price': price,
            'confidence': 0.75,
            'explanation': f'Liquidation stock ({context["current_stock"]} unités): -{discount*100}%'
        }
    
    def _test_rl_agent(self, product_id: str, historical_data: List[Dict], n_simulations: int) -> Dict:
        """Tester agent RL"""
        # Placeholder - implémentation réelle chargerait le modèle
        return None
    
    def _simulate_decision_result(self, decision: Dict, actual_data: Dict, context: Dict) -> Dict:
        """Simuler résultat d'une décision"""
        
        # Prix décidé
        decided_price = decision['price']
        
        # Prix réel ce jour-là (pour comparaison)
        actual_price = actual_data.get('current_price', decided_price)
        
        # Demande simulée basée sur prix
        price_ratio = decided_price / max(actual_price, 1)
        elasticity = -1.5
        demand_effect = price_ratio ** elasticity
        
        base_demand = actual_data.get('units_sold', 10)
        simulated_demand = max(0, base_demand * demand_effect * np.random.uniform(0.8, 1.2))
        
        # Coût unitaire (placeholder)
        unit_cost = 50
        
        # Calculer résultats
        units_sold = min(simulated_demand, context['current_stock'])
        revenue = units_sold * decided_price
        cost = units_sold * unit_cost
        profit = revenue - cost
        
        return {
            'price': decided_price,
            'units_sold': units_sold,
            'revenue': revenue,
            'profit': profit,
            'demand_effect': demand_effect
        }
    
    def _analyze_decision_pattern(self, decisions: List[Dict]) -> Dict:
        """Analyser pattern des décisions"""
        
        if not decisions:
            return {}
        
        prices = [d['price'] for d in decisions]
        strategies = [d['strategy'] for d in decisions]
        
        pattern = {
            'price_mean': np.mean(prices),
            'price_std': np.std(prices),
            'price_range': [min(prices), max(prices)],
            'strategy_distribution': {s: strategies.count(s) / len(strategies) 
                                     for s in set(strategies)},
            'price_trend': self._calculate_price_trend(prices),
            'volatility': np.std(prices) / np.mean(prices) if np.mean(prices) != 0 else 0
        }
        
        return pattern
    
    def _calculate_price_trend(self, prices: List[float]) -> float:
        """Calculer tendance prix"""
        if len(prices) < 2:
            return 0
        
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        return slope / np.mean(prices) if np.mean(prices) != 0 else 0
    
    def _analyze_benchmark_results(self, benchmark_results: Dict) -> Dict:
        """Analyser résultats benchmark"""
        
        analysis = {
            'strategies_comparison': [],
            'best_strategy': {},
            'rl_improvement': 0,
            'key_insights': []
        }
        
        # Comparer stratégies
        for strategy_id, data in benchmark_results.items():
            results = data['results']
            
            if 'avg_total_profit' in results:
                analysis['strategies_comparison'].append({
                    'strategy_id': strategy_id,
                    'strategy_name': data['name'],
                    'avg_profit': results['avg_total_profit'],
                    'profit_std': results['std_total_profit'],
                    'profit_range': [results['min_total_profit'], results['max_total_profit']],
                    'profit_per_decision': results['avg_profit_per_decision']
                })
        
        # Déterminer meilleure stratégie
        if analysis['strategies_comparison']:
            best = max(analysis['strategies_comparison'], 
                      key=lambda x: x['avg_profit'])
            analysis['best_strategy'] = best
            
            # Calculer amélioration RL si présent
            if 'rl_agent' in benchmark_results:
                rl_profit = benchmark_results['rl_agent']['results'].get('avg_total_profit', 0)
                if best['avg_profit'] > 0:
                    analysis['rl_improvement'] = (rl_profit - best['avg_profit']) / best['avg_profit'] * 100
        
        # Générer insights
        analysis['key_insights'] = self._generate_benchmark_insights(
            benchmark_results, analysis
        )
        
        return analysis
    
    def _generate_benchmark_insights(self, results: Dict, analysis: Dict) -> List[str]:
        """Générer insights du benchmark"""
        
        insights = []
        
        # Insight profit
        best_profit = analysis.get('best_strategy', {}).get('avg_profit', 0)
        insights.append(f"Profit moyen meilleure stratégie: {best_profit:.2f}€")
        
        # Insight volatilité
        strategy_volatilities = {}
        for sid, data in results.items():
            if 'results' in data and 'std_total_profit' in data['results']:
                strategy_volatilities[sid] = data['results']['std_total_profit']
        
        if strategy_volatilities:
            most_stable = min(strategy_volatilities.items(), key=lambda x: x[1])
            insights.append(f"Stratégie la plus stable: {results[most_stable[0]]['name']}")
        
        # Insight RL
        if 'rl_agent' in results:
            rl_improvement = analysis.get('rl_improvement', 0)
            if rl_improvement > 0:
                insights.append(f"Agent RL améliore de {rl_improvement:.1f}% vs baselines")
            else:
                insights.append("Agent RL ne surpasse pas les baselines (besoin de plus de training)")
        
        return insights
    
    def generate_benchmark_report(self, 
                                product_id: str, 
                                benchmark_results: Dict) -> str:
        """Générer rapport benchmark formaté"""
        
        report_lines = [
            "=" * 60,
            f"RAPPORT BENCHMARK - Produit {product_id}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 60,
            ""
        ]
        
        # Résumé
        report_lines.append("📊 RÉSUMÉ DES PERFORMANCES")
        report_lines.append("-" * 40)
        
        for strategy_id, data in benchmark_results.items():
            if 'results' in data and 'avg_total_profit' in data['results']:
                profit = data['results']['avg_total_profit']
                std = data['results']['std_total_profit']
                report_lines.append(
                    f"{data['name']:20} | Profit: {profit:8.2f}€ (±{std:.2f})"
                )
        
        # Meilleure stratégie
        if 'analysis' in benchmark_results:
            best = benchmark_results['analysis'].get('best_strategy', {})
            if best:
                report_lines.append("")
                report_lines.append("🏆 MEILLEURE STRATÉGIE")
                report_lines.append("-" * 40)
                report_lines.append(f"Stratégie: {best.get('strategy_name', 'N/A')}")
                report_lines.append(f"Profit moyen: {best.get('avg_profit', 0):.2f}€")
                report_lines.append(f"Stabilité: ±{best.get('profit_std', 0):.2f}€")
        
        # Insights
        if 'analysis' in benchmark_results:
            insights = benchmark_results['analysis'].get('key_insights', [])
            if insights:
                report_lines.append("")
                report_lines.append("💡 INSIGHTS CLÉS")
                report_lines.append("-" * 40)
                for insight in insights:
                    report_lines.append(f"• {insight}")
        
        return "\n".join(report_lines)