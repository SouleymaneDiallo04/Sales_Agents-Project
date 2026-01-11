"""
Orchestrateur pour sélectionner et gérer les agents
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from data.database.connection import DatabaseManager
from agents.customer_segments.price_sensitive_agent import PriceSensitiveAgent
from agents.customer_segments.premium_agent import PremiumAgent
from agents.customer_segments.urgent_agent import UrgentAgent
from agents.product_categories.electronics_agent import ElectronicsAgent
from agents.product_categories.fashion_agent import FashionAgent
from agents.product_categories.home_agent import HomeAgent
from agents.sales_strategies.aggressive_pricing import AggressivePricingAgent
from agents.sales_strategies.value_based import ValueBasedAgent
from agents.sales_strategies.bundle_strategy import BundleStrategyAgent

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrateur pour sélectionner le meilleur agent"""
    
    def __init__(self, db_connection: DatabaseManager):
        self.db = db_connection
        self.agents = self._initialize_agents()
        self.agent_performance = {}
        self.decision_history = []
        
    def _initialize_agents(self) -> Dict:
        """Initialiser tous les agents disponibles"""
        
        agents = {
            # Agents par segment client
            'price_sensitive': PriceSensitiveAgent(self.db),
            'premium': PremiumAgent(self.db),
            'urgent': UrgentAgent(self.db),
            
            # Agents par catégorie produit
            'electronics': ElectronicsAgent(self.db),
            'fashion': FashionAgent(self.db),
            'home': HomeAgent(self.db),
            
            # Agents par stratégie
            'aggressive': AggressivePricingAgent(self.db),
            'value_based': ValueBasedAgent(self.db),
            'bundle': BundleStrategyAgent(self.db),
        }
        
        logger.info(f"✅ {len(agents)} agents initialisés")
        return agents
    
    def get_pricing_decision(self, 
                           product_id: str, 
                           context: Dict = None) -> Optional[Dict]:
        """
        Obtenir la meilleure décision de prix pour un produit
        
        Args:
            product_id: ID du produit
            context: Contexte supplémentaire (segment client, etc.)
        
        Returns:
            Décision de pricing ou None
        """
        context = context or {}
        
        try:
            # Vérifier produit
            product = self.db.get_product(product_id)
            if not product or not product.get('is_active', 1):
                logger.warning(f"Produit {product_id} non trouvé ou inactif")
                return None
            
            # Sélectionner agent(s) approprié(s)
            selected_agents = self._select_agents(product, context)
            
            if not selected_agents:
                logger.warning(f"Aucun agent sélectionné pour {product_id}")
                return self._fallback_decision(product)
            
            # Obtenir décisions de chaque agent
            decisions = []
            for agent_name, agent in selected_agents:
                try:
                    decision = agent.decide_price(product_id, context)
                    if decision:
                        decision['agent'] = agent_name
                        decision['product_id'] = product_id
                        decisions.append(decision)
                except Exception as e:
                    logger.error(f"Erreur agent {agent_name}: {e}")
            
            if not decisions:
                return self._fallback_decision(product)
            
            # Sélectionner meilleure décision
            best_decision = self._select_best_decision(decisions, product, context)
            
            # Enrichir avec infos produit
            best_decision.update({
                'product_name': product.get('product_name', 'Unknown'),
                'cost_price': float(product.get('cost_price', 0)),
                'min_price': float(product.get('min_price', 0)),
                'max_price': float(product.get('max_price', 0)),
                'current_stock': product.get('current_stock', 0),
                'decision_timestamp': datetime.now()
            })
            
            # Sauvegarder décision
            decision_id = self._save_decision_to_db(best_decision)
            best_decision['decision_id'] = decision_id
            
            # Mettre à jour performance agent
            self._update_agent_performance(best_decision['agent'], best_decision)
            
            logger.info(f"🎯 Décision pour {product_id}: {best_decision['agent']} "
                       f"-> {best_decision['recommended_price']}€ "
                       f"(confiance: {best_decision['confidence']:.2%})")
            
            return best_decision
            
        except Exception as e:
            logger.error(f"Erreur orchestration: {e}")
            return None
    
    def _select_agents(self, product: Dict, context: Dict) -> List[tuple]:
        """Sélectionner agents appropriés"""
        
        selected = []
        
        # 1. Par segment client (priorité)
        customer_segment = context.get('customer_segment')
        if customer_segment:
            if customer_segment == 'price_sensitive' and 'price_sensitive' in self.agents:
                selected.append(('price_sensitive', self.agents['price_sensitive']))
            elif customer_segment == 'premium' and 'premium' in self.agents:
                selected.append(('premium', self.agents['premium']))
            elif customer_segment == 'urgent' and 'urgent' in self.agents:
                selected.append(('urgent', self.agents['urgent']))
        
        # 2. Par catégorie produit
        category = product.get('category', '').lower()
        if 'electron' in category and 'electronics' in self.agents:
            selected.append(('electronics', self.agents['electronics']))
        elif 'fashion' in category or 'cloth' in category and 'fashion' in self.agents:
            selected.append(('fashion', self.agents['fashion']))
        elif 'home' in category or 'decor' in category and 'home' in self.agents:
            selected.append(('home', self.agents['home']))
        
        # 3. Stratégies générales (toujours disponibles)
        strategic_agents = ['aggressive', 'value_based']
        for agent_name in strategic_agents:
            if agent_name in self.agents:
                selected.append((agent_name, self.agents[agent_name]))
        
        # 4. Si bundle possible
        if context.get('bundle_opportunity') and 'bundle' in self.agents:
            selected.append(('bundle', self.agents['bundle']))
        
        # Éliminer doublons
        unique_selected = []
        seen = set()
        for agent_name, agent in selected:
            if agent_name not in seen:
                unique_selected.append((agent_name, agent))
                seen.add(agent_name)
        
        return unique_selected
    
    def _select_best_decision(self, 
                            decisions: List[Dict], 
                            product: Dict,
                            context: Dict) -> Dict:
        """Sélectionner la meilleure décision"""
        
        if len(decisions) == 1:
            return decisions[0]
        
        # Score chaque décision
        scored_decisions = []
        for decision in decisions:
            score = self._score_decision(decision, product, context)
            decision['selection_score'] = score
            scored_decisions.append((score, decision))
        
        # Trier par score
        scored_decisions.sort(key=lambda x: x[0], reverse=True)
        
        # Prendre la meilleure
        best_score, best_decision = scored_decisions[0]
        
        # Si confiance trop basse, considérer deuxième choix
        if best_decision['confidence'] < 0.6 and len(scored_decisions) > 1:
            second_score, second_decision = scored_decisions[1]
            if second_decision['confidence'] > 0.7 and second_score > best_score * 0.8:
                logger.info(f"Changement décision: faible confiance ({best_decision['confidence']:.2%})")
                return second_decision
        
        return best_decision
    
    def _score_decision(self, decision: Dict, product: Dict, context: Dict) -> float:
        """Noter une décision"""
        
        score = 0.0
        
        # 1. Confiance de l'agent
        confidence = decision.get('confidence', 0.5)
        score += confidence * 0.4
        
        # 2. Performance historique de l'agent
        agent_perf = self.agent_performance.get(decision['agent'], {})
        perf_score = agent_perf.get('success_rate', 0.5)
        score += perf_score * 0.3
        
        # 3. Adéquation avec contexte
        context_score = self._calculate_context_score(decision, context)
        score += context_score * 0.2
        
        # 4. Respect contraintes
        constraints_score = self._check_constraints(decision, product)
        score += constraints_score * 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_context_score(self, decision: Dict, context: Dict) -> float:
        """Calculer score d'adéquation contexte"""
        
        # Vérifier objectif
        objective = context.get('objective', 'profit')
        strategy = decision.get('strategy', '')
        
        if objective == 'profit' and 'premium' in strategy.lower():
            return 0.9
        elif objective == 'volume' and 'aggressive' in strategy.lower():
            return 0.9
        elif objective == 'revenue' and 'value' in strategy.lower():
            return 0.8
        
        # Vérifier horizon
        horizon = context.get('horizon', 'short')
        price_change = abs(decision.get('price_change_percent', 0))
        
        if horizon == 'short' and price_change > 10:
            return 0.6  # Changements importants OK court terme
        elif horizon == 'long' and price_change < 5:
            return 0.7  # Stabilité préférable long terme
        
        return 0.5
    
    def _check_constraints(self, decision: Dict, product: Dict) -> float:
        """Vérifier respect des contraintes"""
        
        recommended = decision['recommended_price']
        min_price = float(product.get('min_price', 0))
        max_price = float(product.get('max_price', float('inf')))
        cost = float(product.get('cost_price', 0))
        
        violations = 0
        
        # Prix dans bornes
        if recommended < min_price:
            violations += 1
        if recommended > max_price:
            violations += 1
        
        # Marge minimale (10%)
        if cost > 0 and recommended < cost * 1.1:
            violations += 1
        
        # Changement max (20%)
        current = float(product.get('current_price', recommended))
        if abs(recommended - current) / current > 0.2:
            violations += 1
        
        return max(0, 1 - violations * 0.25)
    
    def _fallback_decision(self, product: Dict) -> Dict:
        """Décision de secours"""
        
        current_price = float(product.get('current_price', 0))
        cost_price = float(product.get('cost_price', 0))
        
        # Stratégie simple: marge de 30%
        if cost_price > 0:
            recommended = cost_price * 1.3
        else:
            recommended = current_price * 0.95  # Légère baisse
        
        return {
            'agent': 'fallback',
            'current_price': current_price,
            'recommended_price': recommended,
            'price_change_percent': ((recommended - current_price) / current_price) * 100,
            'confidence': 0.5,
            'strategy': 'cost_plus_fallback',
            'explanation': 'Décision de secours (coût + 30%)'
        }
    
    def _save_decision_to_db(self, decision: Dict) -> int:
        """Sauvegarder décision dans DB"""
        
        decision_data = {
            'product_id': decision['product_id'],
            'state_id': None,  # À compléter si on a un state
            'action_taken': decision.get('strategy', 'unknown'),
            'recommended_price': decision['recommended_price'],
            'confidence_score': decision['confidence'],
            'model_id': decision['agent'],
            'timestamp': decision.get('decision_timestamp', datetime.now()),
            'is_executed': 0  # Pas encore exécuté
        }
        
        return self.db.save_pricing_decision(decision_data)
    
    def _update_agent_performance(self, agent_name: str, decision: Dict):
        """Mettre à jour performance agent"""
        
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                'total_decisions': 0,
                'successful_decisions': 0,
                'total_confidence': 0,
                'avg_confidence': 0,
                'success_rate': 0.5,
                'last_used': datetime.now()
            }
        
        perf = self.agent_performance[agent_name]
        perf['total_decisions'] += 1
        perf['total_confidence'] += decision['confidence']
        perf['avg_confidence'] = perf['total_confidence'] / perf['total_decisions']
        perf['last_used'] = datetime.now()
        
        # Note: success réel sera mis à jour quand on aura les résultats
        # Pour l'instant on estime basé sur confidence
        if decision['confidence'] > 0.7:
            perf['successful_decisions'] += 1
        
        if perf['total_decisions'] > 0:
            perf['success_rate'] = perf['successful_decisions'] / perf['total_decisions']
    
    def process_batch_decisions(self, decisions: List[Dict]):
        """Traiter un batch de décisions"""
        
        logger.info(f"Traitement batch: {len(decisions)} décisions")
        
        for decision in decisions:
            try:
                # Mettre à jour prix produit
                self.db.update_product_price(
                    decision['product_id'],
                    decision['recommended_price']
                )
                
                # Marquer comme exécuté
                if 'decision_id' in decision:
                    query = """
                        UPDATE pricing_decisions 
                        SET is_executed = 1 
                        WHERE decision_id = %s
                    """
                    self.db._execute_query(query, (decision['decision_id'],), fetch=False)
                
                logger.debug(f"Prix mis à jour: {decision['product_id']} -> {decision['recommended_price']}€")
                
            except Exception as e:
                logger.error(f"Erreur traitement décision {decision.get('product_id')}: {e}")
    
    def get_agent_statistics(self) -> Dict:
        """Obtenir statistiques des agents"""
        
        stats = {
            'total_agents': len(self.agents),
            'agent_performance': self.agent_performance,
            'recent_decisions': self.decision_history[-10:] if self.decision_history else [],
            'most_used_agent': None,
            'best_performing_agent': None
        }
        
        if self.agent_performance:
            # Agent le plus utilisé
            most_used = max(self.agent_performance.items(), 
                          key=lambda x: x[1].get('total_decisions', 0))
            stats['most_used_agent'] = {
                'agent': most_used[0],
                'decisions': most_used[1]['total_decisions']
            }
            
            # Agent avec meilleure performance
            best_perf = max(self.agent_performance.items(),
                          key=lambda x: x[1].get('success_rate', 0))
            stats['best_performing_agent'] = {
                'agent': best_perf[0],
                'success_rate': best_perf[1]['success_rate']
            }
        
        return stats
    
    def retrain_underperforming_agents(self, threshold: float = 0.6):
        """Re-entraîner agents sous-performants"""
        
        underperforming = []
        
        for agent_name, perf in self.agent_performance.items():
            if perf.get('success_rate', 1.0) < threshold and perf.get('total_decisions', 0) >= 10:
                underperforming.append(agent_name)
        
        if underperforming:
            logger.info(f"Agents sous-performants détectés: {underperforming}")
            # Ici on lancerait le re-training
            # Pour l'instant on log juste
            
        return underperforming