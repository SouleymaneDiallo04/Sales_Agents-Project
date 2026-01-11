from .mysql_config import db_config
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class RLRepository:
    """Repository pour l'accès aux données RL dans la base"""
    
    def __init__(self):
        self.db = db_config
    
    def save_pricing_state(self, state_data: Dict[str, Any]) -> int:
        """Sauvegarde un état de pricing dans pricing_states"""
        query = """
        INSERT INTO pricing_states 
        (product_id, timestamp, current_stock, current_price, 
         competitor_1_price, competitor_2_price, competitor_3_price,
         demand_1d, demand_7d_avg, day_of_week, seasonality_index,
         stock_ratio, price_ratio, state_vector)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            state_data.get('product_id'),
            state_data.get('timestamp', datetime.now()),
            state_data.get('current_stock'),
            state_data.get('current_price'),
            state_data.get('competitor_1_price'),
            state_data.get('competitor_2_price'),
            state_data.get('competitor_3_price'),
            state_data.get('demand_1d'),
            state_data.get('demand_7d_avg'),
            state_data.get('day_of_week'),
            state_data.get('seasonality_index'),
            state_data.get('stock_ratio'),
            state_data.get('price_ratio'),
            json.dumps(state_data.get('state_vector', {}))
        )
        
        return self.db.execute_query(query, values)
    
    def save_pricing_decision(self, decision_data: Dict[str, Any]) -> int:
        """Sauvegarde une décision de pricing"""
        query = """
        INSERT INTO pricing_decisions 
        (state_id, action_taken, recommended_price, confidence_score, model_id)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        values = (
            decision_data.get('state_id'),
            decision_data.get('action_taken'),
            decision_data.get('recommended_price'),
            decision_data.get('confidence_score', 0.0),
            decision_data.get('model_id')
        )
        
        return self.db.execute_query(query, values)
    
    def save_pricing_result(self, result_data: Dict[str, Any]) -> int:
        """Sauvegarde les résultats d'une décision"""
        query = """
        INSERT INTO pricing_results 
        (decision_id, period_start, period_end, units_sold, revenue, 
         profit, stock_level, reward_value, competitor_actions, market_conditions)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            result_data.get('decision_id'),
            result_data.get('period_start'),
            result_data.get('period_end'),
            result_data.get('units_sold', 0),
            result_data.get('revenue', 0.0),
            result_data.get('profit', 0.0),
            result_data.get('stock_level', 0),
            result_data.get('reward_value', 0.0),
            json.dumps(result_data.get('competitor_actions', {})),
            json.dumps(result_data.get('market_conditions', {}))
        )
        
        return self.db.execute_query(query, values)
    
    def save_experience(self, experience_data: Dict[str, Any]) -> int:
        """Sauvegarde une expérience pour le replay buffer"""
        query = """
        INSERT INTO experience_replay 
        (state_id, action_taken, reward_value, next_state_id, is_terminal, model_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        values = (
            experience_data.get('state_id'),
            experience_data.get('action_taken'),
            experience_data.get('reward_value'),
            experience_data.get('next_state_id'),
            experience_data.get('is_terminal', False),
            experience_data.get('model_id')
        )
        
        return self.db.execute_query(query, values)
    
    def get_product_data(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les données d'un produit"""
        query = "SELECT * FROM products WHERE product_id = %s"
        result = self.db.execute_query(query, (product_id,), fetch=True)
        return result[0] if result else None
    
    def get_competitor_prices(self, product_id: str) -> List[Dict[str, Any]]:
        """Récupère les prix des concurrents (simulé pour l'instant)"""
        # En production, cela viendrait d'APIs de scraping
        return [
            {'competitor_id': 'COMP_001', 'price': 689.00},
            {'competitor_id': 'COMP_002', 'price': 719.00},
            {'competitor_id': 'COMP_003', 'price': 705.00}
        ]
    
    def get_training_experiences(self, model_id: str, limit: int = 10000) -> List[Dict[str, Any]]:
        """Récupère des expériences pour l'entraînement"""
        query = """
        SELECT * FROM experience_replay 
        WHERE model_id = %s 
        ORDER BY RAND() 
        LIMIT %s
        """
        return self.db.execute_query(query, (model_id, limit), fetch=True)
    
    def get_performance_metrics(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """Récupère les métriques de performance d'un modèle"""
        query = """
        SELECT 
            AVG(reward_value) as avg_reward,
            AVG(profit) as avg_profit,
            AVG(units_sold) as avg_units_sold,
            COUNT(*) as total_decisions
        FROM pricing_results pr
        JOIN pricing_decisions pd ON pr.decision_id = pd.decision_id
        WHERE pd.model_id = %s 
        AND pr.period_start >= DATE_SUB(NOW(), INTERVAL %s DAY)
        """
        
        result = self.db.execute_query(query, (model_id, days), fetch=True)
        return result[0] if result else {}
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Nettoie les données anciennes"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        queries = [
            "DELETE FROM pricing_results WHERE period_start < %s",
            "DELETE FROM pricing_decisions WHERE created_at < %s", 
            "DELETE FROM pricing_states WHERE timestamp < %s",
            "DELETE FROM experience_replay WHERE created_at < %s"
        ]
        
        for query in queries:
            try:
                self.db.execute_query(query, (cutoff_date,))
                print(f"✅ Nettoyage: {query.split(' ')[1]}")
            except Exception as e:
                print(f"❌ Erreur nettoyage: {e}")