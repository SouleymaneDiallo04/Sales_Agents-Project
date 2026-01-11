import mysql.connector
from mysql.connector import Error
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestionnaire unique pour la base RL - Version MySQL seule"""
    
    def __init__(self, host='127.0.0.1', database='rl_data_base', 
                 user='root', password=''):
        self.config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'charset': 'utf8mb4'
        }
        self.connection = None
        self.cache = {}  # Cache simple en mémoire au lieu de Redis
        self.connect()
    
    def connect(self):
        """Établir connexion MySQL uniquement"""
        try:
            # MySQL seulement - pas de Redis
            self.connection = mysql.connector.connect(**self.config)
            logger.info(f"✅ Connecté à MySQL: {self.config['database']}")
            logger.info("⚠️ Redis désactivé - utilisation de MySQL seulement")
            
        except Error as e:
            logger.error(f"❌ Erreur DB: {e}")
            raise
    
    def get_product(self, product_id: str) -> Optional[Dict]:
        """Récupérer produit depuis DB"""
        # Check cache simple
        cache_key = f"product:{product_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Query DB - CORRIGÉ pour correspondre à votre structure
        query = """
            SELECT product_id, product_name, cost_price, min_price, max_price, 
                   category, supplier_id, is_active, created_at, updated_at
            FROM products 
            WHERE product_id = %s AND is_active = 1
        """
        result = self._execute_query(query, (product_id,), fetch_one=True)
        
        if result:
            # Cache simple en mémoire
            self.cache[cache_key] = result
        
        return result
    
    def save_pricing_decision(self, decision_data: Dict) -> int:
        """Sauvegarder décision de pricing - CORRIGÉ"""
        query = """
            INSERT INTO pricing_decisions 
            (state_id, action_taken, recommended_price, 
             confidence_score, model_id, timestamp, is_executed)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            decision_data.get('state_id', 0),
            decision_data['action_taken'],
            decision_data['recommended_price'],
            decision_data.get('confidence_score', 0.0),
            decision_data['model_id'],
            decision_data.get('timestamp', datetime.now()),
            decision_data.get('is_executed', 0)
        )
        
        return self._execute_query(query, params, fetch=False, return_last_id=True)
    
    def save_pricing_state(self, state_data: Dict) -> int:
        """Sauvegarder état de pricing - IMPORTANT pour votre structure"""
        query = """
            INSERT INTO pricing_states 
            (product_id, timestamp, current_stock, current_price,
             competitor_1_price, competitor_2_price, competitor_3_price,
             demand_1d, demand_7d_avg, day_of_week, seasonality_index,
             stock_ratio, price_ratio, state_vector)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Préparer state_vector JSON
        state_vector = json.dumps(state_data.get('state_vector', {}))
        
        params = (
            state_data['product_id'],
            state_data.get('timestamp', datetime.now()),
            state_data['current_stock'],
            state_data['current_price'],
            state_data.get('competitor_1_price'),
            state_data.get('competitor_2_price'),
            state_data.get('competitor_3_price'),
            state_data.get('demand_1d'),
            state_data.get('demand_7d_avg'),
            state_data.get('day_of_week'),
            state_data.get('seasonality_index'),
            state_data.get('stock_ratio'),
            state_data.get('price_ratio'),
            state_vector
        )
        
        return self._execute_query(query, params, fetch=False, return_last_id=True)
    
    def log_experience(self, experience: Dict):
        """Sauvegarder expérience pour replay buffer"""
        query = """
            INSERT INTO experience_replay 
            (state_id, action_taken, reward_value, next_state_id, 
             is_terminal, model_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self._execute_query(query, (
            experience['state_id'],
            experience['action_taken'],
            experience['reward_value'],
            experience.get('next_state_id'),
            experience.get('is_terminal', 0),
            experience['model_id'],
            datetime.now()
        ), fetch=False)
    
    def get_training_batch(self, batch_size=64):
        """Récupérer batch d'expériences pour training"""
        query = """
            SELECT * FROM experience_replay 
            ORDER BY RAND() 
            LIMIT %s
        """
        return self._execute_query(query, (batch_size,))
    
    def get_competitors(self):
        """Récupérer liste des concurrents"""
        query = "SELECT competitor_id, competitor_name FROM competitors WHERE is_active = 1"
        return self._execute_query(query)
    
    def get_rl_models(self, active_only=True):
        """Récupérer modèles RL"""
        if active_only:
            query = "SELECT * FROM rl_models WHERE is_active = 1"
        else:
            query = "SELECT * FROM rl_models"
        return self._execute_query(query)
    
    def log_system_event(self, component: str, log_level: str, 
                        message: str, metrics: Dict = None):
        """
        Logger un événement système dans la base de données
        
        Args:
            component: Composant qui génère l'événement (ex: 'pricing_agent', 'api')
            log_level: Niveau de log ('INFO', 'WARNING', 'ERROR', 'DEBUG')
            message: Message descriptif de l'événement
            metrics: Données métriques additionnelles (optionnel)
        """
        try:
            # Vérifier si la table existe, sinon la créer
            self._ensure_system_monitoring_table()
            
            # Préparer les données
            metrics_json = json.dumps(metrics) if metrics else None
            
            # Insérer l'événement
            query = """
                INSERT INTO system_monitoring 
                (component, log_level, message, metrics, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """
            params = (
                component,
                log_level.upper(),
                message,
                metrics_json,
                datetime.now()
            )
            
            self._execute_query(query, params, fetch=False)
            logger.debug(f"✅ Événement système loggé: {component} - {log_level}")
            
        except Exception as db_error:
            logger.warning(f"❌ Erreur DB lors du logging système: {db_error}")
            # Méthode de secours : écriture dans fichier log
            self._fallback_file_logging(component, log_level, message, metrics)
    
    def _ensure_system_monitoring_table(self):
        """S'assurer que la table system_monitoring existe"""
        create_table_query = """
            CREATE TABLE IF NOT EXISTS system_monitoring (
                id INT AUTO_INCREMENT PRIMARY KEY,
                component VARCHAR(100) NOT NULL,
                log_level VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                metrics JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_component (component),
                INDEX idx_log_level (log_level),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        self._execute_query(create_table_query, fetch=False)
    
    def _fallback_file_logging(self, component: str, log_level: str, 
                             message: str, metrics: Dict = None):
        """Méthode de secours : logging dans fichier quand DB indisponible"""
        try:
            import os
            from pathlib import Path
            
            # Créer dossier logs s'il n'existe pas
            log_dir = Path("logs/system_monitoring")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Nom du fichier basé sur la date
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = log_dir / f"system_events_{today}.log"
            
            # Formater le message
            timestamp = datetime.now().isoformat()
            metrics_str = f" | metrics={json.dumps(metrics)}" if metrics else ""
            
            log_entry = f"[{timestamp}] {log_level.upper()} | {component} | {message}{metrics_str}\n"
            
            # Écrire dans le fichier
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
            logger.info(f"📄 Événement loggé en secours dans: {log_file}")
            
        except Exception as file_error:
            logger.error(f"❌ Échec logging de secours: {file_error}")
    
    def _execute_query(self, query: str, params: tuple = None, 
                      fetch_one: bool = False, fetch: bool = True,
                      return_last_id: bool = False):
        """Exécuter requête SQL"""
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if fetch:
                if fetch_one:
                    result = cursor.fetchone()
                else:
                    result = cursor.fetchall()
            else:
                self.connection.commit()
                result = cursor.rowcount
            
            if return_last_id:
                cursor.execute("SELECT LAST_INSERT_ID()")
                result = cursor.fetchone()['LAST_INSERT_ID()']
            
            return result
            
        except Exception as e:
            logger.error(f"SQL Error: {e} - Query: {query}")
            if self.connection:
                self.connection.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
    
    def close(self):
        """Fermer connexion MySQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion MySQL fermée")