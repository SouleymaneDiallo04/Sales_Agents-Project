import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
import mysql.connector
from typing import Dict, Any, Optional

class EcommercePricingEnv(gym.Env):
    """
    Environnement Gymnasium pour le pricing dynamique d'un produit e-commerce
    Spécialisé pour PROD_001 - Smartphone Galaxy X
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, product_id: str = 'PROD_001', render_mode: Optional[str] = None):
        super().__init__()
        
        self.product_id = product_id
        self.render_mode = render_mode
        
        # Espace d'état normalisé [0,1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 5 actions discrètes: -10%, -5%, 0%, +5%, +10%
        self.action_space = spaces.Discrete(5)
        
        # Données du produit
        self.product_data = self._load_product_data()
        self.current_state = None
        self.current_step = 0
        self.max_steps = 365  # 1 an de simulation
        self.total_profit = 0.0
        
        # Connexion DB (optionnelle pour simulation)
        self.db_connection = None  # INITIALISATION ICI
        self._init_database()
        
    def _init_database(self):
        """Initialise la connexion à la base de données"""
        try:
            self.db_connection = mysql.connector.connect(
                host='localhost',
                database='rl_data_base',
                user='root',
                password=''
            )
            print("✅ Connexion DB établie")
        except Exception as e:
            print(f"⚠️  Connexion DB échouée, mode simulation: {e}")
            self.db_connection = None
    
    def _load_product_data(self) -> Dict[str, Any]:
        """Charge les données du produit depuis la base ou valeurs par défaut"""
        default_data = {
            'product_id': 'PROD_001',
            'product_name': 'Smartphone Galaxy X',
            'cost_price': 450.00,
            'min_price': 495.00,
            'max_price': 900.00,
            'initial_stock': 100
        }
        
        # CORRECTION : Vérifier si db_connection existe ET est connecté
        if hasattr(self, 'db_connection') and self.db_connection is not None and self.db_connection.is_connected():
            try:
                cursor = self.db_connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM products WHERE product_id = %s", (self.product_id,))
                result = cursor.fetchone()
                cursor.close()
                return result if result else default_data
            except Exception as e:
                print(f"Erreur chargement produit: {e}")
                print("Utilisation des données par défaut")
                return default_data
        else:
            # Pas de connexion DB, utiliser les valeurs par défaut
            print("ℹ️  Mode simulation - données par défaut")
            return default_data
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Réinitialise l'environnement à l'état initial"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_profit = 0.0
        self.current_state = self._get_initial_state()
        
        # Sauvegarde état initial en DB (si connecté)
        if self.db_connection and self.db_connection.is_connected():
            try:
                self._save_state_to_db(self.current_state)
            except Exception as e:
                print(f"Erreur sauvegarde état initial: {e}")
        
        return self._normalize_state(self.current_state), {}
    
    def _get_initial_state(self) -> Dict[str, float]:
        """Génère un état initial réaliste"""
        return {
            'stock_ratio': 0.8,           # 80% du stock initial
            'price_ratio': 0.6,           # 60% de la marge max
            'competitor_1_ratio': 0.95,   # 95% de notre prix
            'competitor_2_ratio': 1.05,   # 105% de notre prix
            'demand_trend': 1.0,          # Demande normale
            'day_of_week': datetime.now().weekday() / 6.0,  # Normalisé
            'seasonality': 0.5,           # Saisonnalité moyenne
            'stock_risk': 0.1,            # Faible risque rupture
            'market_volatility': 0.2      # Volatilité marché
        }
    
    def step(self, action: int):
        """Exécute une action et retourne le résultat"""
        self.current_step += 1
        
        # Appliquer l'action de pricing
        price_change = self._apply_price_action(action)
        new_price_ratio = self._calculate_new_price_ratio(price_change)
        
        # Simuler la demande et les résultats
        demand = self._simulate_demand(new_price_ratio)
        reward = self._calculate_reward(demand, new_price_ratio, price_change)
        self.total_profit += reward * 100  # Dénormaliser
        
        # Mettre à jour l'état
        self.current_state = self._update_state(demand, new_price_ratio)
        
        # Vérifier conditions de fin
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Sauvegarder l'expérience (si connecté)
        if self.db_connection and self.db_connection.is_connected():
            try:
                self._save_experience(action, reward)
            except Exception as e:
                print(f"Erreur sauvegarde expérience: {e}")
        
        info = {
            'step': self.current_step,
            'demand': demand,
            'price_change': price_change,
            'total_profit': self.total_profit
        }
        
        return self._normalize_state(self.current_state), reward, terminated, truncated, info
    
    def _apply_price_action(self, action: int) -> float:
        """Convertit l'action en changement de prix"""
        action_map = {0: -0.10, 1: -0.05, 2: 0.00, 3: 0.05, 4: 0.10}
        return action_map.get(action, 0.00)
    
    def _calculate_new_price_ratio(self, price_change: float) -> float:
        """Calcule le nouveau ratio de prix avec contraintes"""
        current_ratio = self.current_state['price_ratio']
        new_ratio = current_ratio * (1 + price_change)
        return np.clip(new_ratio, 0.1, 0.9)  # Bornes métier
    
    def _simulate_demand(self, price_ratio: float) -> float:
        """Simule la demande basée sur le prix et l'état du marché"""
        base_demand = 15.0
        
        # Élasticité-prix
        price_elasticity = -1.8
        price_effect = (price_ratio / 0.6) ** price_elasticity  # Référence à 0.6
        
        # Effet concurrentiel
        comp_effect = 1.2 - abs(self.current_state['competitor_1_ratio'] - 1.0)
        
        # Effet saisonnier
        seasonal_effect = 1.0 + 0.3 * np.sin(2 * np.pi * self.current_step / 30)
        
        # Effet jour de semaine
        day_effect = 1.3 if self.current_state['day_of_week'] > 0.66 else 1.0  # Weekend
        
        # Bruit aléatoire
        noise = np.random.normal(1.0, 0.15)
        
        demand = base_demand * price_effect * comp_effect * seasonal_effect * day_effect * noise
        return max(0.0, demand)
    
    def _calculate_reward(self, demand: float, price_ratio: float, price_change: float) -> float:
        """Calcule la récompense (profit + pénalités)"""
        # Calcul du prix réel
        cost = self.product_data['cost_price']
        min_price = self.product_data['min_price']
        max_price = self.product_data['max_price']
        price_range = max_price - min_price
        
        current_price = min_price + (price_ratio * price_range)
        
        # Profit
        margin = current_price - cost
        profit = margin * demand
        
        # Pénalités
        stock_penalty = 0.0
        if self.current_state['stock_ratio'] < 0.1:  # Rupture imminente
            stock_penalty = 100.0
        
        change_penalty = abs(price_change) * 50.0  # Pénalité changement brutal
        
        # Récompense normalisée
        reward = (profit - stock_penalty - change_penalty) / 100.0
        
        return float(reward)
    
    def _update_state(self, demand: float, new_price_ratio: float) -> Dict[str, float]:
        """Met à jour l'état après l'action"""
        # Consommation stock
        stock_consumption = demand / self.product_data['initial_stock']
        new_stock_ratio = max(0.0, self.current_state['stock_ratio'] - stock_consumption)
        
        # Évolution concurrents (simulée)
        comp1_move = np.random.choice([-0.03, 0.0, 0.03], p=[0.3, 0.4, 0.3])
        comp2_move = np.random.choice([-0.02, 0.0, 0.02], p=[0.4, 0.2, 0.4])
        
        return {
            'stock_ratio': new_stock_ratio,
            'price_ratio': new_price_ratio,
            'competitor_1_ratio': np.clip(self.current_state['competitor_1_ratio'] + comp1_move, 0.7, 1.3),
            'competitor_2_ratio': np.clip(self.current_state['competitor_2_ratio'] + comp2_move, 0.7, 1.3),
            'demand_trend': demand / 20.0,  # Normalisé
            'day_of_week': (self.current_state['day_of_week'] + 1/6.0) % 1.0,
            'seasonality': 0.5 + 0.3 * np.sin(2 * np.pi * self.current_step / 365),
            'stock_risk': 1.0 - new_stock_ratio,
            'market_volatility': self.current_state['market_volatility'] + np.random.normal(0, 0.02)
        }
    
    def _normalize_state(self, state: Dict[str, float]) -> np.ndarray:
        """Normalise l'état pour l'agent RL"""
        return np.array([
            state['stock_ratio'],
            state['price_ratio'],
            state['competitor_1_ratio'],
            state['competitor_2_ratio'],
            state['demand_trend'],
            state['day_of_week'],
            state['seasonality'],
            state['stock_risk'],
            state['market_volatility']
        ], dtype=np.float32)
    
    def _is_terminated(self) -> bool:
        """Vérifie si l'épisode est terminé"""
        return self.current_step >= self.max_steps
    
    def _is_truncated(self) -> bool:
        """Vérifie si l'épisode doit être interrompu"""
        return (self.current_state['stock_ratio'] <= 0.0 or 
                self.total_profit < -1000)  # Pertes trop importantes
    
    def _save_state_to_db(self, state: Dict[str, float]):
        """Sauvegarde l'état dans la base de données"""
        if not self.db_connection or not self.db_connection.is_connected():
            return
        
        try:
            cursor = self.db_connection.cursor()
            query = """
            INSERT INTO pricing_states 
            (product_id, current_stock, current_price, competitor_1_price, 
             competitor_2_price, demand_1d, day_of_week, state_vector)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Calcul des prix réels pour la DB
            cost = self.product_data['cost_price']
            min_price = self.product_data['min_price']
            max_price = self.product_data['max_price']
            price_range = max_price - min_price
            
            current_price = min_price + (state['price_ratio'] * price_range)
            comp1_price = current_price * state['competitor_1_ratio']
            comp2_price = current_price * state['competitor_2_ratio']
            
            values = (
                self.product_id,
                int(state['stock_ratio'] * self.product_data['initial_stock']),
                float(current_price),
                float(comp1_price),
                float(comp2_price),
                int(state['demand_trend'] * 20),
                int(state['day_of_week'] * 6),
                str(state)  # state_vector comme JSON
            )
            
            cursor.execute(query, values)
            self.db_connection.commit()
            cursor.close()
            
        except Exception as e:
            print(f"Erreur sauvegarde état DB: {e}")
    
    def _save_experience(self, action: int, reward: float):
        """Sauvegarde l'expérience pour le replay buffer"""
        # Implémentation simplifiée - à compléter selon vos besoins
        pass
    
    def render(self):
        """Affiche l'état actuel"""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step:3d} | "
                  f"Stock: {self.current_state['stock_ratio']:.2f} | "
                  f"Price Ratio: {self.current_state['price_ratio']:.2f} | "
                  f"Profit: {self.total_profit:.2f}€")
    
    def close(self):
        """Ferme l'environnement et la connexion DB"""
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            print("✅ Connexion DB fermée")