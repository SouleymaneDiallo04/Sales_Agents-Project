import os
from typing import Dict, Any

class Config:
    """Configuration globale du projet RL Pricing"""
    
    # Chemins
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Produit
    PRODUCT_ID = "PROD_001"
    
    # Environnement
    ENV_CONFIG = {
        'max_steps': 365,
        'initial_stock': 100,
        'cost_price': 450.0,
        'min_price': 495.0, 
        'max_price': 900.0
    }
    
    # Entraînement
    TRAINING_CONFIG = {
        'total_timesteps': 100000,
        'eval_freq': 10000,
        'n_eval_episodes': 10,
        'save_freq': 50000
    }
    
    # Algorithmes
    ALGORITHMS = {
        'ppo': {
            'n_steps': 2048,
            'batch_size': 64,
            'learning_rate': 0.0003,
            'ent_coef': 0.01
        },
        'dqn': {
            'learning_rate': 0.0001,
            'buffer_size': 100000,
            'exploration_fraction': 0.1
        },
        'a2c': {
            'n_steps': 5,
            'learning_rate': 0.0007,
            'ent_coef': 0.0
        }
    }
    
    @classmethod
    def get_tensorboard_logdir(cls, algo_name: str) -> str:
        """Retourne le chemin des logs TensorBoard"""
        return os.path.join(cls.LOG_DIR, "tensorboard", algo_name)
    
    @classmethod
    def get_model_path(cls, algo_name: str) -> str:
        """Retourne le chemin de sauvegarde du modèle"""
        return os.path.join(cls.MODEL_DIR, f"{algo_name}_pricing_model")
    
    @classmethod
    def create_directories(cls):
        """Crée tous les répertoires nécessaires"""
        directories = [
            cls.MODEL_DIR,
            cls.LOG_DIR,
            os.path.join(cls.LOG_DIR, "tensorboard"),
            os.path.join(cls.LOG_DIR, "evaluations"),
            os.path.join(cls.MODEL_DIR, "checkpoints"),
            os.path.join(cls.MODEL_DIR, "best_models")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ Répertoires créés")

# Initialisation
config = Config()