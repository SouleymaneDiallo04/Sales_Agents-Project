"""
Trainer spécialisé pour fine-tuning ventes
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, List, Optional
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class SalesTrainer:
    """Trainer pour fine-tuning agents de vente"""
    
    def __init__(self, db_connection, model_type: str = "PPO"):
        self.db = db_connection
        self.model_type = model_type
        self.model = None
        self.training_history = []
        
    def load_pretrained(self, model_path: Optional[str] = None):
        """Charger modèle pré-entraîné"""
        if model_path:
            try:
                if self.model_type == "PPO":
                    self.model = PPO.load(model_path)
                elif self.model_type == "DQN":
                    self.model = DQN.load(model_path)
                logger.info(f"✅ Modèle chargé: {model_path}")
            except Exception as e:
                logger.error(f"❌ Erreur chargement: {e}")
                self.model = None
        return self.model
    
    def create_sales_environment(self, product_id: str):
        """Créer environnement de vente"""
        from environments.pricing_env import PricingEnvironment
        
        env = PricingEnvironment()
        env = DummyVecEnv([lambda: env])
        
        return env
    
    def fine_tune_on_product(self, product_id: str, total_steps: int = 20000):
        """Fine-tuning sur un produit spécifique"""
        
        logger.info(f"🎯 Fine-tuning sur produit {product_id}")
        
        # Créer environnement
        env = self.create_sales_environment(product_id)
        
        # Charger ou créer modèle
        if not self.model:
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log=f"./logs/sales_training/{product_id}/"
            )
        
        # Callback personnalisé
        callback = SalesTrainingCallback(self.db, product_id)
        
        # Entraînement
        self.model.learn(
            total_timesteps=total_steps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
            tb_log_name=f"sales_fine_tune_{product_id}"
        )
        
        # Sauvegarder le modèle
        save_path = f"./pretrained_models/custom_trained/sales_{product_id}_{datetime.now().strftime('%Y%m%d')}"
        self.model.save(save_path)
        
        logger.info(f"💾 Modèle sauvegardé: {save_path}")
        
        # Enregistrer dans DB (avec gestion d'erreur)
        self._save_training_record(product_id, total_steps, save_path)
        
        return self.model
    
    def _save_training_record(self, product_id: str, steps: int, model_path: str):
        """Sauvegarder record d'entraînement avec JSON valide"""
        
        try:
            # 1. Créer un JSON VALIDE pour hyperparameters
            hyperparams = {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gae_lambda": 0.95,
                "clip_range": 0.2
            }
            hyperparams_json = json.dumps(hyperparams, ensure_ascii=False)
            
            # 2. JSON pour performance_metrics (vide)
            perf_metrics_json = "{}"
            
            # 3. JSON pour training_history
            training_hist = {
                "product_id": product_id,
                "total_steps": steps,
                "model_path": model_path,
                "training_date": datetime.now().isoformat(),
                "model_type": self.model_type
            }
            training_hist_json = json.dumps(training_hist, ensure_ascii=False)
            
            # 4. Model ID
            model_id = f"sales_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 5. Requête SQL avec TOUTES les colonnes requises
            query = """
                INSERT INTO rl_models 
                (model_id, algorithm_name, hyperparameters, model_path, created_at, is_active,
                 performance_metrics, training_history)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # 6. Exécuter
            self.db._execute_query(query, (
                model_id,
                self.model_type,
                hyperparams_json,
                model_path,
                datetime.now(),
                1,  # is_active = True
                perf_metrics_json,
                training_hist_json
            ), fetch=False)
            
            logger.info(f"✅ Record sauvegardé dans DB: {model_id}")
            
            # 7. Sauvegarder aussi localement
            record = {
                'model_id': model_id,
                'product_id': product_id,
                'training_date': datetime.now(),
                'total_steps': steps,
                'model_path': model_path,
                'model_type': self.model_type,
                'hyperparameters': hyperparams
            }
            self.training_history.append(record)
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde DB: {e}")
            # Sauvegarde locale de secours
            self._save_local_fallback(product_id, steps, model_path)
    
    def _save_local_fallback(self, product_id: str, steps: int, model_path: str):
        """Sauvegarde locale en cas d'erreur DB"""
        record = {
            'product_id': product_id,
            'training_date': datetime.now().isoformat(),
            'total_steps': steps,
            'model_path': model_path,
            'model_type': self.model_type
        }
        
        os.makedirs("./logs/training_records", exist_ok=True)
        filename = f"sales_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = f"./logs/training_records/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
        
        logger.info(f"💾 Record sauvegardé localement: {filepath}")
        self.training_history.append(record)
    
    def batch_fine_tuning(self, product_ids: List[str], steps_per_product: int = 10000):
        """Fine-tuning batch sur plusieurs produits"""
        
        results = {}
        
        for product_id in product_ids:
            logger.info(f"🔄 Fine-tuning produit: {product_id}")
            
            try:
                model = self.fine_tune_on_product(product_id, steps_per_product)
                results[product_id] = {
                    'success': True,
                    'model_path': f"./pretrained_models/custom_trained/sales_{product_id}",
                    'steps': steps_per_product
                }
            except Exception as e:
                logger.error(f"❌ Erreur sur {product_id}: {e}")
                results[product_id] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def evaluate_model(self, product_id: str, n_episodes: int = 10) -> Dict:
        """Évaluer modèle fine-tuné"""
        
        from environments.pricing_env import PricingEnvironment
        
        env = PricingEnvironment()
        
        if not self.model:
            logger.error("❌ Modèle non chargé")
            return {}
        
        results = {
            'total_profit': 0,
            'total_sales': 0,
            'avg_profit_per_episode': 0,
            'price_changes': [],
            'decisions': []
        }
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_profit = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)  # Nouvelle API Gym
                done = terminated or truncated  # Logique de fin d'épisode
                
                if 'profit' in info:
                    episode_profit += info['profit']
                    results['price_changes'].append(info.get('price_change', 0))
                    results['decisions'].append({
                        'action': action,
                        'price': info.get('price', 0),
                        'profit': info['profit']
                    })
            
            results['total_profit'] += episode_profit
            results['total_sales'] += 1
        
        if n_episodes > 0:
            results['avg_profit_per_episode'] = results['total_profit'] / n_episodes
        
        return results

class SalesTrainingCallback(BaseCallback):
    """Callback personnalisé pour training ventes"""
    
    def __init__(self, db_connection, product_id: str, save_freq: int = 1000):
        super().__init__()
        self.db = db_connection
        self.product_id = product_id
        self.save_freq = save_freq
        self.best_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Évaluer modèle
            avg_reward = self._evaluate_current_model()
            
            # Sauvegarder si meilleur
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.model.save(f"./pretrained_models/custom_trained/best_{self.product_id}")
                
                logger.info(f"🏆 Nouveau meilleur modèle: reward={avg_reward:.2f}")
                
                # Log dans DB - CORRECTION ICI: level -> log_level
                self.db.log_system_event(
                    component="training",
                    log_level="INFO",  # CORRECTION: changé de level à log_level
                    message=f"New best model for {self.product_id}",
                    metrics={'reward': avg_reward, 'step': self.num_timesteps}
                )
        
        return True
    
    def _evaluate_current_model(self, n_episodes: int = 3) -> float:
        """Évaluer modèle rapidement"""
        from environments.pricing_env import PricingEnvironment

        env = PricingEnvironment()
        total_reward = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)  # Nouvelle API Gym
                done = terminated or truncated  # Logique de fin d'épisode
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / n_episodes if n_episodes > 0 else 0