import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import os

class PricingTensorboardCallback(BaseCallback):
    """
    Callback personnalisé pour le monitoring avancé du pricing
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_prices = []
        self.episode_demands = []
        
    def _on_training_start(self) -> None:
        # Initialise le format TensorBoard
        self._log_freq = 1000  # Log toutes les 1000 steps
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter for formatter in output_formats 
            if isinstance(formatter, TensorBoardOutputFormat)
        )
    
    def _on_step(self) -> bool:
        # Log des métriques custom
        if self.n_calls % self._log_freq == 0:
            self._log_custom_metrics()
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Métriques à la fin de chaque rollout
        if len(self.model.ep_info_buffer) > 0:
            latest_episode = self.model.ep_info_buffer[-1]
            if 'r' in latest_episode:
                self.episode_rewards.append(latest_episode['r'])
                
                # Log dans TensorBoard
                self.logger.record('rollout/episode_reward', latest_episode['r'])
                self.logger.record('rollout/episode_length', latest_episode['l'])
    
    def _log_custom_metrics(self):
        """Log des métriques business personnalisées"""
        try:
            # Récupère les données de l'environnement
            if hasattr(self.model, 'env') and self.model.env is not None:
                env = self.model.env.envs[0] if hasattr(self.model.env, 'envs') else self.model.env
                
                if hasattr(env, 'current_state'):
                    state = env.current_state
                    
                    # Métriques de pricing
                    self.logger.record('business/stock_ratio', state.get('stock_ratio', 0))
                    self.logger.record('business/price_ratio', state.get('price_ratio', 0))
                    self.logger.record('business/competitor_ratio', state.get('competitor_1_ratio', 0))
                    self.logger.record('business/demand_trend', state.get('demand_trend', 0))
                    
                    # Métriques dérivées
                    current_price = env.product_data['min_price'] + \
                                  (state.get('price_ratio', 0) * \
                                   (env.product_data['max_price'] - env.product_data['min_price']))
                    
                    self.logger.record('business/current_price', current_price)
                    self.logger.record('business/total_profit', getattr(env, 'total_profit', 0))
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️  Erreur logging métriques: {e}")

class TrainingMonitor:
    """Monitor d'entraînement avec sauvegarde des métriques"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics_history = {
            'rewards': [],
            'losses': [],
            'episode_lengths': []
        }
    
    def log_episode(self, reward: float, length: int, info: dict = None):
        """Log les métriques d'un épisode"""
        self.metrics_history['rewards'].append(reward)
        self.metrics_history['episode_lengths'].append(length)
        
        if info:
            for key, value in info.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)
    
    def get_summary(self) -> dict:
        """Retourne un résumé des métriques"""
        return {
            'mean_reward': np.mean(self.metrics_history['rewards']),
            'std_reward': np.std(self.metrics_history['rewards']),
            'max_reward': np.max(self.metrics_history['rewards']),
            'min_reward': np.min(self.metrics_history['rewards']),
            'total_episodes': len(self.metrics_history['rewards'])
        }