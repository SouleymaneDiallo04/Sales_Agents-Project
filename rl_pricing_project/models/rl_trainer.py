import os
import yaml
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from environments.pricing_env import EcommercePricingEnv
from utils.monitoring import PricingTensorboardCallback
import gymnasium as gym

class RLTrainer:
    """Classe pour l'entraînement des modèles RL avec SB3 Zoo"""
    
    def __init__(self, algo_name: str = 'ppo', config_path: str = None):
        self.algo_name = algo_name.lower()
        self.config_path = config_path
        self.model = None
        self.env = None
        
        # Mapping algorithmes
        self.algo_map = {
            'ppo': PPO,
            'dqn': DQN, 
            'a2c': A2C
        }
    
    def create_env(self, n_envs: int = 1, product_id: str = 'PROD_001'):
        """Crée l'environnement vectorisé"""
        def make_env():
            env = EcommercePricingEnv(product_id=product_id)
            env = Monitor(env)  # Pour le tracking
            return env
        
        # Environnement vectorisé
        self.env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # Normalisation (optionnelle)
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        
        return self.env
    
    def load_hyperparameters(self, config_file: str):
        """Charge les hyperparamètres depuis un fichier YAML"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Fichier de config non trouvé: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extraction config spécifique à l'environnement
        env_config = config.get('EcommercePricing-v1', {})
        return env_config
    
    def create_model(self, env=None, hyperparams=None):
        """Crée le modèle RL"""
        if env is None:
            env = self.env
        
        if hyperparams is None:
            # Chargement depuis fichier de config
            config_file = f'models/hyperparameters/{self.algo_name}.yml'
            hyperparams = self.load_hyperparameters(config_file)
        
        # Filtrage des paramètres spécifiques à l'algorithme
        algo_class = self.algo_map[self.algo_name]
        
        # Création du modèle
        self.model = algo_class(
            policy='MlpPolicy',
            env=env,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
            **hyperparams
        )
        
        return self.model
    
    def train(self, total_timesteps: int = 100000, 
              eval_freq: int = 10000, 
              save_freq: int = 50000):
        """Entraîne le modèle avec callbacks"""
        
        # Callbacks
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=f'./models/best_{self.algo_name}/',
            log_path='./logs/evaluations/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f'./models/checkpoints/',
            name_prefix=f'{self.algo_name}_model'
        )
        
        pricing_callback = PricingTensorboardCallback()
        
        # Entraînement
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, pricing_callback],
            tb_log_name=f"{self.algo_name}_pricing"
        )
        
        # Sauvegarde finale
        self.model.save(f"./models/final_{self.algo_name}_pricing")
        
        return self.model
    
    def evaluate(self, num_episodes: int = 10):
        """Évalue le modèle entraîné"""
        if self.model is None:
            raise ValueError("Modèle non entraîné")
        
        eval_env = self.create_env(n_envs=1)
        
        rewards = []
        for episode in range(num_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            print(f"Épisode {episode + 1}: Reward = {episode_reward:.2f}")
        
        print(f"Reward moyen sur {num_episodes} épisodes: {np.mean(rewards):.2f}")
        return rewards

# Usage example
if __name__ == "__main__":
    trainer = RLTrainer(algo_name='ppo')
    trainer.create_env(n_envs=4)
    trainer.create_model()
    trainer.train(total_timesteps=100000)