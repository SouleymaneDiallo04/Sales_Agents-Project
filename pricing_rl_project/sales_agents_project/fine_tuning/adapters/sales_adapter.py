"""
Adapter pour transformer modèles génériques en agents de vente
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class SalesAdapter:
    """Adapte les modèles RL génériques au domaine des ventes"""
    
    def __init__(self, base_model, sales_env):
        self.base_model = base_model
        self.sales_env = sales_env
        self.adapted_model = None
        
    def adapt_model_architecture(self):
        """Adapter l'architecture du modèle pour les ventes"""
        
        # Sauvegarder les poids originaux
        original_state_dict = self.base_model.policy.state_dict()
        
        # Créer nouvelle politique adaptée
        class SalesPolicy(ActorCriticPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Ajouter couches spécialisées pour ventes
                self.sales_head = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.action_space.n)
                )
            
            def forward(self, obs, deterministic=False):
                # Forward pass avec spécialisation ventes
                features = self.extract_features(obs)
                sales_features = self.sales_head(features)
                return sales_features
        
        # Créer nouveau modèle avec politique adaptée
        self.adapted_model = PPO(
            SalesPolicy,
            self.sales_env,
            learning_rate=self.base_model.learning_rate,
            n_steps=self.base_model.n_steps,
            batch_size=self.base_model.batch_size,
            n_epochs=self.base_model.n_epochs,
            gamma=self.base_model.gamma,
            gae_lambda=self.base_model.gae_lambda,
            clip_range=self.base_model.clip_range,
            verbose=1
        )
        
        # Transférer les poids quand possible
        self._transfer_weights(original_state_dict)
        
        return self.adapted_model
    
    def _transfer_weights(self, original_state_dict: Dict):
        """Transférer les poids du modèle original"""
        
        new_state_dict = self.adapted_model.policy.state_dict()
        
        for key in original_state_dict:
            if key in new_state_dict:
                # Copier les poids si dimensions compatibles
                if original_state_dict[key].shape == new_state_dict[key].shape:
                    new_state_dict[key] = original_state_dict[key]
                else:
                    # Adapter les poids pour nouvelles dimensions
                    self._adapt_weights(key, original_state_dict[key], new_state_dict[key])
        
        self.adapted_model.policy.load_state_dict(new_state_dict)
    
    def _adapt_weights(self, key: str, original_weights, new_weights):
        """Adapter les poids pour nouvelles dimensions"""
        
        if 'weight' in key:
            # Pour les couches linéaires
            if len(original_weights.shape) == 2:
                orig_rows, orig_cols = original_weights.shape
                new_rows, new_cols = new_weights.shape
                
                # Copier autant que possible
                min_rows = min(orig_rows, new_rows)
                min_cols = min(orig_cols, new_cols)
                
                new_weights[:min_rows, :min_cols] = original_weights[:min_rows, :min_cols]
                
                # Initialiser le reste
                if new_rows > orig_rows or new_cols > orig_cols:
                    nn.init.xavier_uniform_(new_weights[min_rows:, min_cols:])
        
        elif 'bias' in key:
            # Pour les biais
            if len(original_weights.shape) == 1:
                min_len = min(len(original_weights), len(new_weights))
                new_weights[:min_len] = original_weights[:min_len]
    
    def fine_tune_on_sales_data(self, sales_data: Dict, epochs: int = 10):
        """Fine-tuning sur données de ventes spécifiques"""
        
        if not self.adapted_model:
            self.adapt_model_architecture()
        
        # Préparer données d'entraînement
        train_states, train_actions, train_rewards = self._prepare_training_data(sales_data)
        
        # Fine-tuning
        for epoch in range(epochs):
            epoch_loss = self._training_epoch(train_states, train_actions, train_rewards)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return self.adapted_model
    
    def _prepare_training_data(self, sales_data: Dict):
        """Préparer données pour fine-tuning"""
        
        states = []
        actions = []
        rewards = []
        
        for sale in sales_data.get('transactions', []):
            # Convertir en format RL
            state = self._sale_to_state(sale)
            action = self._sale_to_action(sale)
            reward = self._calculate_sale_reward(sale)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        
        return np.array(states), np.array(actions), np.array(rewards)
    
    def _sale_to_state(self, sale: Dict) -> np.ndarray:
        """Convertir vente en état RL"""
        state = np.zeros(10, dtype=np.float32)
        
        # Features de vente
        state[0] = sale.get('product_price', 0) / 1000.0
        state[1] = sale.get('quantity', 1) / 10.0
        state[2] = sale.get('customer_loyalty', 0.5)
        state[3] = 1.0 if sale.get('is_weekend', False) else -1.0
        state[4] = sale.get('competitor_price_ratio', 1.0) - 1.0
        state[5] = sale.get('stock_level', 0.5)
        state[6] = np.sin(2 * np.pi * sale.get('day_of_year', 0) / 365)
        state[7] = sale.get('promotion_active', 0)
        state[8] = sale.get('basket_size', 1) / 5.0
        state[9] = sale.get('conversion_rate', 0.5)
        
        return state
    
    def _sale_to_action(self, sale: Dict) -> int:
        """Déterminer action basée sur la vente"""
        price_change = sale.get('price_change_percent', 0)
        
        if price_change < -7.5:
            return 0  # -10%
        elif price_change < -2.5:
            return 1  # -5%
        elif price_change < 2.5:
            return 2  # 0%
        elif price_change < 7.5:
            return 3  # +5%
        else:
            return 4  # +10%
    
    def _calculate_sale_reward(self, sale: Dict) -> float:
        """Calculer récompense pour la vente"""
        profit = sale.get('profit', 0)
        quantity = sale.get('quantity', 1)
        customer_satisfaction = sale.get('customer_rating', 0.5)
        
        reward = profit / 100.0
        reward += quantity * 0.1
        reward += (customer_satisfaction - 0.5) * 2.0
        
        return float(reward)
    
    def _training_epoch(self, states, actions, rewards):
        """Exécuter une epoch de training"""
        
        # Convertir en tensors
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        
        # Forward pass
        action_logits, values = self.adapted_model.policy(states_t)
        
        # Calculer loss
        action_loss = nn.CrossEntropyLoss()(action_logits, actions_t)
        value_loss = nn.MSELoss()(values.squeeze(), rewards_t)
        
        total_loss = action_loss + 0.5 * value_loss
        
        # Backward pass
        self.adapted_model.policy.optimizer.zero_grad()
        total_loss.backward()
        self.adapted_model.policy.optimizer.step()
        
        return total_loss.item()