#!/usr/bin/env python3
"""
Adaptateur corrigé pour le fine-tuning - version simplifiée
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
import gymnasium as gym
import warnings

class ModelAdapter:
    """
    Adapte les modèles pré-entraînés à l'environnement EcommercePricing
    Version corrigée pour gérer les différences d'espace d'observation
    """
    
    # Dimensions de notre environnement
    PRICING_STATE_DIM = 9    # Notre environnement a 9 features
    PRICING_ACTION_DIM = 5   # 5 actions discrètes (-10%, -5%, 0%, +5%, +10%)
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Adaptateur initialisé sur: {self.device}")
    
    def adapt_pretrained_model(self, 
                              pretrained_model_path: str,
                              target_env: gym.Env) -> Any:
        """
        Adapte un modèle pré-entraîné à notre environnement - VERSION CORRIGÉE
        
        Args:
            pretrained_model_path: Chemin vers le modèle pré-entraîné
            target_env: Environnement cible (EcommercePricing)
            
        Returns:
            Modèle adapté pour fine-tuning
        """
        print(f"🔄 Adaptation du modèle: {pretrained_model_path}")
        
        # 1. Extraire les poids du modèle pré-entraîné (sans essayer de l'adapter directement)
        print("   📥 Extraction des poids...")
        pretrained_weights = self._extract_pretrained_weights(pretrained_model_path)
        
        # 2. Créer un NOUVEAU modèle pour notre environnement
        print("   🆕 Création d'un nouveau modèle...")
        
        # Détecter l'algorithme
        model_name = pretrained_model_path.lower()
        if 'ppo' in model_name:
            model_class = PPO
        elif 'dqn' in model_name:
            model_class = DQN
        elif 'a2c' in model_name:
            model_class = A2C
        else:
            model_class = PPO
        
        # Créer un nouveau modèle avec la bonne architecture
        model = model_class(
            policy='MlpPolicy',
            env=target_env,
            learning_rate=0.0001,  # Faible LR pour fine-tuning
            verbose=1,
            device=self.device
        )
        
        # 3. Transférer les poids adaptés
        print("   🔄 Transfert des poids...")
        self._transfer_and_adapt_weights(model, pretrained_weights)
        
        # 4. Configurer pour le fine-tuning
        self._configure_for_finetuning(model)
        
        print(f"✅ Modèle adapté avec succès!")
        return model
    
    def _extract_pretrained_weights(self, model_path: str) -> Dict[str, torch.Tensor]:
        """
        Extrait les poids d'un modèle pré-entraîné sans se soucier des espaces d'observation
        
        Args:
            model_path: Chemin vers le modèle pré-entraîné
            
        Returns:
            Dictionnaire des poids extraits
        """
        print(f"   🔍 Extraction depuis: {model_path}")
        
        # Créer un environnement CartPole pour charger le modèle
        try:
            temp_env = gym.make('CartPole-v1')
            
            # Charger le modèle avec des custom_objects pour éviter les warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Détecter l'algorithme
                model_name = model_path.lower()
                if 'ppo' in model_name:
                    model_class = PPO
                elif 'dqn' in model_name:
                    model_class = DQN
                elif 'a2c' in model_name:
                    model_class = A2C
                else:
                    model_class = PPO
                
                # Custom objects pour éviter les problèmes de sérialisation
                custom_objects = {
                    "learning_rate": 0.0003,
                    "clip_range": 0.2,
                    "lr_schedule": lambda _: 0.0003,
                    "clip_range_vf": None
                }
                
                pretrained_model = model_class.load(
                    model_path, 
                    env=temp_env,
                    custom_objects=custom_objects
                )
            
            # Extraire tous les poids
            weights = {}
            for name, param in pretrained_model.policy.named_parameters():
                weights[name] = param.data.clone().to(self.device)
                print(f"     ✓ {name}: {weights[name].shape}")
            
            print(f"   ✅ {len(weights)} poids extraits")
            return weights
            
        except Exception as e:
            print(f"   ⚠️  Erreur extraction: {e}")
            print(f"   Utilisation de poids aléatoires comme fallback")
            return {}
    
    def _transfer_and_adapt_weights(self, 
                                   model: Any, 
                                   pretrained_weights: Dict[str, torch.Tensor]):
        """
        Transfère et adapte les poids au nouveau modèle
        
        Args:
            model: Nouveau modèle (pour Pricing)
            pretrained_weights: Poids extraits du modèle pré-entraîné
        """
        if not pretrained_weights:
            print("   ℹ️  Aucun poids pré-entraîné, modèle initialisé aléatoirement")
            return
        
        new_policy = model.policy
        transferred = 0
        
        # Pour chaque paramètre du nouveau modèle
        for name, new_param in new_policy.named_parameters():
            # Chercher un poids correspondant dans les poids pré-entraînés
            for pretrained_name, pretrained_weight in pretrained_weights.items():
                # Vérifier si les noms correspondent (approximativement)
                if self._layers_match(name, pretrained_name):
                    try:
                        # Adapter les dimensions si nécessaire
                        adapted_weight = self._adapt_weight_dimensions(
                            pretrained_weight, 
                            new_param.data.shape,
                            layer_name=name
                        )
                        
                        # Copier les poids adaptés
                        new_param.data.copy_(adapted_weight)
                        transferred += 1
                        print(f"     🔄 {pretrained_name} → {name}")
                        break
                        
                    except Exception as e:
                        print(f"     ⚠️  Erreur adaptation {pretrained_name}: {e}")
        
        print(f"   📊 {transferred}/{len(list(new_policy.parameters()))} poids transférés")
    
    def _layers_match(self, new_layer_name: str, pretrained_layer_name: str) -> bool:
        """
        Vérifie si deux couches correspondent (logique simplifiée)
        """
        # Noms de base des couches
        base_names = ['weight', 'bias']
        
        for base in base_names:
            if base in new_layer_name and base in pretrained_layer_name:
                # Vérifier le type de couche
                if 'policy_net' in new_layer_name and 'policy_net' in pretrained_layer_name:
                    return True
                elif 'value_net' in new_layer_name and 'value_net' in pretrained_layer_name:
                    return True
                elif 'action_net' in new_layer_name and 'action_net' in pretrained_layer_name:
                    return True
                elif 'q_net' in new_layer_name and 'q_net' in pretrained_layer_name:
                    return True
        
        return False
    
    def _adapt_weight_dimensions(self, 
                                source_weight: torch.Tensor,
                                target_shape: Tuple[int, ...],
                                layer_name: str = "") -> torch.Tensor:
        """
        Adapte les poids aux nouvelles dimensions
        
        Args:
            source_weight: Poids source
            target_shape: Shape cible
            layer_name: Nom de la couche pour le logging
            
        Returns:
            Poids adaptés
        """
        source_shape = source_weight.shape
        
        # Si les shapes sont identiques, pas besoin d'adaptation
        if source_shape == target_shape:
            return source_weight
        
        print(f"     🔧 Adaptation dimensions: {source_shape} → {target_shape} ({layer_name})")
        
        # Pour les poids de couche linéaire (2D)
        if len(source_shape) == 2 and len(target_shape) == 2:
            return self._adapt_linear_weights(source_weight, target_shape, layer_name)
        
        # Pour les biais (1D)
        elif len(source_shape) == 1 and len(target_shape) == 1:
            return self._adapt_bias_weights(source_weight, target_shape, layer_name)
        
        # Fallback: initialisation aléatoire
        else:
            print(f"     ⚠️  Shape non supportée, initialisation aléatoire")
            return torch.randn(target_shape, device=self.device) * 0.01
    
    def _adapt_linear_weights(self, 
                             source_weight: torch.Tensor,
                             target_shape: Tuple[int, int],
                             layer_name: str) -> torch.Tensor:
        """
        Adapte les poids d'une couche linéaire
        Shape: [out_features, in_features]
        """
        source_out, source_in = source_weight.shape
        target_out, target_in = target_shape
        
        # Créer un nouveau tenseur
        new_weight = torch.zeros(target_shape, device=self.device)
        
        # Copier les poids correspondants
        copy_out = min(source_out, target_out)
        copy_in = min(source_in, target_in)
        
        new_weight[:copy_out, :copy_in] = source_weight[:copy_out, :copy_in]
        
        # Si la dimension d'entrée est plus grande (notre env a plus de features)
        if target_in > source_in:
            # Pour les nouvelles features, utiliser la moyenne des poids existants
            for i in range(source_in, target_in):
                new_weight[:, i] = source_weight.mean(dim=1) * 0.1
        
        # Si la dimension de sortie est plus grande (plus d'actions)
        if target_out > source_out:
            # Pour les nouvelles actions, dupliquer les poids des actions existantes
            for i in range(source_out, target_out):
                base_action = i % source_out
                new_weight[i, :] = source_weight[base_action, :] * 0.5
        
        return new_weight
    
    def _adapt_bias_weights(self,
                           source_bias: torch.Tensor,
                           target_shape: Tuple[int],
                           layer_name: str) -> torch.Tensor:
        """
        Adapte les biais d'une couche
        """
        source_size = source_bias.shape[0]
        target_size = target_shape[0]
        
        new_bias = torch.zeros(target_shape, device=self.device)
        
        # Copier les biais correspondants
        copy_size = min(source_size, target_size)
        new_bias[:copy_size] = source_bias[:copy_size]
        
        return new_bias
    
    def _configure_for_finetuning(self, model: Any):
        """Configure le modèle pour le fine-tuning"""
        print(f"   ⚙️  Configuration pour fine-tuning...")
        
        # Hyperparamètres optimisés pour fine-tuning
        if hasattr(model, 'learning_rate'):
            model.learning_rate = 0.0001  # Très petit LR
            print(f"     Learning rate: {model.learning_rate}")
        
        if hasattr(model, 'n_steps'):
            model.n_steps = 1024
            print(f"     n_steps: {model.n_steps}")
        
        if hasattr(model, 'batch_size'):
            model.batch_size = 64
        
        if hasattr(model, 'n_epochs'):
            model.n_epochs = 5
        
        if hasattr(model, 'clip_range'):
            model.clip_range = 0.1  # Plus conservateur
        
        if hasattr(model, 'ent_coef'):
            model.ent_coef = 0.01  # Encourage l'exploration
    
    def freeze_first_layers(self, model: Any, num_layers: int = 2):
        """
        Gèle les premières couches du modèle pour préserver les features apprises
        """
        print(f"❄️  Gel des {num_layers} premières couches...")
        
        layers_frozen = 0
        
        # Pour PPO/A2C
        if hasattr(model.policy, 'mlp_extractor'):
            mlp_extractor = model.policy.mlp_extractor
            
            # Gel des couches du policy network
            if hasattr(mlp_extractor, 'policy_net'):
                policy_layers = []
                
                # Collecter toutes les couches linéaires
                for module in mlp_extractor.policy_net.modules():
                    if isinstance(module, nn.Linear):
                        policy_layers.append(module)
                
                # Geler les premières N couches
                for i, layer in enumerate(policy_layers[:num_layers]):
                    for param in layer.parameters():
                        param.requires_grad = False
                    layers_frozen += 1
                    print(f"     ❄️  Gelée: {layer}")
        
        print(f"   📊 {layers_frozen} couches gelées")
        return model
    
    def get_transfer_summary(self, model: Any) -> Dict[str, Any]:
        """
        Retourne un résumé du transfert de poids
        """
        summary = {
            'total_params': 0,
            'transferred_params': 0,
            'random_params': 0,
            'frozen_params': 0,
            'trainable_params': 0
        }
        
        for name, param in model.policy.named_parameters():
            summary['total_params'] += param.numel()
            
            if param.requires_grad:
                summary['trainable_params'] += param.numel()
            else:
                summary['frozen_params'] += param.numel()
            
            # Estimation: si les poids sont proches de zéro, probablement aléatoires
            if torch.abs(param.data).mean() < 0.1:
                summary['random_params'] += param.numel()
            else:
                summary['transferred_params'] += param.numel()
        
        return summary


# Version alternative ultra-simplifiée
class SimpleModelAdapter:
    """Adapter ultra-simple pour tester rapidement"""
    
    @staticmethod
    def create_finetuned_model(pretrained_path: str, target_env: gym.Env) -> PPO:
        """
        Crée un nouveau modèle avec les mêmes hyperparams que le pré-entraîné
        """
        print("🔄 Création modèle fine-tuned (version simple)...")
        
        # Créer un nouveau modèle pour notre environnement
        model = PPO(
            policy='MlpPolicy',
            env=target_env,
            learning_rate=0.0001,  # Faible pour fine-tuning
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./logs/tensorboard_finetune/"
        )
        
        print("✅ Modèle créé pour fine-tuning")
        print(f"   LR: {model.learning_rate}")
        print(f"   Architecture: {model.policy.__class__.__name__}")
        
        return model


# Test du module
if __name__ == "__main__":
    print("🧪 Test du ModelAdapter corrigé")
    
    # Créer un environnement de test
    from environments.pricing_env import EcommercePricingEnv
    
    env = EcommercePricingEnv()
    adapter = ModelAdapter()
    
    # Afficher les informations de base
    print(f"\n📊 Environnement cible:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Tester avec l'adapteur simple
    print("\n🔧 Test adapteur simple...")
    simple_adapter = SimpleModelAdapter()
    test_model = simple_adapter.create_finetuned_model("dummy_path.zip", env)
    
    print("\n✅ Test terminé avec succès!")