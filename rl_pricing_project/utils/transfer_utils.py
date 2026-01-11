#!/usr/bin/env python3
"""
Utilitaires pour le transfer learning et fine-tuning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import warnings

def freeze_model_layers(model: nn.Module, 
                       layers_to_freeze: List[str] = None,
                       freeze_percentage: float = 0.3) -> nn.Module:
    """
    Gèle les couches d'un modèle PyTorch
    
    Args:
        model: Modèle PyTorch
        layers_to_freeze: Noms spécifiques des couches à geler
        freeze_percentage: Pourcentage de couches à geler (si layers_to_freeze=None)
    
    Returns:
        Modèle avec couches gelées
    """
    if layers_to_freeze is None:
        # Geler un pourcentage des premières couches
        all_layers = list(model.named_parameters())
        num_to_freeze = int(len(all_layers) * freeze_percentage)
        
        layers_to_freeze = []
        for i, (name, param) in enumerate(all_layers):
            if i < num_to_freeze:
                layers_to_freeze.append(name)
    
    print(f"❄️  Gel de {len(layers_to_freeze)} couches:")
    
    for name, param in model.named_parameters():
        if any(freeze_name in name for freeze_name in layers_to_freeze):
            param.requires_grad = False
            print(f"   ✓ {name} (gelé)")
        else:
            param.requires_grad = True
    
    return model

def unfreeze_model_layers(model: nn.Module, 
                         layers_to_unfreeze: List[str] = None) -> nn.Module:
    """
    Dégèle les couches d'un modèle PyTorch
    
    Args:
        model: Modèle PyTorch
        layers_to_unfreeze: Noms spécifiques des couches à dégeler
    
    Returns:
        Modèle avec couches dégelées
    """
    if layers_to_unfreeze is None:
        # Dégeler toutes les couches
        layers_to_unfreeze = [name for name, _ in model.named_parameters()]
    
    print(f"🔓 Dégel de {len(layers_to_unfreeze)} couches:")
    
    for name, param in model.named_parameters():
        if any(unfreeze_name in name for unfreeze_name in layers_to_unfreeze):
            param.requires_grad = True
            print(f"   ✓ {name} (dégelé)")
    
    return model

def get_model_layer_info(model: nn.Module) -> Dict[str, Any]:
    """
    Obtient des informations détaillées sur les couches d'un modèle
    
    Returns:
        Dictionnaire avec informations des couches
    """
    layer_info = {
        'total_params': 0,
        'trainable_params': 0,
        'frozen_params': 0,
        'layers': [],
        'by_type': {}
    }
    
    for name, module in model.named_modules():
        if list(module.children()):  # Skip containers
            continue
        
        layer_type = module.__class__.__name__
        num_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen_params = num_params - trainable_params
        
        layer_data = {
            'name': name,
            'type': layer_type,
            'num_params': num_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'trainable_percent': (trainable_params / num_params * 100) if num_params > 0 else 0
        }
        
        layer_info['layers'].append(layer_data)
        layer_info['total_params'] += num_params
        layer_info['trainable_params'] += trainable_params
        layer_info['frozen_params'] += frozen_params
        
        # Compter par type
        if layer_type not in layer_info['by_type']:
            layer_info['by_type'][layer_type] = {
                'count': 0,
                'total_params': 0
            }
        
        layer_info['by_type'][layer_type]['count'] += 1
        layer_info['by_type'][layer_type]['total_params'] += num_params
    
    # Calculer les pourcentages
    if layer_info['total_params'] > 0:
        layer_info['trainable_percent'] = (layer_info['trainable_params'] / 
                                          layer_info['total_params'] * 100)
        layer_info['frozen_percent'] = (layer_info['frozen_params'] / 
                                       layer_info['total_params'] * 100)
    
    return layer_info

def print_layer_summary(model: nn.Module):
    """Affiche un résumé des couches du modèle"""
    info = get_model_layer_info(model)
    
    print("📊 Résumé des couches du modèle:")
    print("=" * 80)
    
    # Informations générales
    print(f"Paramètres totaux: {info['total_params']:,}")
    print(f"Paramètres entraînables: {info['trainable_params']:,} ({info.get('trainable_percent', 0):.1f}%)")
    print(f"Paramètres gelés: {info['frozen_params']:,} ({info.get('frozen_percent', 0):.1f}%)")
    
    # Par type de layer
    print(f"\n📈 Répartition par type:")
    for layer_type, type_info in info['by_type'].items():
        print(f"  {layer_type}: {type_info['count']} layers, "
              f"{type_info['total_params']:,} params")
    
    # Détail des couches
    print(f"\n🔍 Détail des couches (premières 10):")
    for i, layer in enumerate(info['layers'][:10]):
        status = "✓" if layer['trainable_percent'] > 50 else "❄️"
        print(f"  {status} {layer['name']} [{layer['type']}]: "
              f"{layer['trainable_params']:,}/{layer['num_params']:,} "
              f"({layer['trainable_percent']:.1f}%) entraînables")
    
    if len(info['layers']) > 10:
        print(f"  ... et {len(info['layers']) - 10} autres couches")

def adapt_observation_space(source_dim: int, 
                          target_dim: int, 
                          source_weights: torch.Tensor) -> torch.Tensor:
    """
    Adapte les poids d'une couche linéaire à une nouvelle dimension d'observation
    
    Args:
        source_dim: Dimension source
        target_dim: Dimension cible
        source_weights: Poids de la couche source [out_features, in_features]
    
    Returns:
        Poids adaptés [out_features, target_dim]
    """
    if source_dim == target_dim:
        return source_weights
    
    out_features = source_weights.shape[0]
    target_weights = torch.zeros(out_features, target_dim)
    
    # Copy matching dimensions
    min_dim = min(source_dim, target_dim)
    target_weights[:, :min_dim] = source_weights[:, :min_dim]
    
    # For additional dimensions, initialize with small random values
    if target_dim > source_dim:
        # Initialize new dimensions with scaled version of existing weights
        for i in range(source_dim, target_dim):
            # Use average of existing weights for new dimension
            target_weights[:, i] = source_weights.mean(dim=1) * 0.1
    
    print(f"🔧 Adaptation observation space: {source_dim} → {target_dim}")
    return target_weights

def adapt_action_space(source_dim: int,
                      target_dim: int,
                      source_weights: torch.Tensor) -> torch.Tensor:
    """
    Adapte les poids d'une couche linéaire à une nouvelle dimension d'action
    
    Args:
        source_dim: Dimension source (nombre d'actions)
        target_dim: Dimension cible (nombre d'actions)
        source_weights: Poids de la couche source [out_features, in_features]
    
    Returns:
        Poids adaptés [target_dim, in_features]
    """
    if source_dim == target_dim:
        return source_weights
    
    in_features = source_weights.shape[1]
    
    if target_dim > source_dim:
        # Plus d'actions cibles: étendre
        target_weights = torch.zeros(target_dim, in_features)
        target_weights[:source_dim, :] = source_weights
        
        # Initialiser les nouvelles actions avec une petite perturbation
        for i in range(source_dim, target_dim):
            # Mélanger les poids des actions existantes
            base_action = i % source_dim
            target_weights[i, :] = source_weights[base_action, :] * 0.5 + \
                                  torch.randn(in_features) * 0.01
    
    else:
        # Moins d'actions cibles: réduire
        target_weights = source_weights[:target_dim, :]
    
    print(f"🔧 Adaptation action space: {source_dim} → {target_dim} actions")
    return target_weights

def calculate_layer_similarity(layer1: nn.Module, 
                              layer2: nn.Module, 
                              metric: str = 'cosine') -> float:
    """
    Calcule la similarité entre deux couches
    
    Args:
        layer1, layer2: Couches à comparer
        metric: 'cosine', 'l2', ou 'correlation'
    
    Returns:
        Score de similarité (0-1, 1 = identique)
    """
    if not (isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear)):
        warnings.warn("Comparaison uniquement pour Linear layers")
        return 0.0
    
    w1 = layer1.weight.data.flatten()
    w2 = layer2.weight.data.flatten()
    
    if metric == 'cosine':
        similarity = torch.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0))
        return similarity.item()
    
    elif metric == 'l2':
        distance = torch.norm(w1 - w2, p=2)
        # Normaliser par la norme moyenne
        norm_mean = (torch.norm(w1, p=2) + torch.norm(w2, p=2)) / 2
        return 1.0 / (1.0 + distance.item() / norm_mean.item())
    
    elif metric == 'correlation':
        correlation = torch.corrcoef(torch.stack([w1, w2]))[0, 1]
        return (correlation.item() + 1) / 2  # Convertir de [-1,1] à [0,1]
    
    else:
        raise ValueError(f"Métrique non supportée: {metric}")

def progressive_unfreezing_schedule(total_steps: int, 
                                   num_layers: int) -> List[Tuple[int, int]]:
    """
    Génère un planning de dégel progressif
    
    Args:
        total_steps: Nombre total de steps d'entraînement
        num_layers: Nombre total de couches à dégeler
    
    Returns:
        Liste de (step, num_layers_to_unfreeze)
    """
    schedule = []
    
    # Exemple: dégeler une couche chaque 20% du training
    for i in range(num_layers):
        unfreeze_step = int(total_steps * (i + 1) / (num_layers + 1))
        schedule.append((unfreeze_step, i + 1))
    
    return schedule

def compute_feature_importance(model: nn.Module, 
                              observation: np.ndarray,
                              num_perturbations: int = 100) -> np.ndarray:
    """
    Calcule l'importance des features en perturbant l'input
    
    Args:
        model: Modèle à analyser
        observation: Observation d'entrée
        num_perturbations: Nombre de perturbations par feature
    
    Returns:
        Scores d'importance pour chaque feature
    """
    if not isinstance(observation, torch.Tensor):
        observation = torch.FloatTensor(observation)
    
    with torch.no_grad():
        # Prédiction de base
        base_output = model(observation.unsqueeze(0))
        
        # Perturber chaque feature
        feature_importance = []
        for i in range(observation.shape[0]):
            perturbations = []
            
            for _ in range(num_perturbations):
                # Perturber la feature i
                perturbed_obs = observation.clone()
                perturbed_obs[i] += torch.randn(1) * 0.1  # Bruit gaussien
                
                # Prédiction avec feature perturbée
                perturbed_output = model(perturbed_obs.unsqueeze(0))
                perturbations.append(perturbed_output)
            
            # Calculer la variance due à la perturbation
            perturbations = torch.stack(perturbations)
            variance = perturbations.var(dim=0).mean().item()
            feature_importance.append(variance)
    
    # Normaliser
    feature_importance = np.array(feature_importance)
    if feature_importance.sum() > 0:
        feature_importance = feature_importance / feature_importance.sum()
    
    return feature_importance

# Tests unitaires
if __name__ == "__main__":
    print("🧪 Tests des utilitaires de transfer learning")
    
    # Créer un modèle de test
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 5)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.layer2(x)
            return x
    
    model = TestModel()
    
    print("\n1. Informations des couches:")
    print_layer_summary(model)
    
    print("\n2. Gel des couches:")
    freeze_model_layers(model, freeze_percentage=0.5)
    print_layer_summary(model)
    
    print("\n3. Dégel des couches:")
    unfreeze_model_layers(model)
    print_layer_summary(model)