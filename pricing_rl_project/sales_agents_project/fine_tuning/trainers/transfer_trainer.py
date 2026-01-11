"""
Transfer learning entre produits/catégories
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TransferTrainer:
    """Trainer pour transfer learning"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.source_models = {}
        self.transfer_history = []
    
    def transfer_from_source(self, 
                           source_product_id: str, 
                           target_product_id: str,
                           transfer_method: str = 'feature_extraction') -> Dict:
        """Transférer apprentissage d'un produit source à cible"""
        
        logger.info(f"🔄 Transfert: {source_product_id} → {target_product_id}")
        
        # Charger modèle source
        source_model = self._load_source_model(source_product_id)
        if not source_model:
            return {'success': False, 'error': 'Source model not found'}
        
        # Analyser similarité produits
        similarity = self._analyze_product_similarity(source_product_id, target_product_id)
        
        # Appliquer transfer selon méthode
        if transfer_method == 'feature_extraction':
            transferred_model = self._transfer_feature_extraction(
                source_model, target_product_id, similarity
            )
        elif transfer_method == 'fine_tuning':
            transferred_model = self._transfer_fine_tuning(
                source_model, target_product_id, similarity
            )
        else:
            transferred_model = self._transfer_full_model(source_model, target_product_id)
        
        # Évaluer transfert
        transfer_score = self._evaluate_transfer(source_product_id, target_product_id, transferred_model)
        
        # Sauvegarder
        save_path = f"./pretrained_models/custom_trained/transfer_{source_product_id}_to_{target_product_id}"
        transferred_model.save(save_path)
        
        result = {
            'success': True,
            'source_product': source_product_id,
            'target_product': target_product_id,
            'transfer_method': transfer_method,
            'similarity_score': similarity,
            'transfer_score': transfer_score,
            'model_path': save_path
        }
        
        self.transfer_history.append(result)
        self._save_transfer_record(result)
        
        logger.info(f"✅ Transfert réussi: score={transfer_score:.3f}")
        
        return result
    
    def _load_source_model(self, product_id: str):
        """Charger modèle source"""
        # Chercher modèle dans custom_trained
        import os
        model_files = []
        
        custom_dir = "./pretrained_models/custom_trained/"
        if os.path.exists(custom_dir):
            for file in os.listdir(custom_dir):
                if product_id in file and file.endswith('.zip'):
                    model_files.append(os.path.join(custom_dir, file))
        
        if model_files:
            # Prendre le plus récent
            from stable_baselines3 import PPO
            latest = max(model_files, key=os.path.getmtime)
            return PPO.load(latest)
        
        return None
    
    def _analyze_product_similarity(self, product_id_1: str, product_id_2: str) -> float:
        """Analyser similarité entre produits"""
        
        # Récupérer produits depuis DB
        product1 = self.db.get_product(product_id_1)
        product2 = self.db.get_product(product_id_2)
        
        if not product1 or not product2:
            return 0.0
        
        # Calculer similarité
        similarities = []
        
        # Catégorie
        if product1.get('category') == product2.get('category'):
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # Fourchette prix
        price1 = product1.get('current_price', 0)
        price2 = product2.get('current_price', 0)
        price_sim = 1.0 - abs(price1 - price2) / max(price1, price2, 1)
        similarities.append(max(0, price_sim))
        
        # Marge
        cost1 = product1.get('cost_price', 0)
        cost2 = product2.get('cost_price', 0)
        margin1 = (price1 - cost1) / max(cost1, 1)
        margin2 = (price2 - cost2) / max(cost2, 1)
        margin_sim = 1.0 - abs(margin1 - margin2)
        similarities.append(max(0, margin_sim))
        
        return np.mean(similarities)
    
    def _transfer_feature_extraction(self, source_model, target_product_id: str, similarity: float):
        """Transfert par extraction de features"""
        
        from stable_baselines3 import PPO
        from environments.pricing_env import PricingEnvironment
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Créer environnement cible
        env = DummyVecEnv([lambda: PricingEnvironment()])
        
        # Créer nouveau modèle
        transferred_model = PPO(
            'MlpPolicy',
            env,
            learning_rate=0.0001 * similarity,  # LR adapté à similarité
            n_steps=2048,
            batch_size=64,
            n_epochs=5,  # Moins d'epochs pour fine-tuning
            gamma=0.99,
            verbose=1
        )
        
        # Transférer les poids des couches feature extraction
        self._transfer_feature_weights(source_model, transferred_model)
        
        return transferred_model
    
    def _transfer_feature_weights(self, source_model, target_model):
        """Transférer les poids de feature extraction"""
        
        # Copier les poids des premières couches
        source_params = dict(source_model.policy.named_parameters())
        target_params = dict(target_model.policy.named_parameters())
        
        for name, param in target_params.items():
            if name in source_params and 'features_extractor' in name:
                # Vérifier compatibilité dimensions
                if param.shape == source_params[name].shape:
                    param.data.copy_(source_params[name].data)
                    param.requires_grad = False  # Geler les poids transférés
        
        # Réinitialiser les autres poids
        for name, param in target_params.items():
            if name not in source_params or 'features_extractor' not in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def _transfer_fine_tuning(self, source_model, target_product_id: str, similarity: float):
        """Transfert par fine-tuning complet"""
        
        from stable_baselines3 import PPO
        
        # Créer copie du modèle source
        transferred_model = PPO(
            policy=source_model.policy,
            env=source_model.env,
            learning_rate=0.00005,  # LR très bas pour fine-tuning
            n_steps=1024,
            batch_size=32,
            n_epochs=3,
            verbose=1
        )
        
        return transferred_model
    
    def _transfer_full_model(self, source_model, target_product_id: str):
        """Transférer modèle complet"""
        return source_model
    
    def _evaluate_transfer(self, source_id: str, target_id: str, transferred_model) -> float:
        """Évaluer qualité du transfert"""
        # Métrique simplifiée
        return 0.8  # Placeholder
    
    def _save_transfer_record(self, transfer_record: Dict):
        """Sauvegarder record de transfert"""
        
        query = """
            INSERT INTO system_monitoring 
            (component, log_level, message, metrics, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        metrics = {
            'transfer_method': transfer_record['transfer_method'],
            'similarity_score': transfer_record['similarity_score'],
            'transfer_score': transfer_record['transfer_score'],
            'source_product': transfer_record['source_product'],
            'target_product': transfer_record['target_product']
        }
        
        self.db._execute_query(query, (
            'transfer_learning',
            'INFO',
            f"Transfer from {transfer_record['source_product']} to {transfer_record['target_product']}",
            str(metrics),
            datetime.now()
        ), fetch=False)