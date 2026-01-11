#!/usr/bin/env python3
"""
Téléchargeur de modèles pré-entraînés depuis SB3 Zoo (HuggingFace Hub)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import warnings

# Essayer d'importer huggingface_hub
try:
    from huggingface_hub import hf_hub_download, snapshot_download, list_models
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("huggingface_hub non installé. Installation: pip install huggingface-hub")

class PretrainedModelDownloader:
    """
    Gère le téléchargement et la gestion des modèles pré-entraînés SB3 Zoo
    """
    
    # Répertoire par défaut pour les modèles téléchargés
    DEFAULT_CACHE_DIR = "pretrained_models/checkpoints"
    
    # Mapping des modèles SB3 Zoo disponibles
    # Format: {model_id: {"algo": "ppo", "env": "CartPole-v1", "description": "..."}}
    AVAILABLE_MODELS = {
        # Modèles PPO (les plus stables pour fine-tuning)
        "ppo_cartpole": {
            "repo_id": "sb3/ppo-CartPole-v1",
            "filename": "ppo-CartPole-v1.zip",
            "algo": "ppo",
            "env": "CartPole-v1",
            "description": "PPO sur CartPole-v1 (4 observations, 2 actions)",
            "state_dim": 4,
            "action_dim": 2
        },
        "ppo_lunarlander": {
            "repo_id": "sb3/ppo-LunarLander-v2",
            "filename": "ppo-LunarLander-v2.zip",
            "algo": "ppo",
            "env": "LunarLander-v2",
            "description": "PPO sur LunarLander-v2 (8 observations, 4 actions)",
            "state_dim": 8,
            "action_dim": 4
        },
        "ppo_mountaincar": {
            "repo_id": "sb3/ppo-MountainCar-v0",
            "filename": "ppo-MountainCar-v0.zip",
            "algo": "ppo",
            "env": "MountainCar-v0",
            "description": "PPO sur MountainCar-v0 (2 observations, 3 actions)",
            "state_dim": 2,
            "action_dim": 3
        },
        
        # Modèles DQN
        "dqn_cartpole": {
            "repo_id": "sb3/dqn-CartPole-v1",
            "filename": "dqn-CartPole-v1.zip",
            "algo": "dqn",
            "env": "CartPole-v1",
            "description": "DQN sur CartPole-v1",
            "state_dim": 4,
            "action_dim": 2
        },
        "dqn_lunarlander": {
            "repo_id": "sb3/dqn-LunarLander-v2",
            "filename": "dqn-LunarLander-v2.zip",
            "algo": "dqn",
            "env": "LunarLander-v2",
            "description": "DQN sur LunarLander-v2",
            "state_dim": 8,
            "action_dim": 4
        },
        
        # Modèles A2C
        "a2c_cartpole": {
            "repo_id": "sb3/a2c-CartPole-v1",
            "filename": "a2c-CartPole-v1.zip",
            "algo": "a2c",
            "env": "CartPole-v1",
            "description": "A2C sur CartPole-v1",
            "state_dim": 4,
            "action_dim": 2
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialise le téléchargeur
        
        Args:
            cache_dir: Répertoire où stocker les modèles téléchargés
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(self.DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not HF_AVAILABLE:
            print("⚠️  huggingface_hub non disponible. Installation:")
            print("   pip install huggingface-hub")
            print("   ou ajoutez à requirements.txt: huggingface-hub>=0.16.0")
    
    def list_available_models(self) -> Dict[str, Dict]:
        """
        Liste tous les modèles disponibles
        
        Returns:
            Dictionnaire des modèles disponibles
        """
        print("📋 Modèles pré-entraînés disponibles:")
        print("-" * 80)
        
        for model_id, info in self.AVAILABLE_MODELS.items():
            print(f"🔹 {model_id}:")
            print(f"   Algorithme: {info['algo'].upper()}")
            print(f"   Environnement: {info['env']}")
            print(f"   Description: {info['description']}")
            print(f"   Dimensions: état={info['state_dim']}, actions={info['action_dim']}")
            print()
        
        return self.AVAILABLE_MODELS
    
    def download_model(self, model_id: str, force_download: bool = False) -> Optional[Path]:
        """
        Télécharge un modèle spécifique
        
        Args:
            model_id: Identifiant du modèle (ex: "ppo_cartpole")
            force_download: Force le re-téléchargement même s'il existe déjà
            
        Returns:
            Chemin vers le modèle téléchargé, ou None en cas d'erreur
        """
        if not HF_AVAILABLE:
            print("❌ huggingface_hub non disponible. Installation requise.")
            return None
        
        if model_id not in self.AVAILABLE_MODELS:
            print(f"❌ Modèle '{model_id}' non reconnu.")
            print("   Utilisez --list pour voir les modèles disponibles")
            return None
        
        model_info = self.AVAILABLE_MODELS[model_id]
        repo_id = model_info["repo_id"]
        filename = model_info["filename"]
        
        # Chemin de destination
        dest_path = self.cache_dir / filename
        
        # Vérifier si le modèle existe déjà
        if dest_path.exists() and not force_download:
            print(f"✅ Modèle déjà téléchargé: {dest_path}")
            print(f"   Pour re-télécharger, utilisez --force")
            return dest_path
        
        print(f"📥 Téléchargement de {model_id}...")
        print(f"   Repository: {repo_id}")
        print(f"   Fichier: {filename}")
        print(f"   Destination: {dest_path}")
        
        try:
            # Télécharger depuis HuggingFace Hub
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                force_download=force_download,
                resume_download=True
            )
            
            # S'assurer que le fichier est au bon endroit
            downloaded_path = Path(downloaded_path)
            if downloaded_path != dest_path:
                # Créer un lien symbolique ou copier
                import shutil
                shutil.copy2(downloaded_path, dest_path)
            
            print(f"✅ Téléchargement réussi!")
            print(f"   Taille: {dest_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"   Chemin: {dest_path}")
            
            # Vérifier que le modèle est valide
            if self._validate_model(dest_path):
                print("   ✓ Modèle valide et prêt à l'emploi")
            else:
                print("   ⚠️  Modèle téléchargé mais validation échouée")
            
            return dest_path
            
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            return None
    
    def _validate_model(self, model_path: Path) -> bool:
        """
        Valide qu'un modèle téléchargé est utilisable
        
        Args:
            model_path: Chemin vers le modèle
            
        Returns:
            True si le modèle est valide
        """
        try:
            # Essayer de charger le modèle pour vérifier qu'il n'est pas corrompu
            from stable_baselines3 import PPO, DQN, A2C
            
            # Détecter l'algorithme à partir du nom de fichier
            model_name = model_path.name.lower()
            
            if 'ppo' in model_name:
                model_class = PPO
            elif 'dqn' in model_name:
                model_class = DQN
            elif 'a2c' in model_name:
                model_class = A2C
            else:
                print(f"⚠️  Impossible de détecter l'algorithme pour {model_path.name}")
                return False
            
            # Créer un environnement dummy pour le chargement
            import gymnasium as gym
            dummy_env = gym.make('CartPole-v1')
            
            # Essayer de charger (sans vraiment l'utiliser)
            model = model_class.load(str(model_path), env=dummy_env)
            
            # Vérifications de base
            if hasattr(model, 'policy'):
                print(f"   Architecture: {model.policy.__class__.__name__}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"   Erreur validation: {e}")
            return False
    
    def download_all_models(self, force_download: bool = False) -> Dict[str, Path]:
        """
        Télécharge tous les modèles disponibles
        
        Args:
            force_download: Force le re-téléchargement
            
        Returns:
            Dictionnaire {model_id: chemin}
        """
        print("📥 Téléchargement de TOUS les modèles disponibles...")
        
        downloaded = {}
        for model_id in self.AVAILABLE_MODELS:
            print(f"\n{'='*60}")
            print(f"Téléchargement: {model_id}")
            print(f"{'='*60}")
            
            path = self.download_model(model_id, force_download)
            if path:
                downloaded[model_id] = path
        
        print(f"\n✅ Téléchargement terminé: {len(downloaded)}/{len(self.AVAILABLE_MODELS)} modèles")
        
        return downloaded
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """
        Obtient les informations d'un modèle spécifique
        
        Args:
            model_id: Identifiant du modèle
            
        Returns:
            Informations du modèle ou None si non trouvé
        """
        if model_id in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_id]
        return None
    
    def search_models_by_algo(self, algorithm: str) -> List[str]:
        """
        Recherche les modèles par algorithme
        
        Args:
            algorithm: 'ppo', 'dqn', 'a2c'
            
        Returns:
            Liste des IDs de modèle correspondants
        """
        algorithm = algorithm.lower()
        return [model_id for model_id, info in self.AVAILABLE_MODELS.items() 
                if info['algo'] == algorithm]
    
    def cleanup_cache(self, keep_last_n: int = 3):
        """
        Nettoie le cache en gardant seulement les N derniers modèles
        
        Args:
            keep_last_n: Nombre de modèles à conserver
        """
        import glob
        import time
        
        model_files = list(self.cache_dir.glob("*.zip"))
        
        if len(model_files) <= keep_last_n:
            print(f"✅ Cache propre: {len(model_files)} fichiers (limite: {keep_last_n})")
            return
        
        # Trier par date de modification (plus récent d'abord)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Garder les N premiers, supprimer les autres
        to_keep = model_files[:keep_last_n]
        to_delete = model_files[keep_last_n:]
        
        print(f"🗑️  Nettoyage du cache: {len(to_delete)} fichiers à supprimer")
        
        for file_path in to_delete:
            try:
                file_path.unlink()
                print(f"   Supprimé: {file_path.name}")
            except Exception as e:
                print(f"   Erreur suppression {file_path}: {e}")
        
        print(f"✅ Cache nettoyé: {len(to_keep)} fichiers conservés")


def main():
    """Fonction principale pour le téléchargement en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Téléchargeur de modèles pré-entraînés SB3 Zoo"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="Liste tous les modèles disponibles"
    )
    
    parser.add_argument(
        "--download", 
        type=str,
        help="Télécharge un modèle spécifique (ex: ppo_cartpole)"
    )
    
    parser.add_argument(
        "--download-all", 
        action="store_true",
        help="Télécharge tous les modèles disponibles"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force le re-téléchargement même si le modèle existe déjà"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=str,
        default="pretrained_models/checkpoints",
        help="Répertoire de cache pour les modèles téléchargés"
    )
    
    parser.add_argument(
        "--info", 
        type=str,
        help="Affiche les informations d'un modèle spécifique"
    )
    
    parser.add_argument(
        "--search", 
        type=str,
        help="Recherche les modèles par algorithme (ppo, dqn, a2c)"
    )
    
    parser.add_argument(
        "--cleanup", 
        action="store_true",
        help="Nettoie le cache en gardant seulement les 3 derniers modèles"
    )
    
    args = parser.parse_args()
    
    # Initialiser le téléchargeur
    downloader = PretrainedModelDownloader(cache_dir=args.cache_dir)
    
    # Exécuter l'action demandée
    if args.list:
        downloader.list_available_models()
    
    elif args.info:
        info = downloader.get_model_info(args.info)
        if info:
            print(f"📊 Informations pour {args.info}:")
            for key, value in info.items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ Modèle '{args.info}' non trouvé")
    
    elif args.search:
        models = downloader.search_models_by_algo(args.search)
        if models:
            print(f"🔍 Modèles {args.search.upper()} disponibles:")
            for model_id in models:
                print(f"   - {model_id}")
        else:
            print(f"❌ Aucun modèle trouvé pour l'algorithme '{args.search}'")
    
    elif args.download_all:
        downloader.download_all_models(force_download=args.force)
    
    elif args.download:
        downloader.download_model(args.download, force_download=args.force)
    
    elif args.cleanup:
        downloader.cleanup_cache()
    
    else:
        # Mode interactif par défaut
        print("🤖 Téléchargeur de modèles pré-entraînés SB3 Zoo")
        print("=" * 60)
        print("Utilisation:")
        print("  --list              : Liste les modèles disponibles")
        print("  --download MODEL    : Télécharge un modèle spécifique")
        print("  --info MODEL        : Affiche les infos d'un modèle")
        print("  --search ALGO       : Recherche par algorithme")
        print("\nExemples:")
        print("  python -m pretrained_models.downloader --list")
        print("  python -m pretrained_models.downloader --download ppo_cartpole")
        print("  python -m pretrained_models.downloader --download-all")


if __name__ == "__main__":
    main()