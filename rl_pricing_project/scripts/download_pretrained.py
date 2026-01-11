#!/usr/bin/env python3
"""
Script wrapper pour télécharger des modèles pré-entraînés
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrained_models.downloader import PretrainedModelDownloader

def main():
    parser = argparse.ArgumentParser(
        description="Interface de téléchargement de modèles pré-entraînés"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        required=True,
        help="ID du modèle à télécharger (ex: ppo_cartpole, dqn_lunarlander)"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force le re-téléchargement"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=str,
        default="pretrained_models/checkpoints",
        help="Répertoire de cache"
    )
    
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="Liste tous les modèles disponibles"
    )
    
    args = parser.parse_args()
    
    downloader = PretrainedModelDownloader(cache_dir=args.cache_dir)
    
    if args.list_models:
        downloader.list_available_models()
        return
    
    print(f"🚀 Début du téléchargement du modèle: {args.model}")
    print("-" * 50)
    
    model_path = downloader.download_model(args.model, force_download=args.force)
    
    if model_path:
        print(f"\n✅ Téléchargement réussi!")
        print(f"📁 Modèle disponible à: {model_path}")
        print(f"\n💡 Pour utiliser ce modèle:")
        print(f"   python scripts/finetune.py --pretrained {model_path}")
    else:
        print(f"\n❌ Échec du téléchargement")
        print(f"   Vérifiez l'ID du modèle avec: python scripts/download_pretrained.py --list-models")

if __name__ == "__main__":
    main()