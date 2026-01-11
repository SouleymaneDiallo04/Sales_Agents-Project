#!/usr/bin/env python3
"""
Script principal pour le fine-tuning de modèles pré-entraînés
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import yaml
from models.finetune_trainer import FineTuneTrainer

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Fine-tuning de modèles pré-entraînés SB3 Zoo pour le pricing"
    )
    
    parser.add_argument(
        "--pretrained", 
        type=str,
        required=True,
        help="Chemin vers le modèle pré-entraîné (ex: pretrained_models/checkpoints/ppo-CartPole-v1.zip)"
    )
    
    parser.add_argument(
        "--timesteps", 
        type=int,
        default=50000,
        help="Nombre de timesteps pour le fine-tuning (défaut: 50000)"
    )
    
    parser.add_argument(
        "--product", 
        type=str,
        default="PROD_001",
        help="ID du produit (défaut: PROD_001)"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        default=None,
        help="Chemin vers fichier de configuration YAML"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        default=None,
        help="Chemin de sortie pour le modèle fine-tuné"
    )
    
    parser.add_argument(
        "--eval-freq", 
        type=int,
        default=5000,
        help="Fréquence d'évaluation (défaut: 5000)"
    )
    
    parser.add_argument(
        "--n-envs", 
        type=int,
        default=4,
        help="Nombre d'environnements parallèles (défaut: 4)"
    )
    
    parser.add_argument(
        "--compare-with", 
        type=str,
        default=None,
        help="Compare avec un modèle baseline (from-scratch)"
    )
    
    parser.add_argument(
        "--no-freeze", 
        action="store_true",
        help="Ne pas geler les couches pendant le fine-tuning"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Forcer le fine-tuning même si le modèle existe déjà"
    )
    
    return parser.parse_args()

def validate_model_path(model_path: str) -> bool:
    """Valide que le modèle existe et est valide"""
    path = Path(model_path)
    
    if not path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return False
    
    if not path.suffix == '.zip':
        print(f"⚠️  Extension suspecte: {path.suffix}. Doit être .zip")
    
    # Vérifier la taille minimale
    if path.stat().st_size < 1024:  # 1KB minimum
        print(f"⚠️  Fichier trop petit: {path.stat().st_size} bytes")
        return False
    
    return True

def generate_output_path(pretrained_path: str, custom_output: str = None) -> str:
    """Génère un chemin de sortie approprié"""
    if custom_output:
        return custom_output
    
    # Générer automatiquement basé sur le modèle source
    source_path = Path(pretrained_path)
    source_name = source_path.stem  # "ppo-CartPole-v1"
    
    # Nettoyer le nom
    clean_name = source_name.replace('-', '_').replace('.', '_')
    
    return f"models/finetuned_{clean_name}_pricing"

def main():
    """Fonction principale de fine-tuning"""
    args = parse_args()
    
    print("🚀 Fine-tuning de modèle pré-entraîné")
    print("=" * 60)
    
    # 1. Validation du modèle source
    print(f"📂 Modèle source: {args.pretrained}")
    if not validate_model_path(args.pretrained):
        print("❌ Modèle source invalide. Arrêt.")
        return 1
    
    # 2. Générer le chemin de sortie
    output_path = generate_output_path(args.pretrained, args.output)
    
    # Vérifier si le modèle existe déjà
    if Path(f"{output_path}_final.zip").exists() and not args.force:
        print(f"⚠️  Modèle fine-tuné existe déjà: {output_path}_final.zip")
        print("   Utilisez --force pour re-fine-tuner")
        response = input("   Continuer quand même? (y/n): ")
        if response.lower() != 'y':
            return 0
    
    # 3. Charger la configuration personnalisée si fournie
    config_dict = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
            print(f"✅ Configuration chargée: {args.config}")
        except Exception as e:
            print(f"⚠️  Erreur chargement config: {e}")
            print("   Utilisation de la configuration par défaut")
    
    # 4. Ajuster la configuration selon les arguments
    if config_dict is None:
        config_dict = {}
    
    # Override avec les arguments CLI
    if args.n_envs != 4:
        config_dict.setdefault('training', {})['n_envs'] = args.n_envs
    
    if args.no_freeze:
        config_dict.setdefault('adaptation', {})['freeze_layers'] = 0
    
    # 5. Initialiser le trainer
    print(f"\n⚙️  Initialisation du FineTuneTrainer...")
    print(f"   Produit: {args.product}")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Sortie: {output_path}")
    
    try:
        trainer = FineTuneTrainer(
            pretrained_model_path=args.pretrained,
            product_id=args.product,
            config_path=args.config if args.config else None
        )
        
        # 6. Fine-tuning
        print(f"\n🎯 Début du fine-tuning...")
        model = trainer.finetune(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            save_path=output_path
        )
        
        # 7. Évaluation
        print(f"\n📊 Évaluation du modèle fine-tuné...")
        rewards, metrics = trainer.evaluate_finetuned_model(num_episodes=5)
        
        # 8. Comparaison avec baseline si demandé
        if args.compare_with:
            print(f"\n📈 Comparaison avec baseline...")
            if validate_model_path(args.compare_with):
                comparison = trainer.compare_with_baseline(
                    baseline_model_path=args.compare_with,
                    num_episodes=10
                )
                
                # Sauvegarder les résultats de comparaison
                results_file = f"{output_path}_comparison.yaml"
                with open(results_file, 'w') as f:
                    yaml.dump(comparison, f, default_flow_style=False)
                print(f"💾 Résultats sauvegardés: {results_file}")
        
        # 9. Résumé final
        print(f"\n{'='*60}")
        print(f"✨ FINE-TUNING TERMINÉ AVEC SUCCÈS!")
        print(f"{'='*60}")
        print(f"📁 Modèle final: {output_path}_final.zip")
        print(f"📁 Meilleur modèle: {output_path}_best.zip")
        print(f"📁 Checkpoints: {output_path}_checkpoints/")
        print(f"📊 Performance: reward moyen = {metrics['mean_reward']:.2f}")
        print(f"\n💡 Pour utiliser ce modèle:")
        print(f"   python scripts/evaluate.py --model {output_path}_final.zip")
        print(f"   python scripts/deploy.py --model {output_path}_final.zip")
        
        return 0
        
    except Exception as e:
        print(f"❌ Erreur pendant le fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())