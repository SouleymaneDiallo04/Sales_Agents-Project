#!/usr/bin/env python3
"""
Script de lancement de l'entraînement des agents RL
Configure et exécute l'entraînement des modèles d'agents
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import yaml

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fine_tuning.trainers.sales_trainer import SalesTrainer
from config.rl_training import RLTrainingConfig

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Charger la configuration depuis un fichier YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Fonction principale pour l'entraînement"""

    parser = argparse.ArgumentParser(description="Entraînement des agents Sales RL")
    parser.add_argument(
        "--config",
        type=str,
        default="config/rl_training.yaml",
        help="Chemin vers le fichier de configuration"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["premium", "price_sensitive", "urgent", "all"],
        default="all",
        help="Agent à entraîner"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Nombre d'épisodes d'entraînement"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Sauvegarder le modèle entraîné"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Évaluer le modèle après l'entraînement"
    )

    args = parser.parse_args()

    # Charger la configuration
    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"Fichier de configuration non trouvé: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    logger.info(f"Configuration chargée depuis {config_path}")

    # Créer le répertoire de logs
    log_dir = project_root / "logs" / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialiser l'entraîneur
    trainer_config = RLTrainingConfig(**config)
    if args.episodes:
        trainer_config.total_timesteps = args.episodes

    trainer = SalesTrainer(trainer_config)

    try:
        # Liste des agents à entraîner
        agents_to_train = []
        if args.agent == "all":
            agents_to_train = ["premium", "price_sensitive", "urgent"]
        else:
            agents_to_train = [args.agent]

        # Entraîner chaque agent
        for agent_name in agents_to_train:
            logger.info(f"Début de l'entraînement pour l'agent: {agent_name}")

            # Entraîner l'agent
            model = trainer.train_agent(agent_name)

            if args.save_model:
                model_path = project_root / "pretrained_models" / "custom_trained" / f"{agent_name}_model.zip"
                model.save(str(model_path))
                logger.info(f"Modèle sauvegardé: {model_path}")

            if args.evaluate:
                # Évaluation basique
                logger.info(f"Évaluation de l'agent {agent_name}")
                # TODO: Implémenter l'évaluation

        logger.info("Entraînement terminé avec succès")

    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()