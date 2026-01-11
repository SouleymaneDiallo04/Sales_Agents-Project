#!/usr/bin/env python3
"""
Script de lancement de l'API Sales Agents
Démarre le serveur FastAPI avec toutes les configurations
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Fonction principale pour démarrer l'API"""

    # Variables d'environnement
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    logger.info(f"Démarrage de l'API Sales Agents sur {host}:{port}")

    # Configuration uvicorn
    uvicorn_config = {
        "app": "deployment.api.fastapi_app:app",
        "host": host,
        "port": port,
        "workers": workers,
        "reload": reload,
        "log_level": "info",
        "access_log": True
    }

    # Démarrer le serveur
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Arrêt de l'API demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de l'API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()