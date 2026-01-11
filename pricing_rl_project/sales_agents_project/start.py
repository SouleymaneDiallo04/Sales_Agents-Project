#!/usr/bin/env python3
"""
Script de lancement rapide de l'application Sales Agents
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Vérifier que les dépendances sont installées"""
    try:
        import fastapi
        import gymnasium
        import torch
        print("✅ Dépendances vérifiées")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("Installez avec: pip install -r requirements.txt")
        return False

def setup_database():
    """Initialiser la base de données"""
    print("🗄️  Configuration de la base de données...")
    try:
        result = subprocess.run([
            sys.executable, "scripts/setup_database.py"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Base de données configurée")
            return True
        else:
            print(f"❌ Erreur base de données: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def start_api():
    """Démarrer l'API"""
    print("🚀 Démarrage de l'API...")
    try:
        subprocess.run([
            sys.executable, "scripts/run_api.py"
        ])
    except KeyboardInterrupt:
        print("\n🛑 API arrêtée")
    except Exception as e:
        print(f"❌ Erreur API: {e}")

def main():
    """Fonction principale"""
    print("🎯 Sales Agents Project - Lancement rapide")
    print("=" * 50)

    # Vérifier les dépendances
    if not check_requirements():
        return

    # Configuration de la base de données
    if not setup_database():
        print("⚠️  Poursuite sans base de données (mode dégradé)")

    # Démarrer l'API
    start_api()

if __name__ == "__main__":
    main()