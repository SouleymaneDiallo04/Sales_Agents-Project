#!/usr/bin/env python3
"""
Script d'initialisation du projet RL Pricing
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Affiche une bannière stylisée"""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║     RL Pricing Project - Initialisation             ║
    ║     Système de Pricing Dynamique par RL             ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Vérifie la version de Python"""
    import platform
    python_version = platform.python_version()
    print(f"🐍 Python version: {python_version}")
    
    if int(python_version.split('.')[0]) < 3 or int(python_version.split('.')[1]) < 8:
        print("❌ Python 3.8+ requis")
        return False
    return True

def create_project_structure():
    """Crée la structure de projet complète"""
    directories = [
        "environments",
        "models",
        "models/hyperparameters",
        "models/checkpoints",
        "models/best_models",
        "data",
        "data/database",
        "data/simulator",
        "utils",
        "scripts",
        "configs",
        "configs/zoo_configs",
        "logs",
        "logs/tensorboard",
        "logs/evaluations",
        "tests",
        "api"
    ]
    
    print("\n📁 Création de la structure de projet...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}/")
    
    return True

def install_dependencies():
    """Installe les dépendances Python"""
    print("\n📦 Installation des dépendances...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Vérifier si requirements.txt existe
        if os.path.exists("requirements.txt"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dépendances installées depuis requirements.txt")
        else:
            # Installer les dépendances de base
            dependencies = [
                "torch",
                "stable-baselines3",
                "gymnasium",
                "numpy",
                "pandas",
                "mysql-connector-python",
                "fastapi",
                "uvicorn",
                "matplotlib",
                "seaborn",
                "pyyaml"
            ]
            
            for dep in dependencies:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            
            print("✅ Dépendances de base installées")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        return False

def test_imports():
    """Teste les imports essentiels"""
    print("\n🔧 Test des imports...")
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("gymnasium", "Gymnasium"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("mysql.connector", "MySQL Connector"),
        ("fastapi", "FastAPI")
    ]
    
    all_good = True
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError as e:
            print(f"  ❌ {display_name}: {e}")
            all_good = False
    
    return all_good

def create_example_configs():
    """Crée des fichiers de configuration exemple"""
    print("\n⚙️  Création des fichiers de configuration...")
    
    # .env.example
    env_content = """# Configuration de l'environnement RL Pricing

# Base de données
MYSQL_HOST=localhost
MYSQL_DATABASE=rl_data_base
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_PORT=3306

# Produit
PRODUCT_ID=PROD_001
PRODUCT_NAME=Smartphone Galaxy X
COST_PRICE=450.00
MIN_PRICE=495.00
MAX_PRICE=900.00

# Entraînement
TOTAL_TIMESTEPS=100000
N_ENVS=4
EVAL_FREQ=10000

# API
API_HOST=127.0.0.1
API_PORT=8000
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    print("  ✅ .env.example")
    
    return True

def setup_database():
    """Guide pour configurer la base de données"""
    print("\n🗄️  Configuration de la base de données...")
    
    print("""
Pour configurer la base de données MySQL, suivez ces étapes:

1. Créez la base de données:
   CREATE DATABASE rl_data_base;

2. Importez le schéma:
   mysql -u root -p rl_data_base < database/schema.sql

3. Vérifiez la connexion:
   python -c "import mysql.connector; conn = mysql.connector.connect(host='localhost', database='rl_data_base', user='root', password=''); print('✅ Connecté')"

Les tables principales sont:
  - products: Informations produits
  - pricing_states: États de pricing
  - pricing_decisions: Décisions prises
  - pricing_results: Résultats des décisions
  - experience_replay: Expériences pour l'apprentissage
  - rl_models: Métadonnées des modèles
  - competitors: Prix concurrents
  - system_monitoring: Métriques système
""")

def main():
    """Fonction principale d'initialisation"""
    print_banner()
    
    # Vérifications
    if not check_python_version():
        sys.exit(1)
    
    # Création structure
    create_project_structure()
    
    # Installation dépendances
    if not install_dependencies():
        print("⚠️  Installation partielle, continuez manuellement si nécessaire")
    
    # Test imports
    if not test_imports():
        print("⚠️  Certains imports ont échoué, vérifiez les dépendances")
    
    # Configuration
    create_example_configs()
    
    # Base de données
    setup_database()
    
    print("\n" + "="*60)
    print("🎉 Initialisation terminée avec succès!")
    print("="*60)
    
    print("""
Prochaines étapes:
1. Configurez votre base de données MySQL
2. Créez un fichier .env à partir de .env.example
3. Entraînez votre premier modèle:
   python scripts/train.py --algo ppo --timesteps 100000
4. Évaluez le modèle:
   python scripts/evaluate.py --model models/final_ppo_pricing.zip
5. Lancez l'API:
   python scripts/deploy.py --model models/final_ppo_pricing.zip
   
Pour visualiser l'entraînement:
   tensorboard --logdir logs/tensorboard/
""")

if __name__ == "__main__":
    main()