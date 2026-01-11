# RL Pricing Project

Système de Pricing Dynamique par Reinforcement Learning pour E-commerce.

## 📋 Description

Ce projet implémente un agent de Reinforcement Learning (RL) capable d'ajuster dynamiquement les prix d'un produit e-commerce en fonction du marché, du stock, et des comportements des concurrents.

## 🎯 Fonctionnalités

- ✅ Environnement Gymnasium personnalisé pour le pricing
- ✅ Intégration avec Stable-Baselines3 et SB3 Zoo
- ✅ Entraînement avec PPO, DQN, A2C
- ✅ Simulation de marché réaliste
- ✅ Base de données MySQL pour le stockage des données
- ✅ API FastAPI pour le déploiement
- ✅ Dashboard de monitoring et visualisation

## 🚀 Installation

### 1. Prérequis
- Python 3.8+
- MySQL 5.7+
- Git

### 2. Installation
```bash
# Cloner le dépôt
git clone <votre-repo>
cd rl_pricing_project

# Initialiser le projet
python setup_project.py

# Créer l'environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt