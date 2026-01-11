# Sales Agents Project

## Introduction

Le **Sales Agents Project** est un système intelligent de pricing dynamique qui utilise l'apprentissage par renforcement (RL) pour optimiser les stratégies de vente. Le projet combine des agents spécialisés pour différents segments clients et catégories produits, permettant une adaptation fine des prix en temps réel. L'architecture modulaire intègre des environnements RL Gymnasium, des agents rule-based et des modèles fine-tunés, offrant une solution complète pour l'optimisation commerciale automatisée.

Le système vise à maximiser les profits tout en tenant compte des contraintes du marché, de la concurrence et des comportements clients. Il utilise une approche multi-agent où chaque agent spécialisé optimise une partie spécifique du processus de pricing.

## Modélisation MDP (Markov Decision Process)

Le projet repose sur un modèle MDP pour l'optimisation du pricing dynamique. Voici la formalisation mathématique :

### États (States)

L'état du système est représenté par un vecteur de 8 dimensions normalisées entre -1 et 1 :

**S = [stock_norm, price_norm, seasonality, weekend, competition, economy, profit_norm, step_norm]**

- `stock_norm` : Stock normalisé (stock/100.0 * 2 - 1)
- `price_norm` : Prix actuel normalisé (current_price/200.0 * 2 - 1)
- `seasonality` : Saisonnalité (sin(2π * step/30))
- `weekend` : Indicateur weekend (1.0 si weekend, -1.0 sinon)
- `competition` : Niveau de concurrence (aléatoire uniforme [-1,1])
- `economy` : État économique (aléatoire uniforme [-1,1])
- `profit_norm` : Profit total normalisé (total_profit/1000.0)
- `step_norm` : Étape normalisée (step/max_steps * 2 - 1)

**Fichier** : `environments/pricing_env.py` (lignes 25-35 pour l'espace d'observation, lignes 58-75 pour `_get_state()`)

### Actions (Actions)

L'espace d'actions est discret avec 5 actions possibles :

**A = {-10%, -5%, 0%, +5%, +10%}**

Chaque action représente un changement relatif du prix :
- Action 0 : -10% (réduction agressive)
- Action 1 : -5% (réduction modérée)
- Action 2 : 0% (prix stable)
- Action 3 : +5% (augmentation modérée)
- Action 4 : +10% (augmentation forte)

**Fichier** : `environments/pricing_env.py` (lignes 23 pour l'espace d'actions, lignes 76-79 pour `_action_to_change()`)

### Récompenses et Coûts (Rewards and Costs)

La fonction de récompense combine plusieurs composantes :

**R(s,a,s') = R_profit + R_sales - C_change**

Où :
- **R_profit** = profit/100.0 (récompense basée sur le profit)
- **R_sales** = sales/20.0 (bonus basé sur les ventes)
- **C_change** = |price_change| × 3.0 (pénalité pour les changements de prix)

Le profit est calculé comme : **profit = (sales × current_price) - (sales × cost_price)**

**Fichier** : `environments/pricing_env.py` (lignes 110-117 pour `_calculate_reward()`)

### Relations Mathématiques

#### Modèle de Demande
La demande suit une fonction d'élasticité-prix :

**demand = max(0, base_demand × price_effect × seasonal × noise)**

Où :
- **price_effect = (current_price/base_price)^elasticity**
- **elasticity = -1.5** (élasticité constante)
- **seasonal = 1 + 0.3 × sin(2π × step/30)**
- **noise ~ N(1.0, 0.2)** (bruit gaussien)

**Fichier** : `environments/pricing_env.py` (lignes 81-95 pour `_simulate_demand()`)

#### Dynamique du Système
L'évolution temporelle suit :
- **price_{t+1} = price_t × (1 + action_change)**
- **stock_{t+1} = stock_t - min(demand, stock_t)**
- **profit_total_{t+1} = profit_total_t + profit_t**

## Agents Utilisés

Le système utilise une architecture multi-agent avec spécialisation par segment client et stratégie de vente.

### Agents par Segment Client

#### 1. Premium Agent
**Rôle** : Optimise le pricing pour clients premium (moins sensibles au prix, focalisés sur la qualité).

**Stratégie** : Prix élevés avec valeur ajoutée, multiplicateur 1.05-1.15 selon la fidélité client.

**Optimisation** : Maximise la marge tout en maintenant la satisfaction client premium.

**Fichier** : `agents/customer_segments/premium_agent.py`

#### 2. Price Sensitive Agent
**Rôle** : Cible les clients sensibles au prix, recherche le meilleur rapport qualité-prix.

**Stratégie** : Prix compétitifs avec promotions ciblées.

**Optimisation** : Balance volume de ventes et marge.

**Fichier** : `agents/customer_segments/price_sensitive_agent.py`

#### 3. Urgent Agent
**Rôle** : Gère les situations d'urgence (rupture de stock, promotions flash).

**Stratégie** : Ajustements rapides selon l'urgence du contexte.

**Optimisation** : Répond rapidement aux contraintes temporelles.

**Fichier** : `agents/customer_segments/urgent_agent.py`

### Agents par Catégorie Produit

#### 1. Electronics Agent
**Rôle** : Spécialisé dans l'électronique (produits technologiques).

**Stratégie** : Pricing basé sur l'innovation et l'obsolescence rapide.

**Optimisation** : Suit les cycles de vie produit courts.

**Fichier** : `agents/product_categories/electronics_agent.py`

#### 2. Fashion Agent
**Rôle** : Gère les produits de mode (saisonnalité forte).

**Stratégie** : Prix dynamiques selon les tendances saisonnières.

**Optimisation** : Maximise les ventes en période de pointe.

**Fichier** : `agents/product_categories/fashion_agent.py`

#### 3. Home Agent
**Rôle** : Produits d'ameublement et décoration.

**Stratégie** : Prix stables avec focus sur la valeur perçue.

**Optimisation** : Maintient la fidélité client à long terme.

**Fichier** : `agents/product_categories/home_agent.py`

### Stratégies de Vente

#### 1. Aggressive Pricing
**Rôle** : Conquête de marché par prix bas.

**Stratégie** : Sous-cotation par rapport aux concurrents (-5%).

**Optimisation** : Volume de ventes prioritaire.

**Fichier** : `agents/sales_strategies/aggressive_pricing.py`

#### 2. Bundle Strategy
**Rôle** : Vente groupée pour augmenter la valeur perçue.

**Stratégie** : Réductions sur packs de produits.

**Optimisation** : Valeur client et rétention.

**Fichier** : `agents/sales_strategies/bundle_strategy.py`

#### 3. Value Based Pricing
**Rôle** : Prix basé sur la valeur perçue par le client.

**Stratégie** : Segmentation fine selon la valeur client.

**Optimisation** : Profit unitaire maximisé.

**Fichier** : `agents/sales_strategies/value_based.py`

## Modèles et Algorithmes

### Modèles d'Apprentissage par Renforcement

#### 1. Stable Baselines3 (SB3)
**Utilisation** : Fine-tuning des politiques d'agents.

**Algorithmes** : PPO, SAC, DQN selon le problème.

**Fichier** : `pretrained_models/sb3_zoo/`

#### 2. Modèles Pré-entraînés
**Utilisation** : Transfer learning pour accélérer l'apprentissage.

**Types** : Modèles custom et HuggingFace.

**Fichiers** :
- `pretrained_models/custom_trained/`
- `pretrained_models/huggingface/`

### Fine-tuning et Adaptation

#### 1. Sales Trainer
**Rôle** : Entraînement spécialisé pour scénarios de vente.

**Méthodes** : Curriculum learning et transfer learning.

**Fichier** : `fine_tuning/trainers/sales_trainer.py`

#### 2. Domain Transfer
**Rôle** : Adaptation inter-domaines (ex: retail vers e-commerce).

**Technique** : Fine-tuning progressif.

**Fichier** : `fine_tuning/adapters/domain_transfer.py`

#### 3. Sales Adapter
**Rôle** : Adaptation des modèles aux données de vente spécifiques.

**Technique** : Ré-entraînement partiel.

**Fichier** : `fine_tuning/adapters/sales_adapter.py`

### Évaluateurs et Benchmarks

#### 1. Sales Evaluator
**Rôle** : Évaluation des performances en conditions réelles.

**Métriques** : Profit, volume, satisfaction client.

**Fichier** : `fine_tuning/evaluators/sales_evaluator.py`

#### 2. Benchmark
**Rôle** : Comparaison avec stratégies baselines.

**Tests** : A/B testing automatisé.

**Fichier** : `fine_tuning/evaluators/benchmark.py`

## Architecture Technique

### Environnements RL
- **Pricing Environment** : Environnement principal pour fine-tuning.
- **Multi-product Environment** : Gestion de catalogues complexes.
- **Negotiation Environment** : Simulation de négociations clients.

**Fichiers** : `environments/`

### API et Déploiement
- **FastAPI Application** : Interface REST pour décisions temps réel.
- **Agent Orchestrator** : Coordination des agents spécialisés.
- **Monitoring** : Métriques et alertes en temps réel.

**Fichiers** : `deployment/`

### Base de Données et Données
- **MySQL** : Stockage persistant des données clients/produits.
- **Redis** : Cache pour décisions rapides.
- **Historique des ventes** : Données pour l'apprentissage.

**Fichiers** : `data/`

## Conclusion

Le Sales Agents Project démontre une approche innovante de l'optimisation commerciale via l'apprentissage par renforcement. En combinant modélisation MDP rigoureuse, architecture multi-agent spécialisée et fine-tuning adaptatif, le système offre une solution scalable pour le pricing dynamique. Les relations mathématiques formalisent les dynamiques marché, tandis que les agents spécialisés permettent une adaptation fine aux différents contextes commerciaux. Cette architecture modulaire facilite l'extension et l'adaptation à de nouveaux domaines.

---

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.11+
- MySQL 8.0+
- Redis 7+
- Docker & Docker Compose

### Installation Rapide
```bash
python start.py
```

### Lancement de l'API
```bash
python scripts/run_api.py
```

L'API sera disponible sur `http://localhost:8000`

### Tests
```bash
python run_tests.py
```

### Scripts Disponibles
- `run_training.py` : Entraînement des agents RL
- `run_simulation.py` : Simulation de scénarios de vente
- `evaluate_agent.py` : Évaluation des performances
- `fine_tune_sales.py` : Fine-tuning spécialisé

## 📊 Monitoring
- **API Docs** : `http://localhost:8000/docs`
- **Health Check** : `http://localhost:8000/health`
- **Métriques** : Dashboard intégré

## 🤝 Contribution
1. Fork le projet
2. Créer une branche feature
3. Commiter vos changements
4. Ouvrir une Pull Request

## 📄 Licence
MIT License