#!/usr/bin/env python3
"""
Script principal pour fine-tuning d'agent de pricing
"""

import sys
import argparse
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.database.connection import DatabaseManager
from environments.pricing_env import PricingEnvironment
from fine_tuning.trainers.sales_trainer import SalesTrainer
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning agent pricing")
    parser.add_argument('--product', type=str, default='PROD_001',
                       help="ID du produit")
    parser.add_argument('--steps', type=int, default=50000,
                       help="Nombre de steps d'entraînement")
    parser.add_argument('--model', type=str, default=None,
                       help="Chemin vers modèle pré-entraîné")
    parser.add_argument('--adaptation', type=int, default=5000,
                       help="Steps d'adaptation domaine")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 FINE-TUNING AGENT DE PRICING RL")
    print("=" * 60)
    
    try:
        # 1. Connexion DB
        logger.info("📊 Connexion base de données...")
        db = DatabaseManager()
        
        # 2. Vérifier produit
        product = db.get_product(args.product)
        if not product:
            logger.error(f"❌ Produit {args.product} non trouvé")
            return
        
        logger.info(f"📦 Produit: {product['product_name']}")
        logger.info(f"💰 Prix actuel: {product.get('current_price', 'N/A')}€")
        logger.info(f"📊 Stock: {product.get('current_stock', 'N/A')}")
        
        # 3. Créer environnement
        logger.info("🎮 Création environnement...")
        env = DummyVecEnv([lambda: PricingEnvironment()])
        
        # 4. Fine-tuning
        logger.info("🤖 Initialisation fine-tuner...")
        fine_tuner = SalesTrainer(db)
        
        # 5. Charger modèle pré-entraîné si fourni
        if args.model:
            logger.info("🔄 Chargement modèle pré-entraîné...")
            fine_tuner.load_pretrained(args.model)
        
        # 6. Fine-tuning sur le produit
        logger.info("🎓 Fine-tuning sur le produit...")
        model = fine_tuner.fine_tune_on_product(
            args.product, 
            total_steps=args.steps
        )
        
        # 7. Test rapide
        logger.info("🧪 Test rapide du modèle fine-tuné...")
        test_model(model, env, args.product, db)
        
        logger.info("✅ Fine-tuning terminé avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
    finally:
        if 'db' in locals():
            db.close()

def test_model(model, env, product_id, db, n_episodes=3):
    """Tester le modèle fine-tuné"""
    print(f"\n🧪 TEST SUR {n_episodes} ÉPISODES:")
    
    total_profits = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_profit = 0
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)  # Nouvelle API Gym
            done = terminated or truncated  # Logique de fin d'épisode
            
            if 'profit' in info[0]:
                episode_profit += info[0]['profit']
            
            steps += 1
            if steps >= 10:  # Test court
                break
        
        total_profits.append(episode_profit)
        print(f"  Épisode {episode + 1}: Profit = {episode_profit:.2f}€")
    
    if total_profits:
        avg_profit = np.mean(total_profits)
        print(f"\n📊 Profit moyen: {avg_profit:.2f}€")
        
        # Sauvegarder résultats
        db.log_system_event(
            component="fine_tuning",
            level="INFO",
            message=f"Test modèle {product_id}",
            metrics={
                'avg_profit': avg_profit,
                'n_episodes': n_episodes,
                'max_profit': max(total_profits),
                'min_profit': min(total_profits)
            }
        )

if __name__ == "__main__":
    main()