#!/usr/bin/env python3
"""
Étude d'ablation pour analyser l'impact du fine-tuning
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml
import json

from models.finetune_trainer import FineTuneTrainer
from environments.pricing_env import EcommercePricingEnv

class AblationStudy:
    """
    Étude d'ablation pour analyser différents aspects du fine-tuning
    """
    
    def __init__(self, pretrained_model_path: str):
        self.pretrained_model_path = pretrained_model_path
        self.results = {}
        
        # Configurations à tester
        self.ablation_configs = self._generate_ablation_configs()
    
    def _generate_ablation_configs(self) -> List[Dict[str, Any]]:
        """Génère les configurations pour l'étude d'ablation"""
        base_config = {
            'finetune': {
                'learning_rate': 0.0001,
                'n_steps': 1024,
                'clip_range': 0.1,
                'ent_coef': 0.01,
            },
            'adaptation': {
                'strategy': 'feature_extractor',
                'freeze_layers': 2,
                'unfreeze_after': 0.5,
            },
            'training': {
                'total_timesteps': 20000,  # Court pour les tests
                'n_envs': 2,
                'eval_freq': 2000,
            }
        }
        
        # Variations à tester
        configs = []
        
        # 1. Différentes stratégies d'adaptation
        for strategy in ['feature_extractor', 'partial_reinit', 'full_reinit']:
            config = base_config.copy()
            config['adaptation']['strategy'] = strategy
            configs.append({
                'name': f'strategy_{strategy}',
                'config': config,
                'description': f'Stratégie: {strategy}'
            })
        
        # 2. Différents nombres de couches gelées
        for freeze_layers in [0, 1, 2, 3, 4]:
            config = base_config.copy()
            config['adaptation']['strategy'] = 'feature_extractor'
            config['adaptation']['freeze_layers'] = freeze_layers
            configs.append({
                'name': f'freeze_{freeze_layers}',
                'config': config,
                'description': f'{freeze_layers} couches gelées'
            })
        
        # 3. Différents learning rates
        for lr in [0.00001, 0.00005, 0.0001, 0.0005, 0.001]:
            config = base_config.copy()
            config['finetune']['learning_rate'] = lr
            configs.append({
                'name': f'lr_{lr}',
                'config': config,
                'description': f'LR: {lr}'
            })
        
        # 4. Différents timesteps de fine-tuning
        for timesteps in [5000, 10000, 20000, 50000]:
            config = base_config.copy()
            config['training']['total_timesteps'] = timesteps
            configs.append({
                'name': f'timesteps_{timesteps}',
                'config': config,
                'description': f'{timesteps} timesteps'
            })
        
        # 5. Baseline: from-scratch
        configs.append({
            'name': 'from_scratch',
            'config': None,  # Spécial: entraînement from-scratch
            'description': 'Entraînement from-scratch (baseline)'
        })
        
        return configs
    
    def run_ablation_study(self, 
                          num_runs: int = 3,
                          num_eval_episodes: int = 10) -> pd.DataFrame:
        """
        Exécute l'étude d'ablation
        
        Args:
            num_runs: Nombre de runs par configuration (pour la variance)
            num_eval_episodes: Nombre d'épisodes d'évaluation par run
        
        Returns:
            DataFrame avec résultats
        """
        print(f"🔬 Début de l'étude d'ablation")
        print(f"   Configurations: {len(self.ablation_configs)}")
        print(f"   Runs par config: {num_runs}")
        print(f"   Épisodes par run: {num_eval_episodes}")
        print("=" * 60)
        
        all_results = []
        
        for config_info in self.ablation_configs:
            config_name = config_info['name']
            config_desc = config_info['description']
            
            print(f"\n🔧 Configuration: {config_name} ({config_desc})")
            
            run_results = []
            
            for run in range(num_runs):
                print(f"   Run {run + 1}/{num_runs}...", end=" ")
                
                try:
                    if config_name == 'from_scratch':
                        # Entraînement from-scratch
                        rewards = self._train_from_scratch_and_evaluate(
                            seed=run,
                            num_eval_episodes=num_eval_episodes
                        )
                    else:
                        # Fine-tuning avec configuration spécifique
                        rewards = self._finetune_and_evaluate(
                            config=config_info['config'],
                            seed=run,
                            num_eval_episodes=num_eval_episodes
                        )
                    
                    run_metrics = {
                        'mean_reward': float(np.mean(rewards)),
                        'std_reward': float(np.std(rewards)),
                        'max_reward': float(np.max(rewards)),
                        'min_reward': float(np.min(rewards)),
                        'median_reward': float(np.median(rewards)),
                        'rewards': rewards
                    }
                    
                    run_results.append(run_metrics)
                    print(f"✓ Reward moyen: {run_metrics['mean_reward']:.2f}")
                    
                except Exception as e:
                    print(f"✗ Erreur: {e}")
                    run_results.append({
                        'mean_reward': 0,
                        'std_reward': 0,
                        'max_reward': 0,
                        'min_reward': 0,
                        'median_reward': 0,
                        'rewards': []
                    })
            
            # Résumé pour cette configuration
            if run_results:
                config_summary = {
                    'config_name': config_name,
                    'description': config_desc,
                    'num_runs': num_runs,
                    'mean_reward_across_runs': np.mean([r['mean_reward'] for r in run_results]),
                    'std_reward_across_runs': np.std([r['mean_reward'] for r in run_results]),
                    'best_run_reward': np.max([r['mean_reward'] for r in run_results]),
                    'worst_run_reward': np.min([r['mean_reward'] for r in run_results]),
                    'run_details': run_results
                }
                
                all_results.append(config_summary)
                self.results[config_name] = config_summary
                
                print(f"   📊 Résumé: {config_summary['mean_reward_across_runs']:.2f} ± "
                      f"{config_summary['std_reward_across_runs']:.2f}")
        
        # Convertir en DataFrame
        df_data = []
        for result in all_results:
            df_data.append({
                'config': result['config_name'],
                'description': result['description'],
                'mean_reward': result['mean_reward_across_runs'],
                'std_reward': result['std_reward_across_runs'],
                'best_reward': result['best_run_reward'],
                'worst_reward': result['worst_run_reward'],
                'num_runs': result['num_runs']
            })
        
        df = pd.DataFrame(df_data)
        
        print(f"\n✅ Étude d'ablation terminée")
        print(f"   Résultats pour {len(df)} configurations")
        
        return df
    
    def _finetune_and_evaluate(self, 
                              config: Dict[str, Any],
                              seed: int,
                              num_eval_episodes: int) -> List[float]:
        """Fine-tune et évalue avec une configuration spécifique"""
        # Sauvegarder la config temporairement
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            config_path = f.name
        
        try:
            # Fine-tuning
            trainer = FineTuneTrainer(
                pretrained_model_path=self.pretrained_model_path,
                product_id='PROD_001',
                config_path=config_path
            )
            
            # Modifier la seed dans la config
            config['training']['seed'] = seed
            
            # Fine-tuning court pour l'étude
            trainer.finetune(
                total_timesteps=config['training']['total_timesteps'],
                eval_freq=config['training']['eval_freq']
            )
            
            # Évaluation
            rewards, _ = trainer.evaluate_finetuned_model(
                num_episodes=num_eval_episodes
            )
            
            return rewards
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def _train_from_scratch_and_evaluate(self,
                                        seed: int,
                                        num_eval_episodes: int) -> List[float]:
        """Entraîne from-scratch et évalue (baseline)"""
        from models.rl_trainer import RLTrainer
        
        # Entraînement from-scratch court
        trainer = RLTrainer(algo_name='ppo')
        trainer.create_env(n_envs=2)
        trainer.create_model()
        
        # Entraînement court pour comparaison équitable
        trainer.train(total_timesteps=20000)
        
        # Évaluation
        rewards = trainer.evaluate(num_episodes=num_eval_episodes)
        
        return rewards
    
    def analyze_results(self, results_df: pd.DataFrame):
        """Analyse les résultats de l'étude d'ablation"""
        print(f"\n📊 Analyse des résultats d'ablation")
        print("=" * 60)
        
        # 1. Meilleure configuration
        best_config = results_df.loc[results_df['mean_reward'].idxmax()]
        worst_config = results_df.loc[results_df['mean_reward'].idxmin()]
        
        print(f"\n🏆 MEILLEURE CONFIGURATION:")
        print(f"   {best_config['config']} ({best_config['description']})")
        print(f"   Reward: {best_config['mean_reward']:.2f} ± {best_config['std_reward']:.2f}")
        
        print(f"\n⚠️  PIRE CONFIGURATION:")
        print(f"   {worst_config['config']} ({worst_config['description']})")
        print(f"   Reward: {worst_config['mean_reward']:.2f} ± {worst_config['std_reward']:.2f}")
        
        # 2. Comparaison avec baseline
        baseline_row = results_df[results_df['config'] == 'from_scratch']
        if not baseline_row.empty:
            baseline = baseline_row.iloc[0]
            
            # Amélioration de la meilleure config vs baseline
            improvement = ((best_config['mean_reward'] - baseline['mean_reward']) / 
                          abs(baseline['mean_reward']) * 100)
            
            print(f"\n📈 COMPARAISON AVEC BASELINE (from-scratch):")
            print(f"   Baseline: {baseline['mean_reward']:.2f} ± {baseline['std_reward']:.2f}")
            print(f"   Amélioration: {improvement:+.1f}%")
            
            # Configurations meilleures que baseline
            better_than_baseline = results_df[
                results_df['mean_reward'] > baseline['mean_reward']
            ]
            print(f"   Configurations meilleures: {len(better_than_baseline)}/{len(results_df)}")
        
        # 3. Analyse par catégorie
        self._analyze_by_category(results_df)
        
        return best_config
    
    def _analyze_by_category(self, results_df: pd.DataFrame):
        """Analyse les résultats par catégorie"""
        print(f"\n🔍 ANALYSE PAR CATÉGORIE:")
        
        # Stratégies d'adaptation
        strategy_mask = results_df['config'].str.startswith('strategy_')
        if strategy_mask.any():
            print(f"\n  📋 STRATÉGIES D'ADAPTATION:")
            strategy_df = results_df[strategy_mask].copy()
            strategy_df['strategy'] = strategy_df['config'].str.replace('strategy_', '')
            
            for _, row in strategy_df.sort_values('mean_reward', ascending=False).iterrows():
                print(f"    {row['strategy']}: {row['mean_reward']:.2f} ± {row['std_reward']:.2f}")
        
        # Couches gelées
        freeze_mask = results_df['config'].str.startswith('freeze_')
        if freeze_mask.any():
            print(f"\n  ❄️  COUCHES GELÉES:")
            freeze_df = results_df[freeze_mask].copy()
            freeze_df['freeze_layers'] = freeze_df['config'].str.replace('freeze_', '').astype(int)
            freeze_df = freeze_df.sort_values('freeze_layers')
            
            for _, row in freeze_df.iterrows():
                print(f"    {row['freeze_layers']} couches: {row['mean_reward']:.2f} ± {row['std_reward']:.2f}")
            
            # Trouver le nombre optimal
            optimal_freeze = freeze_df.loc[freeze_df['mean_reward'].idxmax(), 'freeze_layers']
            print(f"    → Optimal: {optimal_freeze} couches gelées")
        
        # Learning rates
        lr_mask = results_df['config'].str.startswith('lr_')
        if lr_mask.any():
            print(f"\n  📈 LEARNING RATES:")
            lr_df = results_df[lr_mask].copy()
            lr_df['lr'] = lr_df['config'].str.replace('lr_', '').astype(float)
            lr_df = lr_df.sort_values('lr')
            
            for _, row in lr_df.iterrows():
                print(f"    {row['lr']:.6f}: {row['mean_reward']:.2f} ± {row['std_reward']:.2f}")
            
            # Trouver le LR optimal
            optimal_lr = lr_df.loc[lr_df['mean_reward'].idxmax(), 'lr']
            print(f"    → Optimal: LR = {optimal_lr}")
    
    def plot_ablation_results(self, 
                             results_df: pd.DataFrame,
                             save_dir: str = "logs/ablation_studies"):
        """Crée des visualisations pour l'étude d'ablation"""
        
        # Créer le répertoire de sortie
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Bar plot des configurations
        plt.figure(figsize=(12, 6))
        
        # Trier par performance
        sorted_df = results_df.sort_values('mean_reward', ascending=True)
        
        bars = plt.barh(range(len(sorted_df)), sorted_df['mean_reward'], 
                       xerr=sorted_df['std_reward'],
                       alpha=0.7)
        
        # Colorer par catégorie
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            if row['config'] == 'from_scratch':
                bars[i].set_color('gray')
            elif 'strategy' in row['config']:
                bars[i].set_color('skyblue')
            elif 'freeze' in row['config']:
                bars[i].set_color('lightcoral')
            elif 'lr' in row['config']:
                bars[i].set_color('lightgreen')
            elif 'timesteps' in row['config']:
                bars[i].set_color('gold')
        
        plt.yticks(range(len(sorted_df)), sorted_df['description'])
        plt.xlabel('Reward moyen')
        plt.title('Étude d\'Ablation: Performance par Configuration')
        plt.tight_layout()
        
        bar_plot_path = f"{save_dir}/ablation_bar_plot.png"
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Bar plot sauvegardé: {bar_plot_path}")
        
        # 2. Scatter plot: LR vs Performance
        lr_mask = results_df['config'].str.startswith('lr_')
        if lr_mask.any():
            lr_df = results_df[lr_mask].copy()
            lr_df['lr_value'] = lr_df['config'].str.replace('lr_', '').astype(float)
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(lr_df['lr_value'], lr_df['mean_reward'],
                        yerr=lr_df['std_reward'],
                        fmt='o-', capsize=5)
            plt.xscale('log')
            plt.xlabel('Learning Rate (log scale)')
            plt.ylabel('Reward moyen')
            plt.title('Impact du Learning Rate sur le Fine-tuning')
            plt.grid(True, alpha=0.3)
            
            lr_plot_path = f"{save_dir}/learning_rate_impact.png"
            plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 LR plot sauvegardé: {lr_plot_path}")
        
        # 3. Heatmap: Stratégie × Couches gelées
        strategy_mask = results_df['config'].str.startswith('strategy_')
        freeze_mask = results_df['config'].str.startswith('freeze_')
        
        if strategy_mask.any() and freeze_mask.any():
            # Créer une heatmap (simplifiée)
            strategies = results_df[strategy_mask]['config'].tolist()
            freezes = results_df[freeze_mask]['config'].tolist()
            
            # Pour une vraie heatmap, il faudrait tester toutes les combinaisons
            print("⚠️  Heatmap non générée: nécessite toutes combinaisons stratégie×freeze")
        
        # 4. Résumé textuel
        summary_path = f"{save_dir}/ablation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("RÉSUMÉ DE L'ÉTUDE D'ABLATION\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("MEILLEURES CONFIGURATIONS:\n")
            top_5 = results_df.nlargest(5, 'mean_reward')
            for i, (_, row) in enumerate(top_5.iterrows()):
                f.write(f"{i+1}. {row['config']} ({row['description']})\n")
                f.write(f"   Reward: {row['mean_reward']:.2f} ± {row['std_reward']:.2f}\n\n")
            
            f.write("\nRECOMMANDATIONS:\n")
            best_config = results_df.loc[results_df['mean_reward'].idxmax()]
            f.write(f"1. Utiliser: {best_config['config']}\n")
            f.write(f"2. Description: {best_config['description']}\n")
            
            # Analyser les patterns
            f.write("\nOBSERVATIONS:\n")
            
            if 'freeze_' in best_config['config']:
                optimal_freeze = best_config['config'].replace('freeze_', '')
                f.write(f"- Optimal: geler {optimal_freeze} couches\n")
            
            if 'lr_' in best_config['config']:
                optimal_lr = best_config['config'].replace('lr_', '')
                f.write(f"- Optimal: LR = {optimal_lr}\n")
            
            if 'strategy_' in best_config['config']:
                optimal_strategy = best_config['config'].replace('strategy_', '')
                f.write(f"- Optimal stratégie: {optimal_strategy}\n")
        
        print(f"💾 Résumé texte sauvegardé: {summary_path}")
        
        # 5. Sauvegarder les données brutes
        data_path = f"{save_dir}/ablation_results.csv"
        results_df.to_csv(data_path, index=False)
        
        json_path = f"{save_dir}/ablation_results.json"
        results_df.to_json(json_path, orient='records', indent=2)
        
        print(f"💾 Données brutes sauvegardées: {data_path}, {json_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Étude d'ablation pour le fine-tuning"
    )
    
    parser.add_argument(
        "--pretrained", 
        type=str,
        required=True,
        help="Chemin vers le modèle pré-entraîné"
    )
    
    parser.add_argument(
        "--runs", 
        type=int,
        default=3,
        help="Nombre de runs par configuration"
    )
    
    parser.add_argument(
        "--episodes", 
        type=int,
        default=10,
        help="Nombre d'épisodes d'évaluation par run"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="logs/ablation_studies",
        help="Répertoire de sortie"
    )
    
    parser.add_argument(
        "--skip-plot", 
        action="store_true",
        help="Passer la génération des graphiques"
    )
    
    args = parser.parse_args()
    
    # Valider le modèle
    if not os.path.exists(args.pretrained):
        print(f"❌ Modèle non trouvé: {args.pretrained}")
        return 1
    
    # Créer l'étude d'ablation
    study = AblationStudy(args.pretrained)
    
    # Exécuter l'étude
    print("🔬 ÉTUDE D'ABLATION POUR LE FINE-TUNING")
    print("=" * 60)
    
    results_df = study.run_ablation_study(
        num_runs=args.runs,
        num_eval_episodes=args.episodes
    )
    
    # Analyser les résultats
    best_config = study.analyze_results(results_df)
    
    # Visualisations
    if not args.skip_plot:
        study.plot_ablation_results(results_df, save_dir=args.output_dir)
    
    # Sauvegarder la configuration optimale
    optimal_config_path = f"{args.output_dir}/optimal_config.yaml"
    
    # Trouver la configuration optimale dans les configs d'ablation
    for config_info in study.ablation_configs:
        if config_info['name'] == best_config['config']:
            if config_info['config'] is not None:
                with open(optimal_config_path, 'w') as f:
                    yaml.dump(config_info['config'], f, default_flow_style=False)
                print(f"\n💾 Configuration optimale sauvegardée: {optimal_config_path}")
            break
    
    print(f"\n✅ Étude d'ablation terminée!")
    print(f"   Meilleure configuration: {best_config['config']}")
    print(f"   Performance: {best_config['mean_reward']:.2f} ± {best_config['std_reward']:.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())