import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import seaborn as sns

class PricingVisualizer:
    """Visualisation des résultats du pricing RL"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = sns.color_palette("husl", 8)
    
    def plot_training_progress(self, rewards_history: List[float], 
                             window: int = 100, 
                             save_path: str = None):
        """Plot la progression de l'entraînement"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Rewards bruts
        ax1.plot(rewards_history, alpha=0.3, color=self.colors[0])
        
        # Moving average
        if len(rewards_history) >= window:
            moving_avg = pd.Series(rewards_history).rolling(window=window).mean()
            ax1.plot(moving_avg, color=self.colors[1], linewidth=2, 
                    label=f'Moyenne mobile ({window} épisodes)')
        
        ax1.set_title('Progression de l\'Entraînement - Récompenses')
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Récompense')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution des rewards
        ax2.hist(rewards_history, bins=50, alpha=0.7, color=self.colors[2])
        ax2.axvline(np.mean(rewards_history), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(rewards_history):.2f}')
        ax2.set_title('Distribution des Récompenses')
        ax2.set_xlabel('Récompense')
        ax2.set_ylabel('Fréquence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pricing_strategy(self, states_history: List[Dict], 
                            actions_history: List[int],
                            save_path: str = None):
        """Visualise la stratégie de pricing apprise"""
        if not states_history:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Évolution du prix
        prices = [s.get('price_ratio', 0) * 100 for s in states_history]
        ax1.plot(prices, color=self.colors[0], linewidth=2)
        ax1.set_title('Évolution du Ratio de Prix')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Ratio Prix (%)')
        ax1.grid(True, alpha=0.3)
        
        # Évolution du stock
        stocks = [s.get('stock_ratio', 0) * 100 for s in states_history]
        ax2.plot(stocks, color=self.colors[1], linewidth=2)
        ax2.set_title('Évolution du Ratio de Stock')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Ratio Stock (%)')
        ax2.grid(True, alpha=0.3)
        
        # Distribution des actions
        action_names = ['-10%', '-5%', '0%', '+5%', '+10%']
        action_counts = [actions_history.count(i) for i in range(5)]
        ax3.bar(action_names, action_counts, color=self.colors[2:7], alpha=0.7)
        ax3.set_title('Distribution des Actions de Pricing')
        ax3.set_xlabel('Action')
        ax3.set_ylabel('Nombre')
        
        # Prix vs Demande
        demands = [s.get('demand_trend', 0) * 20 for s in states_history]
        scatter = ax4.scatter(prices, demands, c=stocks, cmap='viridis', alpha=0.6)
        ax4.set_title('Relation Prix vs Demande')
        ax4.set_xlabel('Ratio Prix (%)')
        ax4.set_ylabel('Demande')
        plt.colorbar(scatter, ax=ax4, label='Ratio Stock (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_competitor_analysis(self, our_prices: List[float],
                               competitor_prices: Dict[str, List[float]],
                               save_path: str = None):
        """Analyse comparative avec les concurrents"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Évolution des prix
        ax1.plot(our_prices, label='Notre Prix', color=self.colors[0], linewidth=2)
        
        for i, (comp_name, comp_prices) in enumerate(competitor_prices.items()):
            ax1.plot(comp_prices, label=comp_name, color=self.colors[i+1], 
                    alpha=0.7, linewidth=1.5)
        
        ax1.set_title('Évolution des Prix vs Concurrents')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Prix (€)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Positionnement prix
        price_differences = []
        for our_price, comp_prices in zip(our_prices, zip(*competitor_prices.values())):
            avg_comp = np.mean(comp_prices)
            price_differences.append(our_price - avg_comp)
        
        ax2.plot(price_differences, color=self.colors[5], linewidth=2)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Différence de Prix vs Concurrents (Moyenne)')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Différence de Prix (€)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()