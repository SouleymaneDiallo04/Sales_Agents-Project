"""
Tests unitaires pour les environnements RL
"""

import pytest
import numpy as np
from environments.pricing_env import PricingEnvironment

class TestPricingEnvironment:
    """Tests pour l'environnement de pricing"""
    
    def test_init(self):
        """Test d'initialisation de l'environnement"""
        env = PricingEnvironment()
        
        assert env.max_steps == 30
        assert env.current_step == 0
        assert env.base_price == 100.0
        assert env.cost_price == 50.0
        assert env.current_price == 100.0
        assert env.stock == 100
        assert env.action_space.n == 5
        assert env.observation_space.shape == (8,)
    
    def test_reset(self):
        """Test de reset de l'environnement"""
        env = PricingEnvironment()
        
        # Modifier l'état
        env.current_step = 10
        env.current_price = 120.0
        env.stock = 80
        
        state, info = env.reset()
        
        assert env.current_step == 0
        assert env.current_price == 100.0
        assert env.stock == 100
        assert env.total_profit == 0
        assert isinstance(state, np.ndarray)
        assert state.shape == (8,)
        assert isinstance(info, dict)
    
    def test_action_to_change(self):
        """Test de conversion action vers changement de prix"""
        env = PricingEnvironment()
        
        assert env._action_to_change(0) == -0.10  # -10%
        assert env._action_to_change(1) == -0.05  # -5%
        assert env._action_to_change(2) == 0.0    # 0%
        assert env._action_to_change(3) == 0.05   # +5%
        assert env._action_to_change(4) == 0.10   # +10%
    
    def test_step_basic(self):
        """Test de base de l'étape step"""
        env = PricingEnvironment()
        initial_price = env.current_price
        initial_stock = env.stock
        
        action = 3  # +5%
        state, reward, terminated, truncated, info = env.step(action)
        
        # Prix devrait avoir changé
        assert env.current_price != initial_price
        assert env.current_price == initial_price * 1.05
        
        # Stock devrait avoir diminué ou pas
        assert env.stock <= initial_stock
        
        # État retourné
        assert isinstance(state, np.ndarray)
        assert state.shape == (8,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_get_state(self):
        """Test de récupération de l'état"""
        env = PricingEnvironment()
        state = env._get_state()
        
        assert isinstance(state, np.ndarray)
        assert state.shape == (8,)
        
        # Vérifier que toutes les valeurs sont dans [-1, 1]
        assert np.all(state >= -1.0)
        assert np.all(state <= 1.0)
    
    def test_episode_termination(self):
        """Test de terminaison d'épisode"""
        env = PricingEnvironment()
        
        # Avancer jusqu'à max_steps
        for _ in range(env.max_steps):
            _, _, terminated, _, _ = env.step(2)  # Action neutre
        
        assert terminated
        
        # Reset devrait permettre de continuer
        state, _ = env.reset()
        assert not env._is_done()
        assert env.current_step == 0