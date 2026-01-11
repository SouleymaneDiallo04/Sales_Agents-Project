"""
Tests unitaires pour les agents de segments clients
"""

import pytest
import numpy as np
from agents.customer_segments.premium_agent import PremiumAgent

class TestPremiumAgent:
    """Tests pour l'agent Premium"""
    
    def test_init(self, mock_db_connection):
        """Test d'initialisation de l'agent"""
        agent = PremiumAgent(mock_db_connection)
        assert agent.db == mock_db_connection
        assert agent.strategy_type == "premium"
    
    def test_decide_price_loyal_customer(self, mock_db_connection, sample_customer_profile):
        """Test de décision de prix pour client loyal"""
        agent = PremiumAgent(mock_db_connection)
        
        result = agent.decide_price('test_product', sample_customer_profile)
        
        assert result['agent'] == 'premium'
        assert result['current_price'] == 100.0
        assert result['recommended_price'] > 100.0  # Prix augmenté
        assert result['price_change_percent'] > 0
        assert result['confidence'] == 0.75
        assert result['strategy'] == 'value_based_pricing'
        assert 'premium' in result['explanation'].lower()
    
    def test_decide_price_regular_customer(self, mock_db_connection):
        """Test de décision de prix pour client régulier"""
        agent = PremiumAgent(mock_db_connection)
        customer_profile = {'loyalty': 0.4}  # Pas assez loyal
        
        result = agent.decide_price('test_product', customer_profile)
        
        assert result['agent'] == 'premium'
        assert result['recommended_price'] > 100.0
        # Pour client non loyal, augmentation moindre
        assert result['price_change_percent'] < 15  # Environ 5%
    
    def test_decide_price_no_customer_profile(self, mock_db_connection):
        """Test de décision de prix sans profil client"""
        agent = PremiumAgent(mock_db_connection)
        
        result = agent.decide_price('test_product')
        
        assert result['agent'] == 'premium'
        assert result['recommended_price'] > 100.0
    
    def test_validate_price_within_bounds(self, mock_db_connection):
        """Test de validation de prix dans les limites"""
        agent = PremiumAgent(mock_db_connection)
        
        # Prix dans les limites
        validated = agent._validate_price('test_product', 120.0)
        assert validated == 120.0
    
    def test_validate_price_below_min(self, mock_db_connection):
        """Test de validation de prix en dessous du minimum"""
        agent = PremiumAgent(mock_db_connection)
        
        # Prix en dessous du min
        validated = agent._validate_price('test_product', 30.0)
        assert validated == 50.0  # Min price
    
    def test_validate_price_above_max(self, mock_db_connection):
        """Test de validation de prix au-dessus du maximum"""
        agent = PremiumAgent(mock_db_connection)
        
        # Prix au-dessus du max
        validated = agent._validate_price('test_product', 250.0)
        assert validated == 200.0  # Max price