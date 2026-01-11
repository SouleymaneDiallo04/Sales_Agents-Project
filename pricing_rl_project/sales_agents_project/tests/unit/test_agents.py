"""
Tests unitaires pour les agents
"""

import pytest
from unittest.mock import MagicMock
from agents.customer_segments.premium_agent import PremiumAgent
from agents.customer_segments.price_sensitive_agent import PriceSensitiveAgent


class TestPremiumAgent:
    """Tests pour l'agent premium"""

    def test_initialization(self):
        """Test d'initialisation de l'agent premium"""
        mock_db = MagicMock()
        agent = PremiumAgent(mock_db)
        assert agent.strategy_type == "premium"
        assert agent.db == mock_db

    def test_price_decision_loyal_customer(self):
        """Test de décision de prix pour client loyal"""
        mock_db = MagicMock()
        mock_db.get_product.return_value = {
            'current_price': 100.0,
            'min_price': 50.0,
            'max_price': 200.0
        }

        agent = PremiumAgent(mock_db)
        customer_profile = {'loyalty': 0.8}

        result = agent.decide_price('test_product', customer_profile)

        assert result['agent'] == 'premium'
        assert result['current_price'] == 100.0
        assert result['recommended_price'] > 100.0  # Prix augmenté
        assert 'confidence' in result
        assert 'strategy' in result

    def test_price_validation(self):
        """Test de validation des prix"""
        mock_db = MagicMock()
        mock_db.get_product.return_value = {
            'min_price': 50.0,
            'max_price': 200.0
        }

        agent = PremiumAgent(mock_db)

        # Test prix trop bas
        assert agent._validate_price('test', 30.0) == 50.0
        # Test prix trop haut
        assert agent._validate_price('test', 250.0) == 200.0
        # Test prix valide
        assert agent._validate_price('test', 100.0) == 100.0


class TestPriceSensitiveAgent:
    """Tests pour l'agent sensible au prix"""

    def test_initialization(self):
        """Test d'initialisation de l'agent price sensitive"""
        mock_db = MagicMock()
        agent = PriceSensitiveAgent(mock_db)
        assert agent.strategy_type == "price_sensitive"
        assert agent.db == mock_db

    def test_competitive_pricing(self):
        """Test de stratégie de prix compétitif"""
        mock_db = MagicMock()
        mock_db.get_product.return_value = {
            'current_price': 100.0,
            'min_price': 40.0,
            'max_price': 150.0
        }

        agent = PriceSensitiveAgent(mock_db)
        context = {'competitor_prices': [90.0, 95.0, 85.0]}

        result = agent.decide_price('test_product', context)

        assert result['agent'] == 'price_sensitive'
        assert result['recommended_price'] <= 100.0  # Prix réduit
        assert 'market_analysis' in result