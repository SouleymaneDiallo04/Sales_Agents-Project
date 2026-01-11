"""
Configuration pytest pour les tests du projet Sales Agents
"""

import pytest
import sys
import os
from unittest.mock import MagicMock

# Ajouter le répertoire racine au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def mock_db_connection():
    """Fixture pour une connexion de base de données mockée"""
    db = MagicMock()
    
    # Mock des méthodes de base de données
    db.get_product.return_value = {
        'product_id': 'test_product',
        'current_price': 100.0,
        'min_price': 50.0,
        'max_price': 200.0,
        'category': 'electronics'
    }
    
    db.save_pricing_decision.return_value = True
    db.get_customer_history.return_value = []
    
    return db

@pytest.fixture
def sample_customer_profile():
    """Fixture pour un profil client exemple"""
    return {
        'customer_id': 'test_customer',
        'loyalty': 0.8,
        'segment': 'premium',
        'purchase_history': ['product1', 'product2'],
        'avg_spending': 150.0
    }

@pytest.fixture
def sample_product_data():
    """Fixture pour des données produit exemple"""
    return {
        'product_id': 'test_product',
        'name': 'Test Product',
        'category': 'electronics',
        'current_price': 100.0,
        'min_price': 50.0,
        'max_price': 200.0,
        'cost': 40.0,
        'stock': 100
    }