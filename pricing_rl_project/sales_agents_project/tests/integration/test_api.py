"""
Tests d'intégration pour l'API FastAPI
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from deployment.api.fastapi_app import app


@pytest.fixture
def client():
    """Fixture pour le client de test FastAPI"""
    return TestClient(app)


@pytest.fixture
def mock_db():
    """Fixture pour mocker la base de données"""
    with patch('deployment.api.fastapi_app.DatabaseManager') as mock:
        db_instance = MagicMock()
        db_instance.get_product.return_value = {
            'product_id': 'TEST_001',
            'current_price': 100.0,
            'min_price': 50.0,
            'max_price': 200.0
        }
        mock.return_value = db_instance
        yield mock


@pytest.fixture
def mock_orchestrator():
    """Fixture pour mocker l'orchestrateur d'agents"""
    with patch('deployment.api.fastapi_app.AgentOrchestrator') as mock:
        orchestrator_instance = MagicMock()
        orchestrator_instance.get_pricing_decision.return_value = {
            'product_id': 'TEST_001',
            'current_price': 100.0,
            'recommended_price': 110.0,
            'price_change_percent': 10.0,
            'agent_used': 'premium_agent',
            'confidence': 0.85,
            'explanation': 'Test decision',
            'strategy': 'premium_pricing'
        }
        mock.return_value = orchestrator_instance
        yield mock


class TestPricingAPI:
    """Tests d'intégration pour l'API de pricing"""

    def test_health_endpoint(self, client):
        """Test du endpoint de santé"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_pricing_decision_endpoint(self, client, mock_db, mock_orchestrator):
        """Test du endpoint de décision de pricing"""
        request_data = {
            "product_id": "TEST_001",
            "customer_id": "CUST_001",
            "customer_segment": "premium",
            "context": {
                "urgency": "medium",
                "competitor_prices": [95.0, 105.0]
            }
        }

        response = client.post("/api/v1/pricing/decision", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Vérifier la structure de la réponse
        assert "request_id" in data
        assert "product_id" in data
        assert "current_price" in data
        assert "recommended_price" in data
        assert "price_change_percent" in data
        assert "agent_used" in data
        assert "confidence" in data
        assert "explanation" in data
        assert "strategy" in data
        assert "timestamp" in data

        # Vérifier les valeurs
        assert data["product_id"] == "TEST_001"
        assert data["current_price"] == 100.0
        assert data["recommended_price"] == 110.0
        assert data["agent_used"] == "premium_agent"

    def test_batch_pricing_endpoint(self, client, mock_db, mock_orchestrator):
        """Test du endpoint de pricing par lot"""
        request_data = {
            "requests": [
                {
                    "product_id": "TEST_001",
                    "customer_segment": "premium"
                },
                {
                    "product_id": "TEST_002",
                    "customer_segment": "price_sensitive"
                }
            ],
            "priority": "normal"
        }

        response = client.post("/api/v1/pricing/batch-decision", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "batch_id" in data
        assert "decisions" in data
        assert len(data["decisions"]) == 2
        assert "processing_time" in data

    def test_invalid_product_id(self, client, mock_db):
        """Test avec un ID produit invalide"""
        mock_db.return_value.get_product.return_value = None

        request_data = {
            "product_id": "INVALID_PRODUCT",
            "customer_segment": "premium"
        }

        response = client.post("/api/v1/pricing/decision", json=request_data)
        assert response.status_code == 404

    def test_training_endpoint(self, client, mock_db):
        """Test du endpoint d'entraînement"""
        with patch('deployment.api.fastapi_app.SalesFineTuner') as mock_trainer:
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.fine_tune.return_value = {
                "status": "success",
                "model_path": "/path/to/model",
                "metrics": {"final_reward": 150.0}
            }
            mock_trainer.return_value = mock_trainer_instance

            request_data = {
                "product_id": "TEST_001",
                "training_type": "fine_tune",
                "total_steps": 5000
            }

            response = client.post("/api/v1/training/fine-tune", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "success"
            assert "model_path" in data
            assert "metrics" in data