"""
Déploiement API et monitoring
Imports robustes avec gestion des modules manquants
"""

import logging
import warnings

logger = logging.getLogger(__name__)

# Imports principaux (existent)
try:
    from .api.fastapi_app import app, PricingAPI
    logger.info("✅ API imports réussis")
except ImportError as e:
    logger.error(f"❌ Erreur import API: {e}")
    app = None
    PricingAPI = None

try:
    from .api.pricing_endpoints import router as pricing_router
    logger.info("✅ Pricing endpoints import réussi")
except ImportError as e:
    logger.error(f"❌ Erreur import pricing_endpoints: {e}")
    pricing_router = None

try:
    from .orchestrator.agent_orchestrator import AgentOrchestrator
    logger.info("✅ Agent orchestrator import réussi")
except ImportError as e:
    logger.error(f"❌ Erreur import agent_orchestrator: {e}")
    AgentOrchestrator = None

# Imports monitoring (placeholders pour modules vides/manquants)
try:
    from .monitoring.dashboard import SalesDashboard
    logger.info("✅ SalesDashboard import réussi")
except ImportError as e:
    logger.warning(f"⚠️ SalesDashboard non disponible: {e}")
    # Placeholder class
    class SalesDashboard:
        """Placeholder pour SalesDashboard - à implémenter"""
        def __init__(self):
            self.name = "SalesDashboard"
            self.status = "placeholder"

        def get_dashboard_data(self):
            return {"status": "placeholder", "message": "Dashboard non implémenté"}

    warnings.warn("SalesDashboard n'est pas implémenté - utilisant placeholder", UserWarning)

try:
    from .monitoring.metrics_collector import MetricsCollector
    logger.info("✅ MetricsCollector import réussi")
except ImportError as e:
    logger.warning(f"⚠️ MetricsCollector non disponible: {e}")
    # Placeholder class
    class MetricsCollector:
        """Placeholder pour MetricsCollector - à implémenter"""
        def __init__(self):
            self.name = "MetricsCollector"
            self.status = "placeholder"

        def collect_metrics(self):
            return {"status": "placeholder", "message": "Metrics collector non implémenté"}

    warnings.warn("MetricsCollector n'est pas implémenté - utilisant placeholder", UserWarning)

try:
    from .monitoring.alert_system import AlertSystem
    logger.info("✅ AlertSystem import réussi")
except ImportError as e:
    logger.warning(f"⚠️ AlertSystem non disponible: {e}")
    # Placeholder class
    class AlertSystem:
        """Placeholder pour AlertSystem - à implémenter"""
        def __init__(self):
            self.name = "AlertSystem"
            self.status = "placeholder"

        def send_alert(self, message):
            logger.info(f"ALERT (placeholder): {message}")
            return {"status": "placeholder", "message": f"Alerte simulée: {message}"}

    warnings.warn("AlertSystem n'est pas implémenté - utilisant placeholder", UserWarning)

# Import strategy selector (potentiellement manquant)
try:
    from .orchestrator.strategy_selector import StrategySelector
    logger.info("✅ StrategySelector import réussi")
except ImportError as e:
    logger.warning(f"⚠️ StrategySelector non disponible: {e}")
    # Placeholder class
    class StrategySelector:
        """Placeholder pour StrategySelector - à implémenter"""
        def __init__(self):
            self.name = "StrategySelector"
            self.status = "placeholder"

        def select_strategy(self, context):
            return {"strategy": "default", "status": "placeholder"}

    warnings.warn("StrategySelector n'est pas implémenté - utilisant placeholder", UserWarning)

# Liste des exports disponibles
__all__ = []

# Ajouter seulement les imports réussis à __all__
if app is not None:
    __all__.append('app')
if PricingAPI is not None:
    __all__.append('PricingAPI')
if pricing_router is not None:
    __all__.append('pricing_router')
if AgentOrchestrator is not None:
    __all__.append('AgentOrchestrator')
if StrategySelector is not None:
    __all__.append('StrategySelector')
if SalesDashboard is not None:
    __all__.append('SalesDashboard')
if MetricsCollector is not None:
    __all__.append('MetricsCollector')
if AlertSystem is not None:
    __all__.append('AlertSystem')

logger.info(f"Imports terminés. Modules disponibles: {__all__}")

# Fonction utilitaire pour vérifier l'état des imports
def get_import_status():
    """Retourne l'état de tous les imports"""
    return {
        "app": app is not None,
        "PricingAPI": PricingAPI is not None,
        "pricing_router": pricing_router is not None,
        "AgentOrchestrator": AgentOrchestrator is not None,
        "StrategySelector": StrategySelector is not None,
        "SalesDashboard": isinstance(SalesDashboard, type) and SalesDashboard.__name__ == "SalesDashboard",
        "MetricsCollector": isinstance(MetricsCollector, type) and MetricsCollector.__name__ == "MetricsCollector",
        "AlertSystem": isinstance(AlertSystem, type) and AlertSystem.__name__ == "AlertSystem",
        "available_modules": __all__
    }