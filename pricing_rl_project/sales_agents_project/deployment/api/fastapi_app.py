"""
API FastAPI pour service de pricing
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import logging
import json

from data.database.connection import DatabaseManager
from deployment.orchestrator.agent_orchestrator import AgentOrchestrator

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder pour datetime
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Modèles Pydantic
class PricingRequest(BaseModel):
    product_id: str = Field(..., description="ID du produit")
    customer_id: Optional[str] = Field(None, description="ID client")
    customer_segment: Optional[str] = Field(None, description="Segment client")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contexte supplémentaire")

class PricingResponse(BaseModel):
    request_id: str
    product_id: str
    current_price: float
    recommended_price: float
    price_change_percent: float
    agent_used: str
    confidence: float
    explanation: str
    strategy: str
    timestamp: datetime
    constraints: Dict[str, Any]

class BatchPricingRequest(BaseModel):
    requests: List[PricingRequest]
    priority: str = "normal"

class TrainingRequest(BaseModel):
    product_id: str
    training_type: str = "fine_tune"
    total_steps: int = 10000
    force_retrain: bool = False

# Application FastAPI
app = FastAPI(
    title="Sales Agents Pricing API",
    description="API de pricing dynamique par Reinforcement Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dépendances
def get_db():
    """Dépendance pour gestionnaire DB"""
    db = DatabaseManager()
    try:
        yield db
    finally:
        db.close()

def get_orchestrator(db: DatabaseManager = Depends(get_db)):
    """Dépendance pour orchestrateur agents"""
    return AgentOrchestrator(db)

# Routes principales
@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "service": "Sales Agents Pricing API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "pricing": "/api/v1/pricing/{product_id}",
            "batch_pricing": "/api/v1/pricing/batch",
            "training": "/api/v1/training",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check(db: DatabaseManager = Depends(get_db)):
    """Check santé de l'API"""
    try:
        # Vérifier DB
        db._execute_query("SELECT 1", fetch_one=True)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),  # ← ISO format
            "components": {
                "database": "connected",
                "api": "running",
                "redis": "connected" if hasattr(db, 'redis_client') and db.redis_client else "not_configured"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# ============================================================
# IMPORTANT: CET ENDPOINT EST DÉSACTIVÉ CAR IL EST BUGGUÉ
# UTILISEZ pricing_router.py À LA PLACE
# ============================================================

# @app.get("/api/v1/pricing/{product_id}")
# async def get_pricing(
#     product_id: str,
#     customer_id: Optional[str] = None,
#     customer_segment: Optional[str] = None,
#     orchestrator: AgentOrchestrator = Depends(get_orchestrator)
# ) -> PricingResponse:
#     """
#     Obtenir recommandation de prix pour un produit
#     """
#     try:
#         logger.info(f"Pricing request: product={product_id}, customer={customer_id}")
        
#         # Préparer contexte
#         context = {
#             "customer_id": customer_id,
#             "customer_segment": customer_segment,
#             "request_time": datetime.now()
#         }
        
#         # Obtenir décision
#         decision = orchestrator.get_pricing_decision(product_id, context)
        
#         if not decision:
#             raise HTTPException(status_code=404, detail="Product not found or inactive")
        
#         # Construire réponse
#         response = PricingResponse(
#             request_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#             product_id=product_id,
#             current_price=decision["current_price"],
#             recommended_price=decision["recommended_price"],
#             price_change_percent=decision["price_change_percent"],
#             agent_used=decision["agent"],
#             confidence=decision["confidence"],
#             explanation=decision["explanation"],
#             strategy=decision["strategy"],
#             timestamp=datetime.now(),
#             constraints={
#                 "min_price": decision.get("min_price", 0),
#                 "max_price": decision.get("max_price", 0),
#                 "max_change_percent": 20.0
#             }
#         )
        
#         # Log la décision
#         logger.info(f"Pricing decision: {product_id} -> {response.recommended_price}€ "
#                    f"(confidence: {response.confidence:.2%})")
        
#         return response
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error in pricing endpoint: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/pricing/batch")
async def batch_pricing(
    batch_request: BatchPricingRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Traitement batch de pricing pour plusieurs produits
    """
    try:
        logger.info(f"Batch pricing request: {len(batch_request.requests)} products")
        
        results = []
        errors = []
        
        for req in batch_request.requests:
            try:
                decision = orchestrator.get_pricing_decision(
                    req.product_id, 
                    req.context or {}
                )
                
                if decision:
                    results.append({
                        "product_id": req.product_id,
                        "recommended_price": decision["recommended_price"],
                        "confidence": decision["confidence"],
                        "agent": decision["agent"],
                        "status": "success"
                    })
                else:
                    errors.append({
                        "product_id": req.product_id,
                        "error": "Product not found or inactive",
                        "status": "error"
                    })
                    
            except Exception as e:
                errors.append({
                    "product_id": req.product_id,
                    "error": str(e),
                    "status": "error"
                })
        
        # Task background: mise à jour batch
        if results:
            background_tasks.add_task(
                orchestrator.process_batch_decisions, 
                results
            )
        
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_requests": len(batch_request.requests),
            "successful": len(results),
            "errors": len(errors),
            "results": results,
            "errors_detail": errors,
            "priority": batch_request.priority,
            "processed_at": datetime.now().isoformat()  # ← ISO format
        }
        
    except Exception as e:
        logger.error(f"Error in batch pricing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/training")
async def start_training(
    train_request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db)
):
    """
    Démarrer un job de training/fine-tuning
    """
    try:
        # Essayer d'importer le module de fine-tuning
        try:
            from scripts.fine_tune_sales import fine_tune_product
            fine_tune_available = True
        except ImportError:
            logger.warning("Fine-tuning module not available")
            fine_tune_available = False
        
        logger.info(f"Training request: {train_request.product_id}, "
                   f"type: {train_request.training_type}")
        
        # Vérifier produit
        product = db.get_product(train_request.product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Lancer training en background si disponible
        if fine_tune_available:
            background_tasks.add_task(
                fine_tune_product,
                train_request.product_id,
                train_request.total_steps,
                train_request.force_retrain
            )
            message = "Training job started in background"
        else:
            message = "Fine-tuning module not available - training queued"
        
        return {
            "training_id": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "product_id": train_request.product_id,
            "training_type": train_request.training_type,
            "total_steps": train_request.total_steps,
            "status": "started",
            "estimated_duration": f"{train_request.total_steps // 1000} minutes",
            "started_at": datetime.now().isoformat(),  # ← ISO format
            "message": message
        }
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/products")
async def list_products(
    active_only: bool = True,
    category: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
):
    """
    Lister tous les produits disponibles
    """
    try:
        query = "SELECT * FROM products WHERE 1=1"
        params = []
        
        if active_only:
            query += " AND is_active = 1"
        
        if category:
            query += " AND category = %s"
            params.append(category)
        
        query += " ORDER BY product_name"
        
        products = db._execute_query(query, tuple(params))
        
        # Convertir les dates en string pour JSON
        for product in products:
            for key in ['created_at', 'updated_at']:
                if key in product and product[key] and isinstance(product[key], datetime):
                    product[key] = product[key].isoformat()
        
        return {
            "count": len(products),
            "products": products,
            "filters": {
                "active_only": active_only,
                "category": category
            },
            "retrieved_at": datetime.now().isoformat()  # ← ISO format
        }
        
    except Exception as e:
        logger.error(f"Error listing products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics/{product_id}")
async def get_product_metrics(
    product_id: str,
    period_days: int = 30,
    db: DatabaseManager = Depends(get_db)
):
    """
    Obtenir métriques de performance pour un produit
    """
    try:
        # Essayer d'importer l'évaluateur
        try:
            from fine_tuning.evaluators.sales_evaluator import SalesEvaluator
            evaluator = SalesEvaluator(db)
            metrics = evaluator.evaluate_agent_performance(
                "all",  # Tous les agents
                product_id,
                period_days
            )
        except ImportError:
            metrics = {
                "message": "Sales evaluator not available",
                "product_id": product_id,
                "period_days": period_days
            }
        
        return {
            "product_id": product_id,
            "period_days": period_days,
            "metrics": metrics,
            "calculated_at": datetime.now().isoformat()  # ← ISO format
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/decisions/{product_id}")
async def get_decision_history(
    product_id: str,
    limit: int = 50,
    db: DatabaseManager = Depends(get_db)
):
    """
    Historique des décisions de pricing pour un produit
    """
    try:
        query = """
            SELECT 
                pd.*,
                pr.units_sold,
                pr.revenue,
                pr.profit,
                pr.reward_value
            FROM pricing_decisions pd
            LEFT JOIN pricing_results pr ON pd.decision_id = pr.decision_id
            WHERE pd.product_id = %s
            ORDER BY pd.timestamp DESC
            LIMIT %s
        """
        
        decisions = db._execute_query(query, (product_id, limit))
        
        # Convertir les dates en string pour JSON
        for decision in decisions:
            for key in ['timestamp', 'created_at']:
                if key in decision and decision[key] and isinstance(decision[key], datetime):
                    decision[key] = decision[key].isoformat()
        
        return {
            "product_id": product_id,
            "count": len(decisions),
            "decisions": decisions,
            "limit": limit,
            "retrieved_at": datetime.now().isoformat()  # ← ISO format
        }
        
    except Exception as e:
        logger.error(f"Error getting decision history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/feedback")
async def submit_feedback(
    decision_id: str,
    actual_price: Optional[float] = None,
    actual_sales: Optional[int] = None,
    customer_feedback: Optional[str] = None,
    db: DatabaseManager = Depends(get_db)
):
    """
    Soumettre du feedback pour une décision
    """
    try:
        # Récupérer décision
        query = "SELECT * FROM pricing_decisions WHERE decision_id = %s"
        decision = db._execute_query(query, (decision_id,), fetch_one=True)
        
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")
        
        # Calculer déviation
        deviation = None
        if actual_price and decision.get('recommended_price'):
            deviation = abs(actual_price - decision['recommended_price']) / decision['recommended_price']
        
        # Enregistrer feedback
        feedback_data = {
            "decision_id": decision_id,
            "actual_price": actual_price,
            "actual_sales": actual_sales,
            "customer_feedback": customer_feedback,
            "price_deviation": deviation,
            "submitted_at": datetime.now().isoformat()  # ← ISO format
        }
        
        try:
            # Log dans monitoring
            db.log_system_event(
                component="feedback",
                level="INFO",
                message=f"Feedback received for decision {decision_id}",
                metrics=feedback_data
            )
        except:
            pass  # Ignorer si la méthode n'existe pas
        
        return {
            "status": "feedback_received",
            "decision_id": decision_id,
            "feedback": feedback_data,
            "message": "Feedback will be used for model improvement",
            "received_at": datetime.now().isoformat()  # ← ISO format
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# IMPORT DU PRICING ROUTER - CORRIGÉ
# ============================================================

# Import des routes additionnelles depuis pricing_router.py
pricing_router_imported = False

# Tentative 1: Import relatif (même dossier que fastapi_app.py)
try:
    from .pricing_router import router as pricing_router_v1
    app.include_router(pricing_router_v1)
    logger.info("✅ Pricing router importé avec succès (import relatif)")
    pricing_router_imported = True
except ImportError as e:
    logger.warning(f"⚠️ Tentative 1 échouée (import relatif): {e}")
except Exception as e:
    logger.error(f"❌ Erreur tentative 1: {e}")

# Tentative 2: Import sans point (si exécuté directement)
if not pricing_router_imported:
    try:
        from pricing_router import router as pricing_router_v1
        app.include_router(pricing_router_v1)
        logger.info("✅ Pricing router importé avec succès (import direct)")
        pricing_router_imported = True
    except ImportError as e:
        logger.warning(f"⚠️ Tentative 2 échouée (import direct): {e}")
    except Exception as e:
        logger.error(f"❌ Erreur tentative 2: {e}")

# Tentative 3: Chemin complet (depuis la racine)
if not pricing_router_imported:
    try:
        from deployment.api.pricing_router import router as pricing_router_v1
        app.include_router(pricing_router_v1)
        logger.info("✅ Pricing router importé avec succès (chemin complet)")
        pricing_router_imported = True
    except ImportError as e:
        logger.warning(f"⚠️ Tentative 3 échouée (chemin complet): {e}")
    except Exception as e:
        logger.error(f"❌ Erreur tentative 3: {e}")

# Tentative 4: Import depuis pricing_endpoints (si votre fichier s'appelle comme ça)
if not pricing_router_imported:
    try:
        from .pricing_endpoints import router as pricing_router_v1
        app.include_router(pricing_router_v1)
        logger.info("✅ Pricing router importé depuis pricing_endpoints")
        pricing_router_imported = True
    except ImportError as e:
        logger.warning(f"⚠️ Tentative 4 échouée (pricing_endpoints): {e}")
    except Exception as e:
        logger.error(f"❌ Erreur tentative 4: {e}")

# Dernière tentative: Créer un router minimal si tout échoue
if not pricing_router_imported:
    logger.warning("⚠️ Aucun router pricing trouvé, création d'un router minimal")
    from fastapi import APIRouter
    fallback_router = APIRouter(prefix="/api/v1", tags=["pricing"])
    
    @fallback_router.get("/pricing/{product_id}")
    async def fallback_pricing(product_id: str):
        return {
            "message": "Fallback pricing endpoint",
            "product_id": product_id,
            "recommended_price": 0,
            "status": "fallback_mode"
        }
    
    app.include_router(fallback_router)
    logger.info("✅ Router fallback créé")

# Gestion erreurs
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),  # ← ISO format
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),  # ← ISO format
            "path": str(request.url.path)
        }
    )

# Point d'entrée
if __name__ == "__main__":
    uvicorn.run(
        "deployment.api.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Classe PricingAPI pour compatibilité d'import
class PricingAPI:
    """
    Interface principale pour l'API de pricing
    Wrapper autour de l'application FastAPI
    """
    
    def __init__(self):
        self.app = app
        self.db_manager = None
        self.orchestrator = None
    
    def initialize(self, db_manager=None, orchestrator=None):
        """Initialiser l'API avec les dépendances"""
        self.db_manager = db_manager
        self.orchestrator = orchestrator
        return self
    
    def get_app(self):
        """Retourner l'application FastAPI"""
        return self.app
    
    def get_health_status(self):
        """Vérifier l'état de santé de l'API"""
        try:
            # Simulation d'un check de santé
            return {
                "status": "healthy",
                "components": {
                    "database": "connected" if self.db_manager else "not_initialized",
                    "orchestrator": "ready" if self.orchestrator else "not_initialized",
                    "api": "running"
                },
                "timestamp": datetime.now().isoformat()  # ← ISO format
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()  # ← ISO format
            }