class PricingRequest(BaseModel):
    """Requête pour une décision de pricing"""
    product_id: str = "PROD_001"
    current_stock: float
    current_price: float
    competitor_prices: List[float]
    demand_history: List[float]
    timestamp: Optional[str] = None

class PricingResponse(BaseModel):
    """Réponse avec recommandation de prix"""
    recommended_price: float
    price_change: float
    action: str
    confidence: float
    explanation: str
    model_info: Dict[str, Any]

class PricingAPI:
    """API FastAPI pour le pricing RL"""
    
    def __init__(self, model_path: str = "models/final_ppo_pricing.zip"):
        self.app = FastAPI(
            title="RL Pricing API",
            description="API de pricing dynamique par Reinforcement Learning",
            version="1.0.0"
        )
        
        # Configuration CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Chargement du modèle
        self.model = None
        self.env = None
        self.model_path = model_path
        self.load_model()
        
        # Routes
        self.setup_routes()
    
    def load_model(self):
        """Charge le modèle RL entraîné"""
        try:
            # Détecter l'algorithme à partir du nom du fichier
            if "ppo" in self.model_path:
                algo = PPO
            elif "dqn" in self.model_path:
                algo = DQN
            elif "a2c" in self.model_path:
                algo = A2C
            else:
                algo = PPO  # Par défaut
            
            # Créer un environnement minimal
            self.env = DummyVecEnv([lambda: EcommercePricingEnv()])
            
            # Charger le modèle
            self.model = algo.load(self.model_path, env=self.env)
            print(f"✅ Modèle chargé depuis: {self.model_path}")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            self.model = None
    
    def setup_routes(self):
        """Configure les routes de l'API"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "RL Pricing API",
                "status": "running",
                "model_loaded": self.model is not None
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "timestamp": np.datetime64('now').astype(str)
            }
        
        @self.app.post("/api/v1/pricing/decision", response_model=PricingResponse)
        async def get_pricing_decision(request: PricingRequest):
            """Endpoint principal pour obtenir une décision de pricing"""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                # Préparer l'état pour le modèle
                state = self.prepare_state(request)
                
                # Obtenir la prédiction du modèle
                action, _ = self.model.predict(state, deterministic=True)
                
                # Interpréter l'action
                recommended_price, price_change, action_str = self.interpret_action(
                    action[0], request.current_price
                )
                
                # Calculer la confiance (simplifié)
                confidence = self.calculate_confidence(state, action)
                
                # Générer une explication
                explanation = self.generate_explanation(
                    action_str, recommended_price, price_change, confidence
                )
                
                return PricingResponse(
                    recommended_price=float(recommended_price),
                    price_change=float(price_change),
                    action=action_str,
                    confidence=float(confidence),
                    explanation=explanation,
                    model_info={
                        "algorithm": self.model.__class__.__name__,
                        "model_path": self.model_path
                    }
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/pricing/models")
        async def list_available_models():
            """Liste les modèles disponibles"""
            models_dir = "models"
            available_models = []
            
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.endswith(".zip"):
                        model_info = {
                            "name": file,
                            "path": os.path.join(models_dir, file),
                            "size": os.path.getsize(os.path.join(models_dir, file))
                        }
                        available_models.append(model_info)
            
            return {
                "available_models": available_models,
                "current_model": self.model_path
            }
    
    def prepare_state(self, request: PricingRequest) -> np.ndarray:
        """Prépare l'état à partir de la requête"""
        # Normaliser les données
        stock_ratio = request.current_stock / 100.0  # Normalisé par stock max
        
        # Prix normalisé entre min et max
        min_price = 495.0
        max_price = 900.0
        price_range = max_price - min_price
        price_ratio = (request.current_price - min_price) / price_range
        
        # Concurrents normalisés
        comp_ratios = []
        for comp_price in request.competitor_prices[:2]:  # Prendre les 2 premiers
            comp_ratios.append(comp_price / request.current_price)
        
        # Compléter si moins de 2 concurrents
        while len(comp_ratios) < 2:
            comp_ratios.append(1.0)  # Prix équivalent
        
        # Demande normalisée
        if len(request.demand_history) > 0:
            demand_trend = np.mean(request.demand_history[-7:]) / 20.0
        else:
            demand_trend = 0.5
        
        # Créer l'état normalisé
        state = np.array([
            stock_ratio,
            price_ratio,
            comp_ratios[0],
            comp_ratios[1],
            demand_trend,
            0.5,  # day_of_week (placeholder)
            0.5,  # seasonality (placeholder)
            1.0 - stock_ratio,  # stock_risk
            0.2   # market_volatility (placeholder)
        ], dtype=np.float32).reshape(1, -1)
        
        return state
    
    def interpret_action(self, action: int, current_price: float) -> tuple:
        """Interprète l'action en prix recommandé"""
        action_map = {
            0: ("-10%", -0.10),
            1: ("-5%", -0.05),
            2: ("0%", 0.00),
            3: ("+5%", 0.05),
            4: ("+10%", 0.10)
        }
        
        action_str, change_percent = action_map.get(action, ("0%", 0.00))
        recommended_price = current_price * (1 + change_percent)
        
        # Assurer les contraintes business
        min_price = 495.0
        max_price = 900.0
        recommended_price = max(min_price, min(recommended_price, max_price))
        
        price_change = recommended_price - current_price
        
        return recommended_price, price_change, action_str
    
    def calculate_confidence(self, state: np.ndarray, action: np.ndarray) -> float:
        """Calcule un score de confiance pour la prédiction"""
        # Simplifié: basé sur la stabilité du stock et de la demande
        stock_ratio = state[0][0]
        demand_trend = state[0][4]
        
        # Plus le stock est stable et la demande prévisible, plus la confiance est élevée
        confidence = 0.5 + (stock_ratio * 0.3) + (abs(demand_trend - 0.5) * 0.2)
        
        return min(max(confidence, 0.0), 1.0)
    
    def generate_explanation(self, action: str, price: float, 
                           change: float, confidence: float) -> str:
        """Génère une explication lisible de la décision"""
        explanations = {
            "-10%": "Réduction agressive pour stimuler la demande et réduire le stock",
            "-5%": "Réduction modérée pour rester compétitif et maintenir le flux de ventes",
            "0%": "Maintien du prix actuel pour préserver la marge et la perception de valeur",
            "+5%": "Augmentation modérée pour capturer plus de valeur avec demande stable",
            "+10%": "Augmentation significative pour maximiser la marge pendant les pics de demande"
        }
        
        base_explanation = explanations.get(action, "Ajustement de prix standard")
        
        if confidence > 0.7:
            confidence_text = "haute confiance"
        elif confidence > 0.4:
            confidence_text = "confiance moyenne"
        else:
            confidence_text = "confiance faible"
        
        change_text = f"augmentation de {abs(change):.2f}€" if change > 0 else f"réduction de {abs(change):.2f}€" if change < 0 else "aucun changement"
        
        return f"{base_explanation}. Recommandation avec {confidence_text}: {change_text} vers {price:.2f}€."

def run_server(model_path: str = "models/final_ppo_pricing.zip", 
               host: str = "127.0.0.1", 
               port: int = 8000):
    """Lance le serveur API"""
    
    api = PricingAPI(model_path)
    
    print(f"🚀 Démarrage de l'API RL Pricing sur http://{host}:{port}")
    print(f"📊 Modèle utilisé: {model_path}")
    print(f"📚 Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        api.app,
        host=host,
        port=port,
        reload=False
    )

def main():
    parser = argparse.ArgumentParser(description="Déploiement de l'API Pricing RL")
    parser.add_argument("--model", type=str, default="models/final_ppo_pricing.zip",
                       help="Chemin vers le modèle à déployer")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Adresse d'écoute")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port d'écoute")
    
    args = parser.parse_args()
    
    # Vérifier que le modèle existe
    if not os.path.exists(args.model):
        print(f"❌ Modèle non trouvé: {args.model}")
        print("Veuillez d'abord entraîner un modèle avec: python scripts/train.py")
        return
    
    run_server(args.model, args.host, args.port)

if __name__ == "__main__":
    main()