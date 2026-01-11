"""
Middleware de sécurité pour l'API Sales Agents
Inclut logging, rate limiting basique, et validation des requêtes
"""

import time
import logging
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import json

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour logger toutes les requêtes"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log de la requête entrante
        logger.info(f"Requête entrante: {request.method} {request.url}")

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log de la réponse
            logger.info(".2f")

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(".2f")
            raise

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware pour ajouter des headers de sécurité"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Headers de sécurité
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # CORS basique (à adapter selon les besoins)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting basique (en mémoire)"""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # En production, utiliser Redis

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        # Nettoyer les anciennes requêtes
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < self.window_seconds
            ]
        else:
            self.requests[client_ip] = []

        # Vérifier le rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit dépassé pour {client_ip}")
            return Response(
                content=json.dumps({"error": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json"
            )

        # Ajouter la requête actuelle
        self.requests[client_ip].append(current_time)

        response = await call_next(request)
        return response

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware pour valider les requêtes"""

    async def dispatch(self, request: Request, call_next):
        # Vérifier la taille du body
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            if len(body) > 10 * 1024 * 1024:  # 10MB max
                return Response(
                    content=json.dumps({"error": "Request body too large"}),
                    status_code=413,
                    media_type="application/json"
                )

        # Vérifier les headers requis
        if request.method != "OPTIONS":
            content_type = request.headers.get("content-type", "")
            if request.method in ["POST", "PUT", "PATCH"] and not content_type:
                logger.warning(f"Content-Type manquant pour {request.method} {request.url}")

        response = await call_next(request)
        return response