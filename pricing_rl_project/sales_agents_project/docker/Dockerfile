# Dockerfile pour Sales Agents Project

FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p logs/training_logs logs/pricing_decisions data/database

# Variables d'environnement
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Port pour l'API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande par défaut
CMD ["python", "scripts/run_api.py"]