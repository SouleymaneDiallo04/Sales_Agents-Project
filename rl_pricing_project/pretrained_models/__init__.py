"""
Package pour la gestion des modèles pré-entraînés et fine-tuning
"""

from .downloader import PretrainedModelDownloader
from .model_adapter import ModelAdapter

__all__ = [
    "PretrainedModelDownloader",
    "ModelAdapter"
]

__version__ = "1.0.0"