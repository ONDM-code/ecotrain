"""
EcoTrain - Plateforme SaaS pour réduire la consommation énergétique de l'IA
"""

__version__ = "0.1.0"

from .ecomonitor import EcoMonitor
from .optimizer import ModelOptimizer

__all__ = ["EcoMonitor", "ModelOptimizer"]
