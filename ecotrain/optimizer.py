"""
ModelOptimizer - Module pour optimiser les modèles et réduire leur consommation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class ModelOptimizer:
    """
    Optimiseur de modèles pour réduire la consommation énergétique.
    
    Propose des optimisations comme la quantization, le pruning, etc.
    """
    
    @staticmethod
    def analyze_model(model: nn.Module) -> Dict[str, Any]:
        """
        Analyse un modèle et suggère des optimisations.
        
        Args:
            model: Modèle PyTorch à analyser
            
        Returns:
            Dictionnaire contenant les analyses et suggestions
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimation de la taille mémoire (en MB)
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        
        suggestions = []
        
        # Suggestion 1: Quantization
        if param_size_mb > 10:  # Si le modèle est > 10MB
            suggestions.append({
                "type": "quantization",
                "description": "Quantization dynamique recommandée",
                "potential_savings": "Réduction de 75% de la taille du modèle",
                "implementation": "Utilisez torch.quantization.quantize_dynamic()"
            })
        
        # Suggestion 2: Mixed Precision Training
        suggestions.append({
            "type": "mixed_precision",
            "description": "Entraînement en précision mixte (FP16)",
            "potential_savings": "Réduction de ~50% de la mémoire GPU et accélération",
            "implementation": "Utilisez torch.cuda.amp.autocast()"
        })
        
        # Suggestion 3: Réduction du batch size si pertinent
        suggestions.append({
            "type": "batch_optimization",
            "description": "Optimisation de la taille des batchs",
            "potential_savings": "Meilleure utilisation de la mémoire",
            "implementation": "Ajustez batch_size pour maximiser l'utilisation GPU"
        })
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": param_size_mb,
            "suggestions": suggestions
        }
    
    @staticmethod
    def quantize_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
        """
        Applique la quantization dynamique à un modèle.
        
        Args:
            model: Modèle à quantizer
            backend: Backend de quantization ('fbgemm' pour CPU, 'qnnpack' pour mobile)
            
        Returns:
            Modèle quantizé
        """
        # Quantization dynamique des couches Linear et LSTM
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        return quantized_model
    
    @staticmethod
    def compare_models(original_model: nn.Module, optimized_model: nn.Module) -> Dict[str, Any]:
        """
        Compare deux modèles (original vs optimisé).
        
        Args:
            original_model: Modèle original
            optimized_model: Modèle optimisé
            
        Returns:
            Dictionnaire de comparaison
        """
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024 ** 2)
        optimized_size = sum(p.numel() * p.element_size() for p in optimized_model.parameters()) / (1024 ** 2)
        
        size_reduction_percent = ((original_size - optimized_size) / original_size) * 100
        
        return {
            "original_size_mb": original_size,
            "optimized_size_mb": optimized_size,
            "size_reduction_mb": original_size - optimized_size,
            "size_reduction_percent": size_reduction_percent
        }
    
    @staticmethod
    def print_optimization_report(analysis: Dict[str, Any]):
        """
        Affiche un rapport d'analyse et de suggestions d'optimisation.
        
        Args:
            analysis: Résultat de analyze_model()
        """
        print("\n" + "="*60)
        print("RAPPORT D'OPTIMISATION - ModelOptimizer")
        print("="*60)
        print(f"Nombre total de paramètres: {analysis['total_params']:,}")
        print(f"Paramètres entraînables: {analysis['trainable_params']:,}")
        print(f"Taille du modèle: {analysis['model_size_mb']:.2f} MB")
        
        print(f"\n{'Suggestions d\'optimisation:'}")
        for i, suggestion in enumerate(analysis['suggestions'], 1):
            print(f"\n{i}. {suggestion['type'].upper()}")
            print(f"   Description: {suggestion['description']}")
            print(f"   Économies potentielles: {suggestion['potential_savings']}")
            print(f"   Implémentation: {suggestion['implementation']}")
        print("="*60 + "\n")
