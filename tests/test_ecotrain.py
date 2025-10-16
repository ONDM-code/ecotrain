"""
Tests pour le module EcoMonitor
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import torch
import torch.nn as nn
from ecotrain import EcoMonitor, ModelOptimizer


def test_ecomonitor_basic():
    """Test basique du EcoMonitor."""
    monitor = EcoMonitor()
    
    # Test start/stop
    monitor.start()
    time.sleep(0.5)  # Simuler une opération
    metrics = monitor.stop()
    
    # Vérifier que les métriques sont présentes
    assert 'duration_seconds' in metrics
    assert 'cpu_power_watts' in metrics
    assert 'total_power_watts' in metrics
    assert 'energy_wh' in metrics
    assert 'co2_grams' in metrics
    
    # Vérifier que la durée est raisonnable
    assert metrics['duration_seconds'] >= 0.5
    assert metrics['duration_seconds'] < 1.0
    
    print("✓ test_ecomonitor_basic passed")


def test_ecomonitor_track_training():
    """Test de la méthode track_training."""
    monitor = EcoMonitor()
    
    # Fonction de test simple
    def dummy_training():
        time.sleep(0.3)
        return "completed"
    
    result, metrics = monitor.track_training(dummy_training)
    
    # Vérifier le résultat
    assert result == "completed"
    assert 'duration_seconds' in metrics
    assert metrics['duration_seconds'] >= 0.3
    
    print("✓ test_ecomonitor_track_training passed")


def test_model_optimizer_analyze():
    """Test de l'analyse de modèle."""
    # Créer un modèle simple
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    optimizer = ModelOptimizer()
    analysis = optimizer.analyze_model(model)
    
    # Vérifier les résultats
    assert 'total_params' in analysis
    assert 'trainable_params' in analysis
    assert 'model_size_mb' in analysis
    assert 'suggestions' in analysis
    
    assert analysis['total_params'] > 0
    assert len(analysis['suggestions']) > 0
    
    print("✓ test_model_optimizer_analyze passed")


def test_model_optimizer_quantize():
    """Test de la quantization de modèle."""
    # Créer un modèle simple
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    model.eval()
    
    optimizer = ModelOptimizer()
    quantized_model = optimizer.quantize_model(model)
    
    # Vérifier que le modèle fonctionne après quantization
    test_input = torch.randn(1, 100)
    with torch.no_grad():
        output = quantized_model(test_input)
    
    assert output.shape == (1, 10)
    
    print("✓ test_model_optimizer_quantize passed")


def test_model_comparison():
    """Test de la comparaison de modèles."""
    # Créer deux modèles
    model1 = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    model1.eval()
    
    model2 = nn.Sequential(
        nn.Linear(100, 25),
        nn.ReLU(),
        nn.Linear(25, 10)
    )
    model2.eval()
    
    optimizer = ModelOptimizer()
    comparison = optimizer.compare_models(model1, model2)
    
    # Vérifier les résultats
    assert 'original_size_mb' in comparison
    assert 'optimized_size_mb' in comparison
    assert 'size_reduction_mb' in comparison
    assert 'size_reduction_percent' in comparison
    
    # Le modèle 2 devrait être plus petit
    assert comparison['optimized_size_mb'] < comparison['original_size_mb']
    
    print("✓ test_model_comparison passed")


if __name__ == "__main__":
    print("Exécution des tests EcoTrain...\n")
    
    test_ecomonitor_basic()
    test_ecomonitor_track_training()
    test_model_optimizer_analyze()
    test_model_optimizer_quantize()
    test_model_comparison()
    
    print("\n" + "="*60)
    print("Tous les tests ont réussi! ✓")
    print("="*60)
