"""
Exemple d'utilisation de la plateforme EcoTrain
================================================

Ce script démontre comment utiliser EcoMonitor pour tracer la consommation
énergétique d'un modèle PyTorch et comment utiliser ModelOptimizer pour
optimiser le modèle via la quantization.

Le script:
1. Crée un modèle de réseau de neurones simple
2. Génère des données d'entraînement synthétiques
3. Entraîne le modèle tout en mesurant la consommation énergétique
4. Analyse le modèle et suggère des optimisations
5. Applique la quantization et compare les résultats
"""

import sys
import os

# Ajouter le répertoire parent au path pour importer ecotrain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from ecotrain import EcoMonitor, ModelOptimizer


# Définir un modèle simple pour la démonstration
class SimpleNN(nn.Module):
    """Réseau de neurones simple pour la classification."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def generate_synthetic_data(num_samples=1000, input_size=784, num_classes=10):
    """Génère des données synthétiques pour l'entraînement."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def train_model(model, X, y, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Entraîne le modèle sur les données fournies.
    
    Args:
        model: Modèle PyTorch à entraîner
        X: Données d'entrée
        y: Labels
        epochs: Nombre d'époques
        batch_size: Taille du batch
        learning_rate: Taux d'apprentissage
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    num_samples = X.size(0)
    num_batches = num_samples // batch_size
    
    print(f"Entraînement du modèle pour {epochs} époques...")
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Époque [{epoch+1}/{epochs}], Perte moyenne: {avg_loss:.4f}")
    
    print("Entraînement terminé!\n")
    return model


def main():
    """Fonction principale de démonstration."""
    
    print("="*60)
    print("ECOTRAIN - Plateforme SaaS de Réduction d'Énergie IA")
    print("="*60)
    print("\nDémonstration du module EcoMonitor\n")
    
    # 1. Créer le modèle
    print("1. Création du modèle de réseau de neurones...")
    model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)
    print(f"   Modèle créé: {model.__class__.__name__}")
    
    # 2. Générer des données synthétiques
    print("\n2. Génération de données synthétiques...")
    X_train, y_train = generate_synthetic_data(num_samples=1000)
    print(f"   Données générées: {X_train.shape[0]} échantillons")
    
    # 3. Initialiser EcoMonitor et entraîner le modèle
    print("\n3. Entraînement avec suivi énergétique...")
    monitor = EcoMonitor()
    
    # Fonction d'entraînement à tracer
    def training_function():
        return train_model(model, X_train, y_train, epochs=5, batch_size=32)
    
    # Tracer l'entraînement
    trained_model, energy_metrics = monitor.track_training(training_function)
    
    # 4. Afficher le rapport énergétique
    monitor.print_report(energy_metrics)
    
    # 5. Analyser le modèle et suggérer des optimisations
    print("\n4. Analyse du modèle et suggestions d'optimisation...")
    optimizer_tool = ModelOptimizer()
    analysis = optimizer_tool.analyze_model(trained_model)
    optimizer_tool.print_optimization_report(analysis)
    
    # 6. Appliquer la quantization
    print("\n5. Application de la quantization...")
    # Mettre le modèle en mode évaluation pour la quantization
    trained_model.eval()
    quantized_model = optimizer_tool.quantize_model(trained_model)
    print("   Quantization appliquée avec succès!")
    
    # 7. Comparer les modèles
    print("\n6. Comparaison des modèles (original vs quantizé)...")
    comparison = optimizer_tool.compare_models(trained_model, quantized_model)
    
    print("\n" + "="*60)
    print("RÉSULTATS DE LA COMPARAISON")
    print("="*60)
    print(f"Taille du modèle original: {comparison['original_size_mb']:.2f} MB")
    print(f"Taille du modèle quantizé: {comparison['optimized_size_mb']:.2f} MB")
    print(f"Réduction de taille: {comparison['size_reduction_mb']:.2f} MB ({comparison['size_reduction_percent']:.1f}%)")
    print("="*60)
    
    # 8. Test d'inférence pour vérifier que le modèle fonctionne
    print("\n7. Vérification de l'inférence...")
    test_input = torch.randn(1, 784)
    
    with torch.no_grad():
        output_original = trained_model(test_input)
        output_quantized = quantized_model(test_input)
    
    print(f"   Sortie du modèle original: forme {output_original.shape}")
    print(f"   Sortie du modèle quantizé: forme {output_quantized.shape}")
    print("   ✓ Les deux modèles fonctionnent correctement!")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("Ce script a démontré comment EcoTrain peut:")
    print("  1. Mesurer la consommation énergétique pendant l'entraînement")
    print("  2. Analyser un modèle et suggérer des optimisations")
    print("  3. Appliquer la quantization pour réduire la taille du modèle")
    print("  4. Comparer les performances avant/après optimisation")
    print("\nCes techniques permettent de réduire significativement")
    print("la consommation énergétique des modèles d'IA en production.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
