# EcoTrain - Plateforme SaaS de Réduction de Consommation Énergétique IA

EcoTrain est une plateforme SaaS conçue pour réduire la consommation énergétique excessive de l'intelligence artificielle. Elle fournit des outils pour mesurer, analyser et optimiser l'empreinte énergétique des modèles de machine learning.

## 🌱 Fonctionnalités

### EcoMonitor
Module de surveillance de la consommation énergétique qui:
- Mesure la consommation d'énergie pendant l'entraînement des modèles
- Utilise un proxy temporel et des estimations basées sur l'utilisation CPU/GPU
- Estime les émissions de CO2 associées
- Fournit des rapports détaillés sur la consommation énergétique

### ModelOptimizer
Module d'optimisation qui:
- Analyse les modèles PyTorch
- Suggère des optimisations (quantization, mixed precision, etc.)
- Applique la quantization dynamique pour réduire la taille des modèles
- Compare les performances avant/après optimisation

## 📦 Installation

```bash
# Cloner le repository
git clone https://github.com/ONDM-code/ecotrain.git
cd ecotrain

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### Exemple basique

```python
import torch
import torch.nn as nn
from ecotrain import EcoMonitor, ModelOptimizer

# Créer un modèle
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Initialiser le moniteur énergétique
monitor = EcoMonitor()

# Fonction d'entraînement
def train():
    # Votre code d'entraînement ici
    pass

# Tracer l'entraînement
result, metrics = monitor.track_training(train)

# Afficher le rapport
monitor.print_report(metrics)

# Analyser et optimiser le modèle
optimizer = ModelOptimizer()
analysis = optimizer.analyze_model(model)
optimizer.print_optimization_report(analysis)

# Appliquer la quantization
model.eval()
quantized_model = optimizer.quantize_model(model)
```

### Exécuter l'exemple complet

```bash
python examples/basic_example.py
```

Cet exemple démontre:
1. La création d'un réseau de neurones simple
2. L'entraînement avec suivi énergétique
3. L'analyse et les suggestions d'optimisation
4. L'application de la quantization
5. La comparaison des modèles avant/après optimisation

## 📊 Métriques suivies

- **Durée d'entraînement** (secondes)
- **Puissance consommée** (Watts) - CPU et GPU
- **Énergie totale** (Wh et kWh)
- **Émissions CO2** (grammes) - basées sur l'intensité carbone moyenne mondiale

## 🛠️ Architecture

```
ecotrain/
├── ecotrain/
│   ├── __init__.py          # Point d'entrée du package
│   ├── ecomonitor.py        # Module de surveillance énergétique
│   └── optimizer.py         # Module d'optimisation de modèles
├── examples/
│   └── basic_example.py     # Exemple d'utilisation complet
├── requirements.txt         # Dépendances Python
└── README.md               # Documentation
```

## 🔬 Méthodologie

### Estimation énergétique

En l'absence de bibliothèques spécialisées comme CodeCarbon, EcoTrain utilise:
- Un **proxy temporel**: mesure de la durée d'exécution
- **Estimation CPU**: basée sur le pourcentage d'utilisation et le TDP typique
- **Estimation GPU**: basée sur un taux d'utilisation estimé pendant l'entraînement
- **Calcul CO2**: utilise une intensité carbone moyenne de 475g CO2/kWh

Ces estimations fournissent une approximation raisonnable pour comparer différentes approches d'entraînement.

### Optimisations proposées

1. **Quantization dynamique**: Réduit la taille du modèle de ~75%
2. **Mixed Precision Training (FP16)**: Réduit la consommation mémoire GPU de ~50%
3. **Optimisation des batch sizes**: Améliore l'utilisation des ressources

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou soumettre une pull request.

## 📝 Licence

Ce projet est open source et disponible sous licence MIT.

## 🎯 Objectifs futurs

- [ ] Intégration de CodeCarbon pour des mesures plus précises
- [ ] Support pour TensorFlow et JAX
- [ ] Dashboard web pour visualiser les métriques
- [ ] Comparaison de différentes stratégies d'optimisation
- [ ] Support du pruning de modèles
- [ ] Intégration CI/CD pour le suivi automatique

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.
