# Documentation Technique - EcoTrain

## Vue d'ensemble

EcoTrain est une plateforme SaaS développée pour réduire la consommation énergétique excessive de l'intelligence artificielle. Le code fournit un module EcoMonitor qui illustre les capacités de base de la plateforme.

## Architecture

### Structure du projet

```
ecotrain/
├── ecotrain/               # Package principal
│   ├── __init__.py        # Point d'entrée du package
│   ├── ecomonitor.py      # Module de surveillance énergétique
│   └── optimizer.py       # Module d'optimisation de modèles
├── examples/              # Exemples d'utilisation
│   └── basic_example.py   # Démonstration complète
├── tests/                 # Tests unitaires
│   └── test_ecotrain.py   # Tests de base
├── requirements.txt       # Dépendances Python
├── setup.py              # Configuration d'installation
├── LICENSE               # Licence MIT
└── README.md             # Documentation utilisateur
```

## Modules

### 1. EcoMonitor (`ecomonitor.py`)

Le module EcoMonitor mesure la consommation énergétique des modèles PyTorch.

#### Fonctionnalités principales:

- **Mesure temporelle**: Utilise `time.time()` pour mesurer la durée d'exécution
- **Estimation CPU**: Utilise `psutil.cpu_percent()` pour mesurer l'utilisation CPU
- **Estimation GPU**: Suppose une utilisation à 80% si GPU disponible
- **Calcul énergétique**: Combine les mesures CPU/GPU pour estimer la consommation
- **Estimation CO2**: Utilise une intensité carbone de 475g CO2/kWh

#### Méthodes:

```python
monitor = EcoMonitor()
monitor.start()                     # Démarrer le suivi
metrics = monitor.stop()            # Arrêter et obtenir les métriques
result, metrics = monitor.track_training(fn)  # Tracer une fonction
monitor.print_report(metrics)       # Afficher un rapport formaté
```

#### Métriques retournées:

- `duration_seconds`: Durée d'exécution
- `cpu_power_watts`: Puissance CPU estimée
- `gpu_power_watts`: Puissance GPU estimée
- `total_power_watts`: Puissance totale
- `energy_wh`: Énergie en Watt-heures
- `energy_kwh`: Énergie en kilowatt-heures
- `co2_grams`: Émissions CO2 estimées

### 2. ModelOptimizer (`optimizer.py`)

Le module ModelOptimizer analyse et optimise les modèles PyTorch.

#### Fonctionnalités:

- **Analyse de modèle**: Compte les paramètres et estime la taille
- **Suggestions d'optimisation**: Propose quantization, mixed precision, etc.
- **Quantization dynamique**: Applique `torch.quantization.quantize_dynamic()`
- **Comparaison**: Compare modèles original vs optimisé

#### Méthodes:

```python
optimizer = ModelOptimizer()
analysis = optimizer.analyze_model(model)           # Analyser
optimizer.print_optimization_report(analysis)       # Afficher suggestions
quantized = optimizer.quantize_model(model)         # Quantizer
comparison = optimizer.compare_models(m1, m2)       # Comparer
```

## Méthodologie d'estimation énergétique

### Pourquoi un proxy temporel?

En l'absence de bibliothèques spécialisées comme CodeCarbon (non pré-installées), EcoTrain utilise une approche par estimation basée sur:

1. **Durée d'exécution**: Mesure précise via `time.time()`
2. **Utilisation CPU**: Mesure via `psutil.cpu_percent()`
3. **Hypothèses GPU**: Estimation à 80% d'utilisation pendant l'entraînement
4. **TDP typiques**: Utilise des valeurs standards (65W CPU, 250W GPU)

### Formules utilisées:

```
CPU Power (W) = (CPU Usage % / 100) × TDP_CPU
GPU Power (W) = 0.8 × TDP_GPU  (si GPU disponible)
Total Power (W) = CPU Power + GPU Power
Energy (Wh) = Total Power × Duration / 3600
CO2 (g) = Energy (kWh) × 475
```

### Limitations:

- Les TDP sont des approximations
- L'utilisation GPU est estimée, pas mesurée
- L'intensité carbone est une moyenne mondiale
- Pas de prise en compte de la mémoire, du stockage, etc.

### Avantages:

- Fonctionne sans dépendances externes
- Rapide et léger
- Fournit des ordres de grandeur utiles pour comparer des approches
- Éducatif sur l'impact énergétique

## Optimisations implémentées

### 1. Quantization dynamique

Réduit la précision des poids de FP32 à INT8:

```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM, nn.GRU},
    dtype=torch.qint8
)
```

**Avantages:**
- Réduit la taille du modèle de ~75%
- Accélère l'inférence
- Réduit l'utilisation mémoire

### 2. Mixed Precision (suggestion)

Utilise FP16 au lieu de FP32 pendant l'entraînement:

```python
with torch.cuda.amp.autocast():
    output = model(input)
```

**Avantages:**
- Réduit la consommation mémoire GPU de ~50%
- Accélère l'entraînement sur GPU moderne
- Maintient la précision du modèle

### 3. Optimisation des batches (suggestion)

Ajuster `batch_size` pour maximiser l'utilisation GPU sans dépasser la mémoire.

## Utilisation

### Installation

```bash
pip install -r requirements.txt
```

Ou avec setup.py:

```bash
pip install -e .
```

### Exemple basique

```python
from ecotrain import EcoMonitor, ModelOptimizer
import torch.nn as nn

# Créer un modèle
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Surveiller l'entraînement
monitor = EcoMonitor()
result, metrics = monitor.track_training(train_function)
monitor.print_report(metrics)

# Optimiser le modèle
optimizer = ModelOptimizer()
analysis = optimizer.analyze_model(model)
optimizer.print_optimization_report(analysis)

model.eval()
quantized_model = optimizer.quantize_model(model)
```

### Exécuter l'exemple complet

```bash
python examples/basic_example.py
```

## Tests

Exécuter les tests:

```bash
python tests/test_ecotrain.py
```

Tests inclus:
- Test du démarrage/arrêt du moniteur
- Test du suivi de fonction d'entraînement
- Test de l'analyse de modèle
- Test de la quantization
- Test de la comparaison de modèles

## Extensions futures

- Intégration de CodeCarbon pour des mesures réelles
- Support de TensorFlow et JAX
- Dashboard web pour visualisation
- Support du pruning de modèles
- Profiling plus détaillé (couche par couche)
- API REST pour intégration SaaS
- Base de données pour historique des métriques

## Références

- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [CodeCarbon](https://codecarbon.io/)
- [Green AI](https://arxiv.org/abs/1907.10597)

## Licence

MIT License - voir LICENSE pour plus de détails.
