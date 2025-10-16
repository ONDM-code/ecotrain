# Guide de démarrage rapide - EcoTrain

## Installation rapide

```bash
# Cloner le repository
git clone https://github.com/ONDM-code/ecotrain.git
cd ecotrain

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation rapide

### Option 1: Utiliser le CLI

```bash
# Exécuter la démonstration complète
python cli.py demo

# Analyser un modèle existant
python cli.py analyze mon_modele.pth
```

### Option 2: Utiliser l'exemple Python

```bash
python examples/basic_example.py
```

### Option 3: Intégrer dans votre code

```python
from ecotrain import EcoMonitor, ModelOptimizer

# 1. Surveiller l'entraînement
monitor = EcoMonitor()

def ma_fonction_entrainement():
    # Votre code d'entraînement
    pass

result, metrics = monitor.track_training(ma_fonction_entrainement)
monitor.print_report(metrics)

# 2. Optimiser le modèle
optimizer = ModelOptimizer()
analysis = optimizer.analyze_model(mon_modele)
optimizer.print_optimization_report(analysis)

# 3. Appliquer la quantization
mon_modele.eval()
modele_optimise = optimizer.quantize_model(mon_modele)
```

## Résultats attendus

Vous verrez des rapports détaillés sur:
- **Consommation énergétique** (Watts, Wh, kWh)
- **Émissions CO2** (grammes)
- **Durée d'entraînement**
- **Suggestions d'optimisation**
- **Réduction de taille du modèle** après quantization

## Tests

```bash
python tests/test_ecotrain.py
```

## Documentation

- `README.md` - Documentation utilisateur complète
- `TECHNICAL_DOC.md` - Documentation technique détaillée
- `examples/basic_example.py` - Code d'exemple commenté

## Support

Pour toute question, ouvrez une issue sur GitHub:
https://github.com/ONDM-code/ecotrain/issues
