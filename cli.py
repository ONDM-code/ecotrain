#!/usr/bin/env python3
"""
EcoTrain CLI - Interface en ligne de commande pour EcoTrain

Usage:
    python cli.py analyze <model_file>        # Analyser un modèle sauvegardé
    python cli.py train <script>              # Entraîner avec suivi énergétique
    python cli.py demo                        # Exécuter la démo
"""

import sys
import os
import argparse

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecotrain import EcoMonitor, ModelOptimizer
import torch


def analyze_command(model_path):
    """Analyse un modèle PyTorch sauvegardé."""
    print(f"\nAnalyse du modèle: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Erreur: Le fichier {model_path} n'existe pas")
        return 1
    
    try:
        # Charger le modèle
        model = torch.load(model_path)
        model.eval()
        
        # Analyser
        optimizer = ModelOptimizer()
        analysis = optimizer.analyze_model(model)
        optimizer.print_optimization_report(analysis)
        
        return 0
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
        return 1


def train_command(script_path):
    """Exécute un script d'entraînement avec suivi énergétique."""
    print(f"\nExécution de {script_path} avec suivi énergétique...")
    
    if not os.path.exists(script_path):
        print(f"Erreur: Le fichier {script_path} n'existe pas")
        return 1
    
    try:
        # Lire et exécuter le script avec monitoring
        monitor = EcoMonitor()
        monitor.start()
        
        with open(script_path) as f:
            code = f.read()
        
        exec(code)
        
        metrics = monitor.stop()
        monitor.print_report(metrics)
        
        return 0
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
        return 1


def demo_command():
    """Exécute la démonstration complète."""
    print("\nExécution de la démonstration EcoTrain...")
    
    # Essayer plusieurs chemins possibles
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'basic_example.py'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'basic_example.py'),
        'examples/basic_example.py'
    ]
    
    example_path = None
    for path in possible_paths:
        if os.path.exists(path):
            example_path = path
            break
    
    if example_path:
        os.system(f"python {example_path}")
        return 0
    else:
        print(f"Erreur: Fichier de démonstration introuvable")
        print(f"Chemins testés: {possible_paths}")
        return 1


def main():
    """Point d'entrée principal du CLI."""
    parser = argparse.ArgumentParser(
        description='EcoTrain CLI - Outils pour réduire la consommation énergétique de l\'IA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python cli.py demo
  python cli.py analyze model.pth
  python cli.py train train_script.py
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande analyze
    parser_analyze = subparsers.add_parser('analyze', help='Analyser un modèle')
    parser_analyze.add_argument('model_file', help='Chemin vers le fichier du modèle')
    
    # Commande train
    parser_train = subparsers.add_parser('train', help='Entraîner avec suivi')
    parser_train.add_argument('script', help='Script Python d\'entraînement')
    
    # Commande demo
    parser_demo = subparsers.add_parser('demo', help='Exécuter la démonstration')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        return analyze_command(args.model_file)
    elif args.command == 'train':
        return train_command(args.script)
    elif args.command == 'demo':
        return demo_command()
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
