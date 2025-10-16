"""
EcoMonitor - Module pour surveiller la consommation énergétique des modèles
"""

import time
import psutil
from typing import Dict, Optional, Callable, Any
import torch


class EcoMonitor:
    """
    Moniteur de consommation énergétique pour l'entraînement de modèles.
    
    Utilise un proxy temporel et des estimations basées sur l'utilisation CPU/GPU
    pour approximer la consommation énergétique.
    """
    
    # Estimations de consommation (Watts)
    # Ces valeurs sont des approximations basées sur des configurations typiques
    TYPICAL_CPU_TDP = 65  # TDP moyen d'un CPU de bureau
    TYPICAL_GPU_TDP = 250  # TDP moyen d'un GPU de formation (ex: NVIDIA RTX 3080)
    
    def __init__(self):
        """Initialise le moniteur énergétique."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.cpu_percent_samples = []
        self.gpu_available = torch.cuda.is_available()
        
    def start(self):
        """Démarre le suivi énergétique."""
        self.start_time = time.time()
        self.cpu_percent_samples = []
        
    def stop(self) -> Dict[str, Any]:
        """
        Arrête le suivi et calcule les métriques énergétiques.
        
        Returns:
            Dictionnaire contenant les métriques énergétiques estimées
        """
        self.end_time = time.time()
        
        if self.start_time is None:
            raise ValueError("Le moniteur n'a pas été démarré")
        
        duration_seconds = self.end_time - self.start_time
        
        # Estimation de la consommation CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_power_watts = (cpu_percent / 100.0) * self.TYPICAL_CPU_TDP
        
        # Estimation de la consommation GPU (si disponible)
        gpu_power_watts = 0
        if self.gpu_available:
            # Approximation: on suppose que le GPU est utilisé à 80% pendant l'entraînement
            gpu_power_watts = 0.8 * self.TYPICAL_GPU_TDP
        
        # Consommation totale estimée
        total_power_watts = cpu_power_watts + gpu_power_watts
        energy_wh = (total_power_watts * duration_seconds) / 3600  # Watt-heures
        energy_kwh = energy_wh / 1000  # Kilowatt-heures
        
        # Estimation des émissions CO2 (en grammes)
        # Utilise une intensité carbone moyenne mondiale de ~475g CO2/kWh
        co2_intensity = 475  # g CO2/kWh
        co2_grams = energy_kwh * co2_intensity
        
        return {
            "duration_seconds": duration_seconds,
            "cpu_power_watts": cpu_power_watts,
            "gpu_power_watts": gpu_power_watts,
            "total_power_watts": total_power_watts,
            "energy_wh": energy_wh,
            "energy_kwh": energy_kwh,
            "co2_grams": co2_grams,
            "gpu_used": self.gpu_available
        }
    
    def track_training(self, training_fn: Callable, *args, **kwargs) -> tuple:
        """
        Trace une fonction d'entraînement et retourne les résultats avec les métriques.
        
        Args:
            training_fn: Fonction d'entraînement à tracer
            *args: Arguments positionnels pour la fonction
            **kwargs: Arguments nommés pour la fonction
            
        Returns:
            Tuple (résultat de la fonction, métriques énergétiques)
        """
        self.start()
        result = training_fn(*args, **kwargs)
        metrics = self.stop()
        return result, metrics
    
    def print_report(self, metrics: Dict[str, Any]):
        """
        Affiche un rapport formaté des métriques énergétiques.
        
        Args:
            metrics: Dictionnaire de métriques retourné par stop()
        """
        print("\n" + "="*60)
        print("RAPPORT DE CONSOMMATION ÉNERGÉTIQUE - EcoMonitor")
        print("="*60)
        print(f"Durée d'entraînement: {metrics['duration_seconds']:.2f} secondes")
        print(f"\nPuissance estimée:")
        print(f"  - CPU: {metrics['cpu_power_watts']:.2f} W")
        if metrics['gpu_used']:
            print(f"  - GPU: {metrics['gpu_power_watts']:.2f} W")
        print(f"  - Total: {metrics['total_power_watts']:.2f} W")
        print(f"\nÉnergie consommée:")
        print(f"  - {metrics['energy_wh']:.4f} Wh")
        print(f"  - {metrics['energy_kwh']:.6f} kWh")
        print(f"\nÉmissions CO2 estimées: {metrics['co2_grams']:.4f} g CO2")
        print("="*60 + "\n")
