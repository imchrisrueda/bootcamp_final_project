"""
Módulo de particionamiento estratificado temporal-agnostic.

Genera splits train/val/test (80/10/10) preservando distribución de fases fenológicas
early:late, con exportación de manifests JSON conteniendo rutas, metadata y checksums
para trazabilidad completa del pipeline.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from collections import defaultdict


class TemporalAgnosticSplitter:
    """
    Genera particiones estratificadas para entrenamiento temporal-agnostic.
    
    Mantiene proporción de fases fenológicas (early/late) en cada split para
    asegurar que el modelo aprenda características morfológicas independientes
    del estadio de crecimiento, mejorando generalización temporal.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa splitter con configuración del proyecto.
        
        Args:
            config_path: Ruta al archivo de configuración YAML.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.train_split = self.config['data']['train_split']
        self.val_split = self.config['data']['val_split']
        self.test_split = self.config['data']['test_split']
        self.seed = self.config['data']['seed']
        
        # Fijar semilla para reproducibilidad
        random.seed(self.seed)
        
        # Rutas
        self.processed_path = Path(self.config['paths']['processed_data'])
        self.splits_path = Path(self.config['paths']['splits'])
        self.splits_path.mkdir(parents=True, exist_ok=True)
    
    def load_metadata(self, species: str) -> List[Dict]:
        """
        Carga metadata de imágenes procesadas.
        
        Args:
            species: Nombre de la especie.
            
        Returns:
            Lista de diccionarios con metadata de cada imagen.
        """
        metadata_path = self.processed_path / f"{species}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata no encontrada: {metadata_path}. "
                f"Ejecute primero aerial_preprocessing.py"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def stratified_split(self, metadata: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Realiza split estratificado preservando distribución de fases.
        
        Algoritmo:
        1. Agrupa imágenes por fase (early/late)
        2. Calcula proporción de cada fase en dataset completo
        3. Aplica splits manteniendo estas proporciones en train/val/test
        
        Args:
            metadata: Lista completa de metadata de imágenes.
            
        Returns:
            Tupla (train_meta, val_meta, test_meta).
        """
        # Agrupar por fase
        phases_dict = defaultdict(list)
        for meta in metadata:
            phase = meta['phase']
            phases_dict[phase].append(meta)
        
        train_list, val_list, test_list = [], [], []
        
        # Aplicar split a cada fase independientemente
        for phase, phase_images in phases_dict.items():
            # Shuffle dentro de cada fase
            random.shuffle(phase_images)
            
            n_images = len(phase_images)
            n_train = int(n_images * self.train_split)
            n_val = int(n_images * self.val_split)
            
            # Particionar
            train_phase = phase_images[:n_train]
            val_phase = phase_images[n_train:n_train + n_val]
            test_phase = phase_images[n_train + n_val:]
            
            train_list.extend(train_phase)
            val_list.extend(val_phase)
            test_list.extend(test_phase)
        
        # Shuffle final de cada split (mezclar fases)
        random.shuffle(train_list)
        random.shuffle(val_list)
        random.shuffle(test_list)
        
        return train_list, val_list, test_list
    
    def compute_split_statistics(self, split_data: List[Dict], split_name: str) -> Dict:
        """
        Calcula estadísticas de un split.
        
        Args:
            split_data: Metadata del split.
            split_name: Nombre del split (train/val/test).
            
        Returns:
            Diccionario con estadísticas.
        """
        total = len(split_data)
        
        # Contar por fase
        phase_counts = defaultdict(int)
        for meta in split_data:
            phase_counts[meta['phase']] += 1
        
        # Calcular proporciones
        phase_proportions = {
            phase: count / total for phase, count in phase_counts.items()
        }
        
        return {
            "split_name": split_name,
            "total_images": total,
            "phase_counts": dict(phase_counts),
            "phase_proportions": phase_proportions
        }
    
    def save_manifest(self, split_data: List[Dict], split_name: str, species: str):
        """
        Guarda manifest de un split con rutas e información completa.
        
        Args:
            split_data: Metadata del split.
            split_name: Nombre del split.
            species: Especie procesada.
        """
        manifest_path = self.splits_path / f"{species}_{split_name}_manifest.json"
        
        # Construir manifest con información relevante
        manifest = {
            "species": species,
            "split": split_name,
            "num_images": len(split_data),
            "seed": self.seed,
            "split_ratios": {
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split
            },
            "images": []
        }
        
        # Agregar información de cada imagen
        for meta in split_data:
            manifest["images"].append({
                "path": meta['processed_path'],
                "filename": meta['filename'],
                "phase": meta['phase'],
                "image_number": meta['image_number'],
                "original_width": meta['original_width'],
                "original_height": meta['original_height'],
                "md5": meta['md5']
            })
        
        # Guardar manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Manifest guardado: {manifest_path}")
    
    def run(self, species: str = "sorghum"):
        """
        Ejecuta pipeline de particionamiento completo.
        
        Args:
            species: Especie a particionar.
        """
        print(f"Iniciando particionamiento estratificado de {species}")
        print(f"Ratios: Train={self.train_split}, Val={self.val_split}, Test={self.test_split}")
        print(f"Seed: {self.seed}")
        
        # Cargar metadata
        metadata = self.load_metadata(species)
        print(f"\nTotal de imágenes: {len(metadata)}")
        
        # Realizar split estratificado
        train_meta, val_meta, test_meta = self.stratified_split(metadata)
        
        # Calcular estadísticas
        print("\nEstadísticas de particiones:")
        splits = [
            (train_meta, "train"),
            (val_meta, "val"),
            (test_meta, "test")
        ]
        
        for split_data, split_name in splits:
            stats = self.compute_split_statistics(split_data, split_name)
            print(f"\n{split_name.upper()}:")
            print(f"  Total: {stats['total_images']} imágenes")
            print(f"  Distribución por fase:")
            for phase, count in stats['phase_counts'].items():
                prop = stats['phase_proportions'][phase]
                print(f"    {phase}: {count} imágenes ({prop*100:.1f}%)")
            
            # Guardar manifest
            self.save_manifest(split_data, split_name, species)
        
        # Guardar resumen consolidado
        summary = {
            "species": species,
            "total_images": len(metadata),
            "seed": self.seed,
            "splits": {}
        }
        
        for split_data, split_name in splits:
            summary["splits"][split_name] = self.compute_split_statistics(split_data, split_name)
        
        summary_path = self.splits_path / f"{species}_split_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResumen guardado: {summary_path}")
        print("\nParticionamiento completado exitosamente.")


def main():
    """Función principal de ejecución."""
    splitter = TemporalAgnosticSplitter()
    
    # Particionar especie objetivo: Sorghum
    splitter.run(species="sorghum")


if __name__ == "__main__":
    main()
