"""
Pipeline de regularización con imágenes de Maize.

Implementa selección aleatoria estratificada de 300 imágenes de Maize (gramínea C4
morfológicamente similar a Sorghum) para integración en entrenamiento LoRA mediante
class preservation loss, previniendo overfitting a ruido específico de Sorghum y
mejorando generalización a características morfológicas robustas.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import yaml
from collections import defaultdict
import shutil
from tqdm import tqdm


class MaizeRegularizationPipeline:
    """
    Prepara dataset de regularización con imágenes de Maize.
    
    La similaridad estructural entre Maize y Sorghum (ambas gramíneas C4 con
    morfología foliar comparable) permite usar Maize como regularizador que
    previene memorización de artefactos específicos del dataset de Sorghum,
    forzando al modelo a aprender características botánicas generalizables.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa pipeline de regularización.
        
        Args:
            config_path: Ruta al archivo de configuración YAML.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Parámetros de regularización
        self.num_maize_images = self.config['regularization']['num_maize_images']
        self.sampling_mode = self.config['regularization']['sampling_mode']
        self.seed = self.config['data']['seed']
        
        # Ratios de mezcla
        self.sorghum_ratio = self.config['regularization']['sorghum_ratio']
        self.maize_ratio = self.config['regularization']['maize_ratio']
        
        # Fijar semilla
        random.seed(self.seed)
        
        # Rutas
        self.raw_path = Path(self.config['paths']['raw_data'])
        self.processed_path = Path(self.config['paths']['processed_data'])
        self.splits_path = Path(self.config['paths']['splits'])
        
        # Crear directorio para dataset regularizado
        self.regularized_path = self.processed_path / "regularization"
        self.regularized_path.mkdir(parents=True, exist_ok=True)
    
    def collect_maize_images(self) -> List[Path]:
        """
        Recolecta todas las imágenes de Maize disponibles.
        
        Returns:
            Lista de rutas a imágenes de Maize.
        """
        maize_images = []
        
        # Buscar en ambas fases
        for phase in ["early", "late"]:
            phase_path = self.raw_path / phase / "maize"
            
            if phase_path.exists():
                images = list(phase_path.glob("*.jpg"))
                maize_images.extend(images)
        
        return maize_images
    
    def stratified_sampling(self, all_images: List[Path], n_samples: int) -> List[Path]:
        """
        Muestreo estratificado balanceando fases temporales.
        
        Selecciona proporcionalmente desde early y late para mantener
        diversidad temporal en el dataset de regularización.
        
        Args:
            all_images: Lista completa de imágenes disponibles.
            n_samples: Número de imágenes a muestrear.
            
        Returns:
            Lista de imágenes seleccionadas.
        """
        # Agrupar por fase
        phase_groups = defaultdict(list)
        for img_path in all_images:
            # Detectar fase desde estructura de directorio
            if "/early/" in str(img_path):
                phase = "early"
            elif "/late/" in str(img_path):
                phase = "late"
            else:
                phase = "unknown"
            
            phase_groups[phase].append(img_path)
        
        # Calcular cuántas imágenes tomar de cada fase
        total_images = len(all_images)
        selected_images = []
        
        for phase, images in phase_groups.items():
            phase_proportion = len(images) / total_images
            n_phase = int(n_samples * phase_proportion)
            
            # Muestrear aleatoriamente
            sampled = random.sample(images, min(n_phase, len(images)))
            selected_images.extend(sampled)
        
        # Ajustar si no llegamos exactamente a n_samples
        if len(selected_images) < n_samples:
            remaining = n_samples - len(selected_images)
            available = [img for img in all_images if img not in selected_images]
            additional = random.sample(available, min(remaining, len(available)))
            selected_images.extend(additional)
        
        # Si excedimos, recortar aleatoriamente
        if len(selected_images) > n_samples:
            selected_images = random.sample(selected_images, n_samples)
        
        return selected_images
    
    def random_sampling(self, all_images: List[Path], n_samples: int) -> List[Path]:
        """
        Muestreo aleatorio simple.
        
        Args:
            all_images: Lista completa de imágenes.
            n_samples: Número a muestrear.
            
        Returns:
            Lista de imágenes seleccionadas.
        """
        return random.sample(all_images, min(n_samples, len(all_images)))
    
    def process_and_copy_maize(self, selected_images: List[Path]) -> List[Dict]:
        """
        Procesa y copia imágenes de Maize al directorio de regularización.
        
        Args:
            selected_images: Lista de imágenes seleccionadas.
            
        Returns:
            Lista de metadata de imágenes procesadas.
        """
        from src.data.aerial_preprocessing import AerialPreprocessor
        
        preprocessor = AerialPreprocessor()
        metadata_list = []
        
        print(f"\nProcesando {len(selected_images)} imágenes de Maize para regularización")
        
        for img_path in tqdm(selected_images, desc="Procesando Maize"):
            try:
                # Definir ruta de salida
                output_filename = f"maize_reg_{img_path.stem}.jpg"
                output_path = self.regularized_path / output_filename
                
                # Procesar imagen
                metadata = preprocessor.process_image(img_path, output_path)
                metadata['role'] = 'regularization'
                metadata['species'] = 'maize'
                metadata_list.append(metadata)
                
            except Exception as e:
                print(f"\nError procesando {img_path.name}: {str(e)}")
                continue
        
        return metadata_list
    
    def create_combined_manifest(self, maize_metadata: List[Dict]) -> Dict:
        """
        Crea manifest combinado de Sorghum + Maize para entrenamiento.
        
        Args:
            maize_metadata: Metadata de imágenes de Maize.
            
        Returns:
            Diccionario con manifest combinado.
        """
        # Cargar manifest de Sorghum (train)
        sorghum_train_path = self.splits_path / "sorghum_train_manifest.json"
        
        if not sorghum_train_path.exists():
            raise FileNotFoundError(
                f"Manifest de Sorghum no encontrado: {sorghum_train_path}. "
                f"Ejecute primero temporal_agnostic_split.py"
            )
        
        with open(sorghum_train_path, 'r') as f:
            sorghum_manifest = json.load(f)
        
        # Crear manifest combinado
        combined_manifest = {
            "training_mode": "lora_with_regularization",
            "target_species": "sorghum",
            "regularization_species": "maize",
            "sorghum_ratio": self.sorghum_ratio,
            "maize_ratio": self.maize_ratio,
            "class_preservation_weight": self.config['regularization']['class_preservation_weight'],
            "total_images": len(sorghum_manifest['images']) + len(maize_metadata),
            "sorghum_images": len(sorghum_manifest['images']),
            "maize_images": len(maize_metadata),
            "datasets": {
                "sorghum": {
                    "images": sorghum_manifest['images'],
                    "role": "target",
                    "weight": self.sorghum_ratio
                },
                "maize": {
                    "images": [
                        {
                            "path": meta['processed_path'],
                            "filename": meta['filename'],
                            "phase": meta.get('phase', 'unknown'),
                            "species": "maize"
                        }
                        for meta in maize_metadata
                    ],
                    "role": "regularization",
                    "weight": self.maize_ratio
                }
            }
        }
        
        return combined_manifest
    
    def run(self):
        """
        Ejecuta pipeline completo de regularización.
        """
        print("Iniciando pipeline de regularización con Maize")
        print(f"Configuración:")
        print(f"  Imágenes Maize objetivo: {self.num_maize_images}")
        print(f"  Modo de muestreo: {self.sampling_mode}")
        print(f"  Ratio Sorghum:Maize = {self.sorghum_ratio}:{self.maize_ratio}")
        
        # Recolectar imágenes de Maize
        all_maize = self.collect_maize_images()
        print(f"\nImágenes de Maize disponibles: {len(all_maize)}")
        
        # Muestrear según modo configurado
        if self.sampling_mode == "balanced_phases":
            selected_maize = self.stratified_sampling(all_maize, self.num_maize_images)
        else:
            selected_maize = self.random_sampling(all_maize, self.num_maize_images)
        
        print(f"Imágenes seleccionadas: {len(selected_maize)}")
        
        # Procesar y copiar imágenes
        maize_metadata = self.process_and_copy_maize(selected_maize)
        
        # Guardar metadata de Maize
        maize_metadata_path = self.regularized_path / "maize_regularization_metadata.json"
        with open(maize_metadata_path, 'w') as f:
            json.dump(maize_metadata, f, indent=2)
        
        print(f"\nMetadata de Maize guardada: {maize_metadata_path}")
        
        # Crear manifest combinado
        combined_manifest = self.create_combined_manifest(maize_metadata)
        
        # Guardar manifest combinado
        combined_path = self.splits_path / "combined_train_manifest.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_manifest, f, indent=2)
        
        print(f"Manifest combinado guardado: {combined_path}")
        
        # Resumen final
        print("\nResumen del dataset de entrenamiento:")
        print(f"  Sorghum: {combined_manifest['sorghum_images']} imágenes ({self.sorghum_ratio*100:.0f}%)")
        print(f"  Maize: {combined_manifest['maize_images']} imágenes ({self.maize_ratio*100:.0f}%)")
        print(f"  Total: {combined_manifest['total_images']} imágenes")
        print("\nPipeline de regularización completado exitosamente.")


def main():
    """Función principal de ejecución."""
    pipeline = MaizeRegularizationPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
