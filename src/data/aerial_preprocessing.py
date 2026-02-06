"""
Módulo de preprocesamiento para imágenes aéreas de malezas capturadas con dron.

Implementa resize inteligente a 512×512px mediante padding que preserva aspect ratio,
normalización fotométrica adaptada a perspectiva cenital, y procesamiento batch eficiente
del dataset completo de Sorghum (1,703 imágenes).
"""

import os
import yaml
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
from tqdm import tqdm
import hashlib
import json


class AerialPreprocessor:
    """
    Procesador especializado para imágenes aéreas de malezas.
    
    Aplica transformaciones que preservan características morfológicas críticas
    observadas desde perspectiva cenital, incluyendo resize con padding inteligente
    para mantener proporciones geométricas y normalización fotométrica.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa el preprocesador con configuración del pipeline.
        
        Args:
            config_path: Ruta al archivo de configuración YAML.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.target_size = self.config['data']['image_size']
        self.resize_mode = self.config['preprocessing']['resize_mode']
        self.padding_color = tuple(self.config['preprocessing']['padding_color'])
        self.normalize = self.config['preprocessing']['normalize']
        self.mean = np.array(self.config['preprocessing']['mean'])
        self.std = np.array(self.config['preprocessing']['std'])
        
        # Rutas del proyecto
        self.raw_path = Path(self.config['paths']['raw_data'])
        self.processed_path = Path(self.config['paths']['processed_data'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def resize_with_padding(self, image: Image.Image) -> Image.Image:
        """
        Redimensiona imagen preservando aspect ratio mediante padding.
        
        Técnica específica para imágenes aéreas donde la proporción geométrica
        entre elementos (hojas, tallos) es crítica para identificación morfológica.
        El padding negro simula fondo de suelo oscuro común en imágenes de campo.
        
        Args:
            image: Imagen PIL a redimensionar.
            
        Returns:
            Imagen redimensionada a target_size×target_size con padding.
        """
        # Calcular aspect ratio y dimensiones escaladas
        width, height = image.size
        aspect_ratio = width / height
        
        if aspect_ratio > 1:
            # Imagen más ancha que alta
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            # Imagen más alta que ancha
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
        
        # Resize manteniendo proporciones
        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Crear canvas con padding
        canvas = Image.new('RGB', (self.target_size, self.target_size), self.padding_color)
        
        # Calcular posición centrada
        x_offset = (self.target_size - new_width) // 2
        y_offset = (self.target_size - new_height) // 2
        
        # Pegar imagen redimensionada en canvas
        canvas.paste(image_resized, (x_offset, y_offset))
        
        return canvas
    
    def normalize_image(self, image: Image.Image) -> np.ndarray:
        """
        Normaliza fotométricamente imagen usando estadísticas ImageNet.
        
        La normalización con parámetros ImageNet facilita transfer learning
        desde modelos pre-entrenados, preservando distribución de activaciones
        en capas profundas de redes neuronales.
        
        Args:
            image: Imagen PIL a normalizar.
            
        Returns:
            Array numpy normalizado [H, W, C].
        """
        # Convertir a array numpy y normalizar a [0, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        
        if self.normalize:
            # Normalizar con mean y std de ImageNet
            image_array = (image_array - self.mean) / self.std
        
        return image_array
    
    def compute_md5(self, file_path: Path) -> str:
        """
        Calcula hash MD5 para verificación de integridad.
        
        Args:
            file_path: Ruta al archivo.
            
        Returns:
            String hexadecimal del hash MD5.
        """
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def extract_metadata(self, file_path: Path) -> Dict:
        """
        Extrae metadata de imagen desde nombre de archivo y propiedades.
        
        Naming convention: {field}_{phase}_{species}_{number}.jpg
        Ejemplo: maize_1_sorghum_523.jpg
        
        Args:
            file_path: Path al archivo de imagen.
            
        Returns:
            Diccionario con metadata extraída.
        """
        filename = file_path.stem
        parts = filename.split('_')
        
        # Parse filename structure
        field = parts[0] if len(parts) > 0 else "unknown"
        phase_num = parts[1] if len(parts) > 1 else "0"
        species = parts[2] if len(parts) > 2 else "unknown"
        img_number = parts[3] if len(parts) > 3 else "0"
        
        # Mapear fase numérica a nombre
        phase_map = {"1": "early", "2": "late"}
        phase_name = phase_map.get(phase_num, "unknown")
        
        # Abrir imagen para obtener dimensiones originales
        with Image.open(file_path) as img:
            original_width, original_height = img.size
        
        return {
            "filename": file_path.name,
            "field": field,
            "phase": phase_name,
            "phase_num": int(phase_num),
            "species": species,
            "image_number": int(img_number),
            "original_width": original_width,
            "original_height": original_height,
            "original_aspect_ratio": original_width / original_height,
            "md5": self.compute_md5(file_path)
        }
    
    def process_image(self, input_path: Path, output_path: Path) -> Dict:
        """
        Procesa una imagen individual aplicando pipeline completo.
        
        Args:
            input_path: Ruta a imagen original.
            output_path: Ruta para guardar imagen procesada.
            
        Returns:
            Diccionario con metadata del procesamiento.
        """
        # Cargar imagen
        image = Image.open(input_path).convert('RGB')
        
        # Aplicar resize con padding
        image_processed = self.resize_with_padding(image)
        
        # Guardar imagen procesada
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_processed.save(output_path, quality=95, optimize=True)
        
        # Extraer metadata
        metadata = self.extract_metadata(input_path)
        metadata['processed_path'] = str(output_path)
        metadata['processed_size'] = self.target_size
        
        return metadata
    
    def process_species(self, species: str, phases: List[str]) -> List[Dict]:
        """
        Procesa todas las imágenes de una especie en fases especificadas.
        
        Args:
            species: Nombre de la especie (sorghum, maize, atriplex).
            phases: Lista de fases a procesar (early, late).
            
        Returns:
            Lista de diccionarios con metadata de cada imagen procesada.
        """
        all_metadata = []
        
        for phase in phases:
            phase_path = self.raw_path / phase / species
            
            if not phase_path.exists():
                print(f"Advertencia: Directorio no encontrado: {phase_path}")
                continue
            
            # Listar todas las imágenes
            image_files = sorted(list(phase_path.glob("*.jpg")))
            
            print(f"\nProcesando {len(image_files)} imágenes de {species} ({phase})")
            
            # Procesar cada imagen
            for img_file in tqdm(image_files, desc=f"{species}-{phase}"):
                # Definir ruta de salida manteniendo estructura
                relative_path = img_file.relative_to(self.raw_path)
                output_path = self.processed_path / relative_path
                
                try:
                    metadata = self.process_image(img_file, output_path)
                    all_metadata.append(metadata)
                except Exception as e:
                    print(f"\nError procesando {img_file.name}: {str(e)}")
                    continue
        
        return all_metadata
    
    def run(self, species: str = "sorghum", phases: List[str] = ["early", "late"]):
        """
        Ejecuta pipeline de preprocesamiento completo.
        
        Args:
            species: Especie objetivo a procesar.
            phases: Fases temporales a incluir.
        """
        print(f"Iniciando preprocesamiento de {species}")
        print(f"Configuración: {self.target_size}×{self.target_size}, modo={self.resize_mode}")
        
        # Procesar especie
        metadata_list = self.process_species(species, phases)
        
        # Guardar metadata consolidada
        metadata_path = self.processed_path / f"{species}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        # Resumen estadístico
        print(f"\nProcesamiento completado:")
        print(f"  Total imágenes procesadas: {len(metadata_list)}")
        print(f"  Metadata guardada en: {metadata_path}")
        
        # Estadísticas por fase
        phases_count = {}
        for meta in metadata_list:
            phase = meta['phase']
            phases_count[phase] = phases_count.get(phase, 0) + 1
        
        print(f"\nDistribución por fase:")
        for phase, count in phases_count.items():
            print(f"  {phase}: {count} imágenes")


def main():
    """Función principal de ejecución."""
    preprocessor = AerialPreprocessor()
    
    # Procesar especie objetivo: Sorghum
    preprocessor.run(species="sorghum", phases=["early", "late"])
    
    print("\nPreprocesamiento finalizado exitosamente.")


if __name__ == "__main__":
    main()
