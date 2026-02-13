"""
Script de verificación del entorno de entrenamiento LoRA.

Valida disponibilidad de hardware (GPU/CUDA), instalación correcta de dependencias
críticas, capacidad de carga del modelo base Stable Diffusion, presencia del dataset,
y recursos computacionales suficientes para ejecutar el pipeline de fine-tuning.
"""

import sys
import os
from pathlib import Path
import yaml


def print_section(title: str):
    """Imprime encabezado de sección."""
    print(f"\n{'─' * 60}")
    print(f"{title}")
    print('─' * 60)


def check_python_version():
    """Verifica versión de Python."""
    print_section("1. Verificación de Python")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Se requiere Python 3.8 o superior")
        return False
    print("✓ Versión compatible")
    return True


def check_pytorch_cuda():
    """Verifica instalación de PyTorch y disponibilidad de CUDA."""
    print_section("2. Verificación de PyTorch y CUDA")
    
    try:
        import torch
        print(f"PyTorch versión: {torch.__version__}")
        
        # Verificar CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA disponible: {'✓ Sí' if cuda_available else '✗ No'}")
        
        if cuda_available:
            print(f"Versión CUDA: {torch.version.cuda}")
            print(f"GPUs detectadas: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Verificar memoria mínima recomendada
            if gpu_memory < 10:
                print("⚠ ADVERTENCIA: Se recomienda GPU con ≥16GB VRAM para entrenamiento LoRA")
                print("  Considerar reducir batch_size o usar gradient_accumulation")
        else:
            print("⚠ ADVERTENCIA: No se detectó GPU. Entrenamiento será extremadamente lento en CPU")
            print("  Se recomienda encarecidamente usar GPU con CUDA")
        
        return True
    except ImportError as e:
        print(f"❌ ERROR: PyTorch no instalado correctamente: {e}")
        return False


def check_critical_libraries():
    """Verifica instalación de librerías críticas."""
    print_section("3. Verificación de Librerías Críticas")
    
    libraries = {
        'diffusers': '0.26.0',
        'transformers': '4.37.0',
        'accelerate': '0.26.0',
        'peft': '0.8.0',
        'PIL': None,  # Pillow
        'cv2': None,  # opencv-python
        'albumentations': None,
        'yaml': None,
    }
    
    all_ok = True
    for lib_name, min_version in libraries.items():
        try:
            if lib_name == 'PIL':
                from PIL import Image
                import PIL
                version = PIL.__version__
                lib_display = 'Pillow'
            elif lib_name == 'cv2':
                import cv2
                version = cv2.__version__
                lib_display = 'opencv-python'
            elif lib_name == 'yaml':
                import yaml
                version = yaml.__version__ if hasattr(yaml, '__version__') else 'OK'
                lib_display = 'PyYAML'
            else:
                module = __import__(lib_name)
                version = module.__version__
                lib_display = lib_name
            
            print(f"✓ {lib_display}: {version}")
        except ImportError as e:
            print(f"❌ {lib_name}: NO INSTALADO")
            all_ok = False
    
    return all_ok


def check_stable_diffusion_model():
    """Intenta cargar modelo base de Stable Diffusion."""
    print_section("4. Verificación de Stable Diffusion")
    
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        print("Intentando cargar modelo base: runwayml/stable-diffusion-v1-5")
        print("(Primera ejecución descargará ~4GB, puede tomar varios minutos)")
        
        # Cargar en CPU para verificación rápida
        # En entrenamiento real se usará GPU
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            variant="fp16" if torch.cuda.is_available() else None,
        )
        
        print("✓ Modelo base cargado exitosamente")
        print(f"  UNet parámetros: ~860M")
        print(f"  Text Encoder: CLIP ViT-L/14")
        
        # Limpiar memoria
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"❌ ERROR al cargar modelo: {e}")
        print("  Verifique conexión a internet y espacio en disco (~10GB requeridos)")
        return False


def check_dataset():
    """Verifica presencia y estructura del dataset."""
    print_section("5. Verificación del Dataset")
    
    # Cargar configuración
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print(f"❌ No se encontró config.yaml en {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    raw_data_path = Path(config['paths']['raw_data'])
    target_species = config['data']['target_species']
    
    print(f"Directorio raw: {raw_data_path}")
    print(f"Especie objetivo: {target_species}")
    
    if not raw_data_path.exists():
        print(f"❌ Directorio de datos no encontrado: {raw_data_path}")
        return False
    
    # Verificar estructura
    phases = ['early', 'late']
    species_found = {target_species: {'early': 0, 'late': 0}}
    
    for phase in phases:
        species_path = raw_data_path / phase / target_species
        if species_path.exists():
            images = list(species_path.glob("*.jpg")) + list(species_path.glob("*.png"))
            species_found[target_species][phase] = len(images)
            print(f"✓ {phase}/{target_species}: {len(images)} imágenes")
        else:
            print(f"⚠ {phase}/{target_species}: No encontrado")
    
    total_images = sum(species_found[target_species].values())
    
    if total_images == 0:
        print(f"❌ No se encontraron imágenes de {target_species}")
        return False
    elif total_images < 100:
        print(f"⚠ ADVERTENCIA: Solo {total_images} imágenes encontradas")
        print("  Se recomienda ≥500 imágenes para entrenamiento LoRA robusto")
    else:
        print(f"✓ Total: {total_images} imágenes disponibles")
    
    # Verificar Maize para regularización
    if config['regularization']['enabled']:
        maize_count = 0
        for phase in phases:
            maize_path = raw_data_path / phase / "maize"
            if maize_path.exists():
                maize_images = list(maize_path.glob("*.jpg")) + list(maize_path.glob("*.png"))
                maize_count += len(maize_images)
        
        print(f"\nRegularización con Maize: {maize_count} imágenes disponibles")
        if maize_count < config['regularization']['num_maize_images']:
            print(f"⚠ Se requieren {config['regularization']['num_maize_images']} pero solo hay {maize_count}")
    
    return True


def check_disk_space():
    """Verifica espacio en disco disponible."""
    print_section("6. Verificación de Recursos")
    
    try:
        import psutil
        
        # Espacio en disco
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        print(f"Espacio en disco disponible: {disk_free_gb:.1f} GB")
        
        if disk_free_gb < 20:
            print("⚠ ADVERTENCIA: Se recomienda ≥20GB libres para modelos y checkpoints")
        else:
            print("✓ Espacio suficiente")
        
        # Memoria RAM
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024**3)
        ram_available_gb = ram.available / (1024**3)
        print(f"Memoria RAM: {ram_available_gb:.1f} GB disponibles de {ram_total_gb:.1f} GB")
        
        if ram_available_gb < 8:
            print("⚠ Se recomienda ≥16GB RAM para procesamiento eficiente")
        
        return True
    except ImportError:
        print("⚠ psutil no disponible, omitiendo verificación de recursos")
        return True


def check_config_file():
    """Verifica integridad del archivo de configuración."""
    print_section("7. Verificación de Configuración")
    
    config_path = Path("config/config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verificar secciones críticas
        required_sections = ['paths', 'data', 'lora', 'training', 'regularization']
        for section in required_sections:
            if section in config:
                print(f"✓ Sección '{section}' presente")
            else:
                print(f"❌ Sección '{section}' faltante")
                return False
        
        # Verificar parámetros clave
        print(f"\nConfiguración de entrenamiento:")
        print(f"  Base model: {config['lora']['base_model']}")
        print(f"  LoRA rank: {config['lora']['rank']}")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Max steps: {config['training']['max_train_steps']}")
        print(f"  Mixed precision: {config['training']['mixed_precision']}")
        
        return True
    except Exception as e:
        print(f"❌ Error al leer configuración: {e}")
        return False


def main():
    """Ejecuta todas las verificaciones."""
    print("\n" + "═" * 60)
    print("VERIFICACIÓN DE ENTORNO DE ENTRENAMIENTO LORA")
    print("Pipeline de Generación Sintética de Sorghum")
    print("═" * 60)
    
    results = []
    
    # Ejecutar verificaciones
    results.append(("Python", check_python_version()))
    results.append(("PyTorch/CUDA", check_pytorch_cuda()))
    results.append(("Librerías", check_critical_libraries()))
    results.append(("Stable Diffusion", check_stable_diffusion_model()))
    results.append(("Dataset", check_dataset()))
    results.append(("Recursos", check_disk_space()))
    results.append(("Configuración", check_config_file()))
    
    # Resumen final
    print_section("RESUMEN DE VERIFICACIÓN")
    
    all_passed = True
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "═" * 60)
    if all_passed:
        print("✓ ENTORNO LISTO PARA ENTRENAMIENTO")
        print("\nPróximos pasos:")
        print("  1. python src/data/aerial_preprocessing.py")
        print("  2. python src/data/temporal_agnostic_split.py")
        print("  3. python src/training/maize_regularization.py")
        print("  4. python src/training/sorghum_lora.py  # (pendiente implementar)")
    else:
        print("✗ CORRIJA LOS ERRORES ANTES DE CONTINUAR")
        print("\nRevisione los errores marcados arriba y:")
        print("  - Verifique instalación de dependencias: pip install -r requirements.txt")
        print("  - Confirme que las imágenes están en ./imgs/")
        print("  - Asegure disponibilidad de GPU con CUDA")
    print("═" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
