# Pipeline de Generación Sintética de Imágenes de Sorghum

Implementación de pipeline híbrido para generación fotorrealista de imágenes sintéticas de Sorghum mediante fine-tuning LoRA sobre Stable Diffusion, con regularización basada en Maize para mejorar generalización morfológica en contexto de agricultura de precisión.

## Estructura del Proyecto

```
bootcamp_projects/
├── imgs/                    # Dataset original (raw)
│   ├── early/
│   │   ├── sorghum/        # 1,600 imágenes
│   │   ├── maize/          # 1,000 imágenes
│   │   └── atriplex/       # 1,000 imágenes
│   └── late/
│       ├── sorghum/        # 103 imágenes
│       ├── maize/          # 10,000 imágenes
│       └── atriplex/       # 1,459 imágenes
├── src/
│   ├── data/               # Módulos de preprocesamiento
│   ├── training/           # Entrenamiento LoRA
│   ├── generation/         # Generación sintética
│   └── evaluation/         # Métricas y validación
├── data/
│   ├── processed/          # Imágenes 512×512 procesadas
│   └── splits/             # Manifests train/val/test
├── models/                 # Checkpoints LoRA
├── outputs/                # Imágenes sintéticas generadas
├── config/                 # Archivos de configuración
└── logs/                   # TensorBoard y logs

```

## Dataset

**Origen:** Fotografías aéreas de dron capturadas a 11m de altura, recortadas a regiones de interés por especie.

**Especies:**
- **Sorghum** (objetivo): 1,703 imágenes (1,600 early + 103 late)
- **Maize** (regularización): 11,000 imágenes
- **Atriplex** (referencia): 2,459 imágenes

**Distribución:** Temporal-agnostic, combinando fases early y late en modelo unificado.

## Pipeline

### 1. Preprocesamiento
Resize inteligente a 512×512px con padding preservando aspect ratio, normalización fotométrica específica para perspectiva cenital.

### 2. Split Estratificado
Particiones train/val/test (80/10/10) manteniendo distribución temporal early:late (94:6).

### 3. Regularización Maize
Integración de 300 imágenes aleatorias de Maize (ratio 85% Sorghum / 15% Maize) con class preservation loss para prevenir overfitting.

### 4. Entrenamiento LoRA
Fine-tuning sobre Stable Diffusion 1.5: rank=32, lr=1e-4, 2,500 steps, augmentación on-the-fly.

### 5. Generación Sintética
Producción de 5,000 imágenes con variabilidad controlada en iluminación, textura de suelo, densidad vegetal.

### 6. Validación
FID score, Inception Score, filtrado por clasificador (confidence >0.90), detección de artefactos.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# 1. Preprocesar dataset
python src/data/aerial_preprocessing.py

# 2. Crear splits
python src/data/temporal_agnostic_split.py

# 3. Preparar regularización Maize
python src/training/maize_regularization.py

# 4. Entrenar LoRA
python src/training/sorghum_lora.py

# 5. Generar sintéticos
python src/generation/sorghum_synthesis.py

# 6. Evaluar calidad
python src/evaluation/sorghum_quality_metrics.py
```

## Configuración

Editar `config/config.yaml` para ajustar hiperparámetros del pipeline.

## Resultados Esperados

- **FID Score**: <40 (dataset sintético vs real)
- **Imágenes sintéticas finales**: 3,000-4,000 (post-filtrado)
- **Mejora downstream**: +10-15% mAP en detección

## Referencias

- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- Stable Diffusion: [Rombach et al., 2022](https://arxiv.org/abs/2112.10752)
