Infraestructura para tesis — segmentación de iris y pupila
=========================================================

Este repositorio respalda la tesis de la Universidad de Buenos Aires que busca segmentar iris y pupila en imágenes de ojo. La infraestructura mantiene el entrenamiento, evaluación, exportación y demostración de modelos de segmentación (principalmente `unet_se_small`, `seresnext_unet`, y los nuevos modelos **YOLO11**) usando los conjuntos de datos MOBIUS, IrisPupilEye y TayedEyes.

Directorio principal
-------------------

- `irispupilnet/`: contiene la definición del dataset CSV, los modelos registrados (`unet_se_small`, `seresnext_unet`, y las variantes **YOLO11**) y la lógica de entrenamiento (`train.py`). El loader fuerza el modo escala de grises por defecto y genera métricas/plots en cada época, pero se puede habilitar RGB con `--color`.
  * **Nuevos modelos YOLO11**: `yolo11n_seg`, `yolo11s_seg`, `yolo11m_seg`, `yolo11l_seg` - usando YOLO11 pretrenado como backbone con cabezal de segmentación semántica personalizado.
  * **Entrenamiento nativo YOLO**: Scripts en `tools/yolo/` para entrenar con la API nativa de YOLO11 (segmentación de instancias convertida a semántica).
- `data/` y `dataset/`: guardan los archivos de imágenes/máscaras y los CSV originales o generados que apuntan a ellos.  
  * `data/irispupileye_mobius` y `data/tayed_mobius` son los conjuntos de imágenes importadas.  
  * `dataset/mobius_output` contiene las versiones procesadas de MOBIUS (CSV + gráficas).  
- `tools/prepare/` y `tools/analyze/`: scripts de apoyo para crear CSVs (`create_mobius_csv_from_dir.py`), dividir en splits (`split_mobius.py`) y analizar distribuciones de los datos (`plot_mobius_summary.py`).
- `demo/`: el demo con MediaPipe/OpenCV para visualizar predicciones en tiempo real. Usa `--bw` (grayscale, por defecto) o `--rgb` para seleccionar cómo se alimenta el modelo ONNX.
- `export/`: herramientas para convertir checkpoints de PyTorch a ONNX (soporte NHWC, configurable en canales). El README de ese submódulo explica cómo ejecutar la exportación.

CSV de dataset
--------------

Los CSV que alimentan el loader siguen este esquema mínimo:

```
rel_image_path, rel_mask_path, split, dataset_format, ...
```

`rel_image_path` y `rel_mask_path` son rutas relativas desde `--data-root` hacia cada imagen y máscara, `split` determina `train/val/test`, y `dataset_format` informa a `irispupilnet/utils/mask_formats.py` cómo convertir el PNG de máscara a clases (`mobius_3c`, `iris_pupil_eye_cls`, `mobius`, `tayed_3c`, etc.). Los CSV producidos por los scripts de `tools/prepare` agregan columnas adicionales (metadatos por sujeto, fuente, tamaños) para posibilitar análisis.

Conjuntos explorados
--------------------

- **MOBIUS**: el dataset principal de la tesis, anotado con iris/pupila y dividido mediante `tools/prepare/split_mobius.py`. Se encuentra dentro de `dataset/mobius_output`.
- **IrisPupilEye**: dataset complementario almacenado en `data/irispupileye_mobius`, también referido en `dataset/merged/all_datasets.csv`.
- **TayedEyes**: otro dataset sintético del que se mantienen imágenes y máscaras en `data/tayed_mobius`.

Cada CSV que combina esos datos usa los formatos referidos arriba y se carga en el training loop con `--csv`, `--data-root`, y `--dataset csv_seg`.

Modelos YOLO11
---------------

El proyecto ahora soporta modelos YOLO11 para segmentación de iris/pupila mediante **dos enfoques distintos**:

### Enfoque 1: YOLO Backbone + Cabezal Semántico (Integrado)

**Ubicación**: `irispupilnet/models/yolo11_seg.py`

Usa YOLO11 pretrenado en COCO como extractor de características (backbone), añadiendo un decoder personalizado para segmentación semántica. Se integra perfectamente con el pipeline de entrenamiento existente de IrisPupilNet.

**Variantes disponibles**:
- `yolo11n_seg` - Nano (~2.7M parámetros, más rápido)
- `yolo11s_seg` - Small (~9.4M parámetros, balanceado)
- `yolo11m_seg` - Medium (~20M parámetros, mayor precisión)
- `yolo11l_seg` - Large (~25M parámetros, máxima precisión)

**Uso**:
```bash
python -m irispupilnet.train \
  --model yolo11n_seg \
  --csv dataset/merged/all_datasets.csv \
  --data-root dataset \
  --img-size 160 \
  --batch-size 16 \
  --epochs 50
```

### Enfoque 2: Entrenamiento Nativo YOLO (Máximo Rendimiento)

**Ubicación**: `tools/yolo/`

Convierte máscaras semánticas a formato de instancias (1 iris + 1 pupila por imagen) y usa la API nativa de entrenamiento de YOLO11. Aprovecha el modelo completo pretrenado y las optimizaciones de YOLO.

**Flujo de trabajo**:
```bash
# 1. Convertir dataset
python tools/prepare/convert_to_yolo_instance.py \
  --csv dataset/merged/all_datasets.csv \
  --data-root dataset \
  --output yolo_dataset \
  --img-size 160

# 2. Entrenar con API nativa de YOLO
python tools/yolo/train_yolo_native.py \
  --data yolo_dataset/dataset.yaml \
  --model yolo11n-seg \
  --epochs 50 \
  --imgsz 160 \
  --batch 16

# 3. Predecir (convierte instancias → semántico)
python tools/yolo/predict_yolo_to_semantic.py \
  --weights runs/yolo_native/train/weights/best.pt \
  --source images/ \
  --output predictions
```

**Comparación rápida**:
| Característica | Enfoque 1 (Backbone) | Enfoque 2 (Nativo) |
|----------------|---------------------|-------------------|
| Configuración | ✓ Listo para usar | Requiere conversión |
| Entrenamiento | Pipeline IrisPupilNet | API nativa YOLO |
| Salida | Máscaras semánticas directas | Instancias → semántico |
| Uso de YOLO | Solo backbone (~60%) | Modelo completo (100%) |
| Rendimiento | Bueno | Potencialmente mejor |

Ver documentación completa en `CLAUDE.md` y `tools/yolo/README.md`.
