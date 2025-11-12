Infraestructura para tesis — segmentación de iris y pupila
=========================================================

Este repositorio respalda la tesis de la Universidad de Buenos Aires que busca segmentar iris y pupila en imágenes de ojo. La infraestructura mantiene el entrenamiento, evaluación, exportación y demostración de modelos de segmentación (principalmente `unet_se_small` y la variante `seresnext_unet`) usando los conjuntos de datos MOBIUS, IrisPupilEye y TayedEyes.

Directorio principal
-------------------

- `irispupilnet/`: contiene la definición del dataset CSV, los modelos registrados (`unet_se_small`, el nuevo `seresnext_unet`) y la lógica de entrenamiento (`train.py`). El loader fuerza el modo escala de grises por defecto y genera métricas/plots en cada época, pero se puede habilitar RGB con `--color`.
- `data/` y `dataset/`: guardan los archivos de imágenes/máscaras y los CSV originales o generados que apuntan a ellos.  
  * `data/irispupileye_mobius` y `data/tayed_mobius` son los conjuntos de imágenes importadas.  
  * `dataset/mobius_output` contiene las versiones procesadas de MOBIUS (CSV + gráficas).  
- `tools/prepare/` y `tools/analyze/`: scripts de apoyo para crear CSVs (`create_mobius_csv_from_dir.py`), dividir en splits (`split_mobius.py`) y analizar distribuciones de los datos (`plot_mobius_summary.py`).
- `demo/`: el demo con MediaPipe/OpenCV para visualizar predicciones en tiempo real. Usa `--input-channels` para elegir entre el ONNX entrenado en escala de grises o en RGB.
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
