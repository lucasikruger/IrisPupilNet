# Export Module

This module contains utilities for exporting trained PyTorch models to ONNX format.

## Quick Usage

```bash
python export/export_to_onnx.py \
  --checkpoint ../runs/experiment/best.pt \
  --out ../checkpoints/model.onnx \
  --size 160 \
  --classes 3 \
  --base 32
```

## Output Format

The exported ONNX model uses **NHWC layout** (batch, height, width, channels) for compatibility with browser and mobile runtimes:

- **Input**: `[batch, height, width, 3]` - RGB image in range [0, 1]
- **Output**: `[batch, height, width, num_classes]` - Logits (pre-softmax)

The batch dimension is dynamic, allowing flexible inference.

## Architecture Compatibility

The export script now includes **Squeeze-and-Excitation blocks** to match the training architecture (`UNetSESmall`). Previous versions used basic `DoubleConv` blocks and would fail to load trained checkpoints.

## Loading Checkpoints

The script supports two checkpoint formats:

1. **Full checkpoint** (recommended):
   ```python
   {
       'model': state_dict,
       'args': training_args
   }
   ```

2. **State dict only**:
   ```python
   state_dict
   ```

## Testing the Export

After export, verify the model works:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
dummy_input = np.random.rand(1, 160, 160, 3).astype(np.float32)
outputs = session.run(None, {"input": dummy_input})
logits = outputs[0]  # [1, 160, 160, 3]
predictions = np.argmax(logits, axis=-1)  # [1, 160, 160]
```

## Opset Version

The export uses ONNX opset version 17 for broad compatibility with recent ONNX Runtime versions.

## Known Issues

- **Architecture mismatch**: Now fixed (includes SE blocks)
- **Input normalization**: Model expects inputs in [0, 1] range (as during training)
