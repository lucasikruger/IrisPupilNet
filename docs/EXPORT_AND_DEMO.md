# Export and Demo Guide

This guide covers how to export trained models to ONNX format and run real-time inference.

## Quick Start

```bash
# 1. Export trained model to ONNX
python -m irispupilnet.export_onnx \
  --checkpoint runs/experiment/best.pt \
  --output iris_pupil_bw.onnx \
  --img-size 160 \
  --in-channels 1 \
  --num-classes 3

# 2. Run demo with webcam
python -m irispupilnet.demo \
  --model iris_pupil_bw.onnx \
  --source 0 \
  --img-size 160 \
  --color false
```

---

## Export to ONNX

### Basic Export

Export a trained PyTorch model to ONNX format:

```bash
python -m irispupilnet.export_onnx \
  --checkpoint runs/my_experiment/best.pt \
  --output my_model.onnx
```

### Full Options

```bash
python -m irispupilnet.export_onnx \
  --checkpoint runs/experiment/best.pt \
  --output iris_pupil.onnx \
  --model unet_se_small \
  --img-size 160 \
  --in-channels 1 \
  --num-classes 3 \
  --base 32 \
  --opset-version 14
```

### Export Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to PyTorch checkpoint (.pt file) |
| `--output` | str | **required** | Output path for ONNX model (.onnx file) |
| `--model` | str | `unet_se_small` | Model architecture (must match training) |
| `--img-size` | int | `160` | Input image size (must match training) |
| `--in-channels` | int | `1` | Number of input channels (1=grayscale, 3=RGB) |
| `--num-classes` | int | `3` | Number of output classes |
| `--base` | int | `32` | Base channel count (must match training) |
| `--opset-version` | int | `14` | ONNX opset version (14 is widely supported) |

### ONNX Model Format

**Input:**
- Shape: `[batch, channels, height, width]`
- Type: `float32`
- Range: `[0, 1]` (normalized)

**Output:**
- Shape: `[batch, num_classes, height, width]`
- Type: `float32`
- Values: Raw logits (apply argmax to get class predictions)

**Classes:**
- 0 = background
- 1 = iris
- 2 = pupil

---

## Demo / Inference

### Requirements

Install ONNX Runtime:

```bash
# CPU version
pip install onnxruntime

# GPU version (faster, requires CUDA)
pip install onnxruntime-gpu
```

### Webcam Demo

Run real-time segmentation on webcam:

```bash
python -m irispupilnet.demo \
  --model iris_pupil.onnx \
  --source 0 \
  --img-size 160 \
  --color false
```

**Controls:**
- `q` - Quit
- `s` - Save current frame

### Video File

Process a video file:

```bash
python -m irispupilnet.demo \
  --model iris_pupil.onnx \
  --source input_video.mp4 \
  --img-size 160 \
  --color false \
  --save-video output_video.mp4
```

**Controls:**
- `q` - Quit
- `s` - Save current frame
- `SPACE` - Pause/Resume

### Single Image

Process a single image:

```bash
python -m irispupilnet.demo \
  --model iris_pupil.onnx \
  --source image.jpg \
  --img-size 160 \
  --color false \
  --save-frames output/
```

### Image Directory

Process all images in a directory:

```bash
python -m irispupilnet.demo \
  --model iris_pupil.onnx \
  --source images/ \
  --img-size 160 \
  --color false \
  --save-frames output/
```

### Demo Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | **required** | Path to ONNX model file |
| `--source` | str | `0` | Input source (see below) |
| `--img-size` | int | `160` | Model input size (must match export) |
| `--color` | bool | `false` | Use RGB (true) or grayscale (false) |
| `--device` | str | `cpu` | Inference device: `cpu` or `cuda` |
| `--save-frames` | str | `null` | Directory to save frames (press 's') |
| `--save-video` | str | `null` | Path to save output video (video only) |

### Source Types

The `--source` parameter accepts:

1. **Webcam ID**: `0`, `1`, etc. (usually 0 is the default webcam)
2. **Video file**: `video.mp4`, `demo.avi`, etc.
3. **Image file**: `image.jpg`, `photo.png`, etc.
4. **Image directory**: `images/`, `dataset/test/`, etc.

---

## Examples

### Example 1: Grayscale Model Workflow

```bash
# Train grayscale model
python -m irispupilnet.train \
  --config irispupilnet/configs/train_bw_unet.yaml

# Export best checkpoint
python -m irispupilnet.export_onnx \
  --checkpoint runs/train_bw_unet/2025-11-19_15-30-45/best.pt \
  --output iris_pupil_bw.onnx \
  --img-size 160 \
  --in-channels 1 \
  --num-classes 3

# Run webcam demo
python -m irispupilnet.demo \
  --model iris_pupil_bw.onnx \
  --source 0 \
  --img-size 160 \
  --color false
```

### Example 2: RGB Model Workflow

```bash
# Train RGB model
python -m irispupilnet.train \
  --config irispupilnet/configs/example_rgb.yaml

# Export
python -m irispupilnet.export_onnx \
  --checkpoint runs/rgb_experiment/best.pt \
  --output iris_pupil_rgb.onnx \
  --model seresnext_unet \
  --img-size 160 \
  --in-channels 3 \
  --num-classes 3

# Run demo
python -m irispupilnet.demo \
  --model iris_pupil_rgb.onnx \
  --source 0 \
  --img-size 160 \
  --color true
```

### Example 3: Batch Process Video

Process a video and save frames with segmentation:

```bash
python -m irispupilnet.demo \
  --model iris_pupil_bw.onnx \
  --source input_video.mp4 \
  --img-size 160 \
  --color false \
  --save-video segmented_video.mp4 \
  --save-frames video_frames/
```

This will:
- Process the video frame by frame
- Save the output video with segmentation overlay
- Allow you to press 's' to save individual frames to `video_frames/`

---

## GPU Acceleration

To use GPU for faster inference:

1. Install ONNX Runtime GPU:
```bash
pip install onnxruntime-gpu
```

2. Run demo with `--device cuda`:
```bash
python -m irispupilnet.demo \
  --model iris_pupil.onnx \
  --source 0 \
  --device cuda
```

**Note:** Requires NVIDIA GPU with CUDA installed.

---

## Visualization

The demo shows:
- **Green overlay**: Iris region
- **Red overlay**: Pupil region
- **Green contour**: Iris boundary
- **Red contour**: Pupil boundary

The overlay transparency is set to 50% (alpha=0.5) by default.

---

## Troubleshooting

### "No module named onnxruntime"

Install ONNX Runtime:
```bash
pip install onnxruntime
```

### "Cannot open webcam"

Try a different webcam ID:
```bash
python -m irispupilnet.demo --model model.onnx --source 1
```

Or list available cameras:
```bash
# Linux
v4l2-ctl --list-devices

# macOS
system_profiler SPCameraDataType
```

### Export parameters don't match training

Make sure export parameters match your training config:
- `--img-size` must match training `img_size`
- `--in-channels` must match training `in_channels`
- `--num-classes` must match training `num_classes`
- `--model` must match training `model`

Check your training config file or use the config that was saved in the run directory:
```bash
cat runs/my_experiment/2025-11-19_15-30-45/config.yaml
```

### Slow inference

1. Use grayscale (`--color false`) instead of RGB (3x faster)
2. Use GPU (`--device cuda`) with onnxruntime-gpu
3. Reduce input resolution (but must match training)

---

## Integration with Other Tools

### Python Script

```python
from irispupilnet.demo import IrisPupilSegmentor
import cv2

# Initialize
segmentor = IrisPupilSegmentor(
    model_path="iris_pupil.onnx",
    img_size=160,
    color=False,
)

# Load image
image = cv2.imread("eye.jpg")

# Run inference
mask = segmentor.predict(image)  # Returns (H, W) with class labels

# Visualize
vis = segmentor.visualize(image, mask)
cv2.imwrite("result.jpg", vis)
```

### C++ / Other Languages

The exported ONNX model can be used with:
- **C++**: ONNX Runtime C++ API
- **C#**: ML.NET
- **JavaScript**: ONNX.js
- **Mobile**: ONNX Runtime Mobile (iOS/Android)

See [ONNX Runtime documentation](https://onnxruntime.ai/) for language-specific guides.

---

## Performance Tips

1. **Grayscale is faster**: Use `--color false` (3x faster than RGB)
2. **GPU acceleration**: Use `--device cuda` with onnxruntime-gpu
3. **Smaller models**: `unet_se_small` is faster than `seresnext_unet`
4. **Lower resolution**: Use smaller `--img-size` (but must match training)

**Typical performance:**
- CPU (grayscale, 160x160): ~30-50 FPS
- CPU (RGB, 160x160): ~10-20 FPS
- GPU (grayscale, 160x160): ~100+ FPS
- GPU (RGB, 160x160): ~50-80 FPS

*Performance varies by hardware (CPU/GPU model, RAM, etc.)*
