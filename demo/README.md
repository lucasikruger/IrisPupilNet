# Webcam Demo

Real-time iris and pupil segmentation demo using webcam input.

## Features

- Face and eye detection using MediaPipe FaceMesh
- Real-time segmentation with ONNX model
- 4-panel visualization (left/right eyes, raw + overlay)
- Mirrored camera feed option

## Installation

```bash
pip install opencv-python mediapipe==0.10.14 onnxruntime numpy
```

Or install demo dependencies:
```bash
pip install -e ".[demo]"  # From repository root
```

## Usage

### Basic (Detection Only)

```bash
python webcam_demo.py
```

Shows eye detection bounding boxes without segmentation.

### With Segmentation Model

```bash
python webcam_demo.py --model ../checkpoints/model.onnx
```

### Disable Mirror Mode

```bash
python webcam_demo.py --model ../checkpoints/model.onnx --no-mirror
```

## Controls

- **q**: Quit
- **m**: Toggle mirror mode on/off

## Display Layout

```
┌─────────────────┬─────────────────┐
│   Left Eye      │   Right Eye     │
│   (Raw Crop)    │   (Raw Crop)    │
├─────────────────┼─────────────────┤
│ Left Eye + Seg  │ Right Eye + Seg │
│  (Overlay)      │  (Overlay)      │
└─────────────────┴─────────────────┘
```

### Segmentation Colors

- **Green overlay**: Iris region
- **Blue overlay**: Pupil region
- **No overlay**: Background

## Requirements

- Webcam (internal or external)
- Python 3.10+
- ~4GB RAM for ONNX inference (CPU)
- Optional: GPU for faster inference (use `onnxruntime-gpu`)

## Troubleshooting

### Camera Not Found

- Check camera permissions
- Try different camera index: modify `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Slow Inference

- Use smaller model (`--size 128` during training/export)
- Install `onnxruntime-gpu` for GPU acceleration
- Reduce camera resolution

### Face Not Detected

- Ensure good lighting
- Face camera directly
- Move closer to camera
- Avoid extreme angles

### Segmentation Quality Issues

- Model may not generalize to your camera/lighting
- Consider fine-tuning on data similar to your use case
- Adjust overlay transparency in code if needed

## Performance

- **Detection**: ~30 FPS (MediaPipe FaceMesh)
- **Segmentation**: ~15-20 FPS (ONNX CPU, 160×160 input)
- **Total**: ~12-15 FPS (combined pipeline)

GPU acceleration can achieve 60+ FPS.
