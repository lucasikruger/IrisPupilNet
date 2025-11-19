"""
Export trained IrisPupilNet models to ONNX format for deployment.

This script loads a PyTorch checkpoint and exports it to ONNX format
for use with ONNX Runtime, TensorRT, or other inference engines.

Usage:
    python -m irispupilnet.export_onnx \
        --checkpoint runs/experiment/best.pt \
        --output iris_pupil.onnx \
        --img-size 160 \
        --in-channels 1 \
        --num-classes 3

The exported ONNX model expects:
    Input: [batch, channels, height, width] with float32 in range [0, 1]
    Output: [batch, num_classes, height, width] with raw logits

To get segmentation masks, apply argmax on the class dimension (dim=1).
"""

import argparse
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from irispupilnet.models import MODEL_REGISTRY


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    model_name: str = "unet_se_small",
    img_size: int = 160,
    in_channels: int = 1,
    num_classes: int = 3,
    base: int = 32,
    opset_version: int = 14,
):
    """
    Export a trained model to ONNX format.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path: Path to save .onnx file
        model_name: Model architecture name (unet_se_small, seresnext_unet, etc.)
        img_size: Input image size (height and width, must be square)
        in_channels: Number of input channels (1=grayscale, 3=RGB)
        num_classes: Number of output classes (typically 3: bg/iris/pupil)
        base: Base channel count for model
        opset_version: ONNX opset version (14 is widely supported)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint with weights_only=False for compatibility with pathlib objects
    # This is safe because we're loading our own trained checkpoints
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Build model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    print(f"Building model: {model_name}")
    model_fn = MODEL_REGISTRY[model_name]
    model = model_fn(in_channels=in_channels, n_classes=num_classes, base=base)

    # Load weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Create wrapper model that converts NHWC -> NCHW -> model -> NCHW -> NHWC
    # This makes the exported model compatible with TensorFlow/MediaPipe demos
    class NHWCWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # Input: [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
            # Run model: [B, C, H, W] -> [B, num_classes, H, W]
            x = self.model(x)
            # Output: [B, num_classes, H, W] -> [B, H, W, num_classes]
            x = x.permute(0, 2, 3, 1)
            return x

    wrapped_model = NHWCWrapper(model)
    wrapped_model.eval()

    # Create dummy input in NHWC format
    dummy_input = torch.randn(1, img_size, img_size, in_channels)

    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input shape: [batch, {img_size}, {img_size}, {in_channels}] (NHWC format)")
    print(f"  Output shape: [batch, {img_size}, {img_size}, {num_classes}] (NHWC format)")

    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    print(f"✓ Model exported successfully to: {output_path}")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    except ImportError:
        print("⚠ onnx package not installed, skipping validation")
        print("  Install with: pip install onnx")
    except Exception as e:
        print(f"⚠ ONNX validation warning: {e}")

    # Show metadata from checkpoint
    if "epoch" in checkpoint:
        print(f"\nCheckpoint metadata:")
        print(f"  Epoch: {checkpoint['epoch']}")
        if "best_iou" in checkpoint:
            print(f"  Best IoU: {checkpoint['best_iou']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Export IrisPupilNet model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for ONNX model (.onnx file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet_se_small",
        help="Model architecture name",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=160,
        help="Input image size (height and width)",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        help="Number of input channels (1=grayscale, 3=RGB)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of output classes",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=32,
        help="Base channel count for model",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.model,
        img_size=args.img_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base=args.base,
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    main()
