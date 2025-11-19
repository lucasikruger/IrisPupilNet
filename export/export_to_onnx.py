"""
Export a trained model checkpoint to ONNX (NHWC format).

Simple usage (automatic config from checkpoint):
  python export/export_to_onnx.py \
    --checkpoint runs/experiment/best.pt \
    --out model.onnx

Advanced usage (override settings):
  python export/export_to_onnx.py \
    --checkpoint runs/experiment/best.pt \
    --out model.onnx \
    --size 224 \
    --in-channels 3
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from pathlib import Path

def _ensure_project_root():
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_project_root()

import torch

from irispupilnet.models import MODEL_REGISTRY


class NHWCWrapper(torch.nn.Module):
    """Wrap a NCHW model so ONNX receives/outputs NHWC tensors."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.permute(0, 3, 1, 2)
        y = self.net(x)
        return y.permute(0, 2, 3, 1)


def load_checkpoint_config(checkpoint_path: Path) -> dict:
    """
    Load model configuration from checkpoint.

    Returns:
        dict with keys: model, in_channels, num_classes, base, img_size
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device("cpu")
    safe_classes = [Path, pathlib.Path, pathlib.PosixPath, type(None)]

    with torch.serialization.safe_globals(safe_classes):
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Extract config from checkpoint args
    config = {}
    if isinstance(checkpoint, dict) and "args" in checkpoint:
        ckpt_args = checkpoint["args"]
        config = {
            "model": ckpt_args.get("model", "unet_se_small"),
            "in_channels": ckpt_args.get("in_channels", 1),
            "num_classes": ckpt_args.get("num_classes", 3),
            "base": ckpt_args.get("base", 32),
            "img_size": ckpt_args.get("img_size", 160),
        }
        print(f"✓ Loaded config from checkpoint:")
        print(f"  Model: {config['model']}")
        print(f"  Input channels: {config['in_channels']}")
        print(f"  Classes: {config['num_classes']}")
        print(f"  Base: {config['base']}")
        print(f"  Image size: {config['img_size']}")
    else:
        # Fallback defaults if no args in checkpoint
        print("⚠ No args found in checkpoint, using defaults")
        config = {
            "model": "unet_se_small",
            "in_channels": 1,
            "num_classes": 3,
            "base": 32,
            "img_size": 160,
        }

    config["checkpoint"] = checkpoint
    return config


def main():
    ap = argparse.ArgumentParser(description="Export checkpoint to ONNX (NHWC)")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--out", required=True, help="Output ONNX file path")

    # Optional overrides (will use checkpoint config if not provided)
    ap.add_argument("--size", type=int, default=None, help="Override image size (H=W)")
    ap.add_argument("--classes", type=int, default=None, help="Override number of output classes")
    ap.add_argument("--base", type=int, default=None, help="Override base channel count")
    ap.add_argument("--in-channels", type=int, choices=[1, 3], default=None, help="Override input channels")
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default=None,
                    help="Override model architecture")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load config from checkpoint
    config = load_checkpoint_config(ckpt_path)
    checkpoint = config.pop("checkpoint")

    # Apply CLI overrides
    if args.model is not None:
        config["model"] = args.model
        print(f"  Override: model = {args.model}")
    if args.in_channels is not None:
        config["in_channels"] = args.in_channels
        print(f"  Override: in_channels = {args.in_channels}")
    if args.classes is not None:
        config["num_classes"] = args.classes
        print(f"  Override: num_classes = {args.classes}")
    if args.base is not None:
        config["base"] = args.base
        print(f"  Override: base = {args.base}")
    if args.size is not None:
        config["img_size"] = args.size
        print(f"  Override: img_size = {args.size}")

    # Build model
    model_name = config["model"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name}. Registered: {list(MODEL_REGISTRY.keys())}")

    model_ctor = MODEL_REGISTRY[model_name]
    model = model_ctor(
        in_channels=config["in_channels"],
        n_classes=config["num_classes"],
        base=config["base"]
    )

    # Load weights
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("✓ Model weights loaded")

    # Wrap for NHWC
    wrapped = NHWCWrapper(model)

    # Export to ONNX
    device = torch.device("cpu")
    img_size = config["img_size"]
    in_channels = config["in_channels"]
    num_classes = config["num_classes"]

    dummy_input = torch.randn(1, img_size, img_size, in_channels, device=device)

    print(f"\nExporting to ONNX...")
    torch.onnx.export(
        wrapped,
        dummy_input,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"\n✓ Successfully exported ONNX model!")
    print(f"  Output file: {args.out}")
    print(f"  Input shape:  [batch, {img_size}, {img_size}, {in_channels}] (NHWC)")
    print(f"  Output shape: [batch, {img_size}, {img_size}, {num_classes}] (NHWC)")
    print(f"\nUsage with demo:")
    print(f"  python demo/webcam_demo.py --model {args.out}")


if __name__ == "__main__":
    main()
