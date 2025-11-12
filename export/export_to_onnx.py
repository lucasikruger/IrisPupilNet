"""
Export a trained model checkpoint to ONNX (NHWC format).

Usage:
  python export/export_to_onnx.py \
    --checkpoint /path/to/best.pt \
    --out /path/to/out.onnx \
    --size 160 \
    --classes 3 \
    --base 32 \
    --in-channels 1 \
    --model unet_se_small
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


def main():
    ap = argparse.ArgumentParser(description="Export checkpoint to ONNX (NHWC)")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--out", required=True, help="Output ONNX file path")
    ap.add_argument("--size", type=int, default=160, help="Input image size (H=W)")
    ap.add_argument("--classes", type=int, default=3, help="Number of output classes")
    ap.add_argument("--base", type=int, default=32, help="Base channel count (UNet scaling)")
    ap.add_argument("--in-channels", type=int, choices=[1, 3], default=1, help="Number of input channels")
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="unet_se_small",
                    help="Model registered in irispupilnet.models")
    args = ap.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")
    safe_classes = [pathlib.Path, pathlib.PosixPath]
    with torch.serialization.safe_globals(safe_classes):
        checkpoint = torch.load(str(ckpt_path), map_location=device)

    model_name = args.model
    model_kwargs = {
        "in_channels": args.in_channels,
        "n_classes": args.classes,
        "base": args.base,
    }
    ckpt_args = checkpoint.get("args", {})
    if isinstance(ckpt_args, dict):
        model_name = ckpt_args.get("model", model_name)
        model_kwargs["in_channels"] = ckpt_args.get("in_channels", model_kwargs["in_channels"])
        model_kwargs["n_classes"] = ckpt_args.get("num_classes", model_kwargs["n_classes"])
        model_kwargs["base"] = ckpt_args.get("base", model_kwargs["base"])

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name}. Registered: {list(MODEL_REGISTRY.keys())}")

    model_ctor = MODEL_REGISTRY[model_name]
    model = model_ctor(**model_kwargs)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("Loaded checkpoint with metadata")
        if "args" in checkpoint:
            print(f"  Checkpoint args: {checkpoint['args']}")
    else:
        state_dict = checkpoint
        print("Loaded state dict only")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    wrapped = NHWCWrapper(model)

    dummy_input = torch.randn(1, args.size, args.size, args.in_channels, device=device)
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

    print(f"Exported ONNX (NHWC) -> {args.out}")
    print(f"  Input shape:  [batch, {args.size}, {args.size}, {args.in_channels}]")
    print(f"  Output shape: [batch, {args.size}, {args.size}, {args.classes}]")


if __name__ == "__main__":
    main()
