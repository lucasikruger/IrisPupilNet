# export_to_onnx.py
"""
Export a trained UNetSESmall (.pt checkpoint) to ONNX (NHWC format).

FIXED VERSION: Now includes SE (Squeeze-and-Excitation) blocks to match training architecture.

Exports NHWC layout: input [1,H,W,C] -> output [1,H,W,3] logits. Default C=1 (grayscale).
This format is convenient for browser/ONNX Runtime deployment.

Usage:
  python export_to_onnx.py \
    --checkpoint /path/to/unet_mobius_best.pt \
    --out /path/to/unet_mobius_best.onnx \
    --size 160 --classes 3 --base 32
"""

import argparse
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Model definition (matches training/models/unet_se.py) ----------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)      # [B, C]
        s = self.fc(s).view(b, c, 1, 1)  # [B, C, 1, 1]
        return x * s


class DoubleConvSE(nn.Module):
    """Double convolution with Squeeze-and-Excitation attention"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch, reduction=16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.se(x)
        return x


class Down(nn.Module):
    """Downsampling block with max pooling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvSE(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling block with skip connections"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConvSE(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if shapes mismatch
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx//2, dx - dx//2, dy//2, dy - dy//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetSESmall(nn.Module):
    """UNet with Squeeze-and-Excitation blocks"""
    def __init__(self, in_channels: int = 3, n_classes: int = 3, base: int = 32):
        super().__init__()
        self.inc = DoubleConvSE(in_channels, base)   # -> base
        self.d1  = Down(base, base*2)                # -> 2*base
        self.d2  = Down(base*2, base*4)              # -> 4*base
        self.d3  = Down(base*4, base*8)              # -> 8*base

        # Upsampling path (concat doubles channels before conv)
        self.u1  = Up(base*8 + base*4, base*4)       # 8b + 4b -> 4b
        self.u2  = Up(base*4 + base*2, base*2)       # 4b + 2b -> 2b
        self.u3  = Up(base*2 + base,   base)         # 2b + 1b -> 1b

        self.out = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)      # base
        x2 = self.d1(x1)      # 2*base
        x3 = self.d2(x2)      # 4*base
        x4 = self.d3(x3)      # 8*base

        x  = self.u1(x4, x3)  # 4*base
        x  = self.u2(x,  x2)  # 2*base
        x  = self.u3(x,  x1)  # base
        return self.out(x)


# Wrapper to export NHWC layout (browser/ONNX Runtime friendly)
class NHWCWrapper(nn.Module):
    """Wrapper to convert between NHWC (input/output) and NCHW (model)"""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x_nhwc):
        x = x_nhwc.permute(0, 3, 1, 2)    # NHWC -> NCHW
        y = self.net(x)                    # Model expects NCHW
        return y.permute(0, 2, 3, 1)       # NCHW -> NHWC


# ---------- Export Function ----------

def main():
    ap = argparse.ArgumentParser(description="Export UNetSESmall checkpoint to ONNX")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--out", required=True, help="Output ONNX file path")
    ap.add_argument("--size", type=int, default=160, help="Input image size (H=W, default 160)")
    ap.add_argument("--classes", type=int, default=3, help="Number of output classes (default 3)")
    ap.add_argument("--base", type=int, default=32, help="Base number of channels (default 32)")
    ap.add_argument("--in-channels", type=int, choices=[1,3], default=1, help="Number of input channels (default 1 for grayscale)")
    args = ap.parse_args()

    device = torch.device("cpu")

    # Load model
    model = UNetSESmall(in_channels=args.in_channels, n_classes=args.classes, base=args.base)

    # Load checkpoint (supports both state_dict only and full checkpoint)
    safe_classes = [pathlib.Path, pathlib.PosixPath]
    with torch.serialization.safe_globals(safe_classes):
        checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        # Full checkpoint with metadata
        state_dict = checkpoint["model"]
        print(f"Loaded checkpoint with metadata")
        if "args" in checkpoint:
            print(f"  Checkpoint args: {checkpoint['args']}")
    else:
        # State dict only
        state_dict = checkpoint
        print(f"Loaded state dict")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Model loaded successfully")

    # Wrap for NHWC export
    wrapped = NHWCWrapper(model)
    dummy_input = torch.randn(1, args.size, args.size, args.in_channels, device=device)

    # Export to ONNX
    torch.onnx.export(
        wrapped,
        dummy_input,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=17,
        do_constant_folding=True
    )

    print(f"✓ Exported ONNX (NHWC format) to: {args.out}")
    print(f"  Input shape:  [batch, {args.size}, {args.size}, {args.in_channels}]")
    print(f"  Output shape: [batch, {args.size}, {args.size}, {args.classes}]")


if __name__ == "__main__":
    main()
