# export_to_onnx.py
"""
Export a trained UNetSmall (.pt state_dict) to ONNX (NHWC).

Matches the training architecture you posted:
- Up blocks receive (low + skip) channels after concat.
- Exports NHWC: input [1,H,W,3] -> output [1,H,W,3] logits.

Usage:
  python export_to_onnx.py \
    --checkpoint /path/to/unet_mobius_best.pt \
    --out /path/to/unet_mobius_best.onnx \
    --size 160 --classes 3 --base 32
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Model definition (exactly as in training) ----------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    """
    in_ch is already (low_level_ch + skip_ch) because we concatenate first.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if shapes mismatch
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([skip, x], dim=1)   # concat first -> channels = in_ch
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, n_classes=3, base=32):
        super().__init__()
        self.inc = DoubleConv(3, base)             # -> base
        self.d1  = Down(base, base*2)              # -> base*2
        self.d2  = Down(base*2, base*4)            # -> base*4
        self.d3  = Down(base*4, base*8)            # -> base*8

        # after upsample: concat with skip => in_ch must be sum of both
        self.u1  = Up(base*8 + base*4, base*4)     # (x4 up base*8) + (skip x3 base*4)
        self.u2  = Up(base*4 + base*2, base*2)     # (x up base*4) + (skip x2 base*2)
        self.u3  = Up(base*2 + base,     base)     # (x up base*2) + (skip x1 base)

        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)    # base
        x2 = self.d1(x1)    # base*2
        x3 = self.d2(x2)    # base*4
        x4 = self.d3(x3)    # base*8

        x  = self.u1(x4, x3)  # -> base*4
        x  = self.u2(x,  x2)  # -> base*2
        x  = self.u3(x,  x1)  # -> base
        return self.out(x)

# Wrap to export NHWC (browser/ORT convenience)
class NHWCWrapper(nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x_nhwc):
        x = x_nhwc.permute(0,3,1,2)    # NHWC -> NCHW
        y = self.net(x)                # logits NCHW
        return y.permute(0,2,3,1)      # back to NHWC

# ---------- Export ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pt state_dict (torch.save(model.state_dict()))")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--size", type=int, default=160, help="Input H=W (default 160)")
    ap.add_argument("--classes", type=int, default=3, help="Number of classes (default 3)")
    ap.add_argument("--base", type=int, default=32, help="Base channels (default 32)")
    args = ap.parse_args()

    device = torch.device("cpu")
    model = UNetSmall(n_classes=args.classes, base=args.base)
    # Load your state_dict
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    wrapped = NHWCWrapper(model)
    dummy = torch.randn(1, args.size, args.size, 3, device=device)

    torch.onnx.export(
        wrapped, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True
    )
    print(f"Exported ONNX (NHWC) -> {args.out}")

if __name__ == "__main__":
    main()
