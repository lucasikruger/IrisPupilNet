import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_model

class SEBlock(nn.Module):
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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvSE(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConvSE(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if mismatch
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx//2, dx - dx//2, dy//2, dy - dy//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

@register_model("unet_se_small")
class UNetSESmall(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 3, base: int = 32):
        super().__init__()
        self.inc = DoubleConvSE(in_channels, base)   # -> base
        self.d1  = Down(base, base*2)                # -> 2b
        self.d2  = Down(base*2, base*4)              # -> 4b
        self.d3  = Down(base*4, base*8)              # -> 8b

        # up path (concat doubles channels)
        self.u1  = Up(base*8 + base*4, base*4)
        self.u2  = Up(base*4 + base*2, base*2)
        self.u3  = Up(base*2 + base,   base)

        self.out = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)      # b
        x2 = self.d1(x1)      # 2b
        x3 = self.d2(x2)      # 4b
        x4 = self.d3(x3)      # 8b
        x  = self.u1(x4, x3)  # 4b
        x  = self.u2(x,  x2)  # 2b
        x  = self.u3(x,  x1)  # b
        return self.out(x)
