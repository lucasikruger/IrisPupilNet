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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class SEResNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        cardinality: int = 16,
        base_width: int = 4,
        reduction: int = 16,
    ):
        super().__init__()
        width = int(out_channels * (base_width / 64.)) * cardinality

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(identity)
        return F.relu(out, inplace=True)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cardinality: int = 16,
        base_width: int = 4,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = SEResNeXtBlock(
            out_channels + skip_channels,
            out_channels,
            stride=1,
            cardinality=cardinality,
            base_width=base_width,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


@register_model("seresnext_unet")
class SEResNeXtUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 3,
        base: int = 32,
        cardinality: int = 16,
        base_width: int = 4,
    ):
        super().__init__()
        self.inc = SEResNeXtBlock(in_channels, base, stride=1, cardinality=cardinality, base_width=base_width)
        self.down1 = nn.MaxPool2d(2)
        self.layer1 = SEResNeXtBlock(base, base * 2, stride=1, cardinality=cardinality, base_width=base_width)
        self.down2 = nn.MaxPool2d(2)
        self.layer2 = SEResNeXtBlock(base * 2, base * 4, stride=1, cardinality=cardinality, base_width=base_width)
        self.down3 = nn.MaxPool2d(2)
        self.layer3 = SEResNeXtBlock(base * 4, base * 8, stride=1, cardinality=cardinality, base_width=base_width)
        self.down4 = nn.MaxPool2d(2)
        self.layer4 = SEResNeXtBlock(base * 8, base * 16, stride=1, cardinality=cardinality, base_width=base_width)

        self.up1 = UpBlock(base * 16, base * 8, base * 8, cardinality=cardinality, base_width=base_width)
        self.up2 = UpBlock(base * 8, base * 4, base * 4, cardinality=cardinality, base_width=base_width)
        self.up3 = UpBlock(base * 4, base * 2, base * 2, cardinality=cardinality, base_width=base_width)
        self.up4 = UpBlock(base * 2, base, base, cardinality=cardinality, base_width=base_width)

        self.out_conv = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.layer1(self.down1(x0))
        x2 = self.layer2(self.down2(x1))
        x3 = self.layer3(self.down3(x2))
        x4 = self.layer4(self.down4(x3))

        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)
        u4 = self.up4(u3, x0)
        return self.out_conv(u4)
