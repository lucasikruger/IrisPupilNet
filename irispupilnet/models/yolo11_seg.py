"""
YOLO11 Segmentation Models for IrisPupilNet

This module integrates YOLO11 as a backbone for semantic segmentation.
Since YOLO11-seg is designed for instance segmentation (not semantic),
we use YOLO as a feature extractor + custom semantic segmentation head.

Architecture:
- YOLO11 backbone (layers 0-22): Feature extraction
- Custom decoder: Upsampling + skip connections
- Segmentation head: Final 1x1 conv to produce (B, n_classes, H, W) logits

Supported variants:
- yolo11n_seg (nano): Smallest, fastest
- yolo11s_seg (small): Balanced speed/accuracy
- yolo11m_seg (medium): Higher accuracy
- yolo11l_seg (large): Best accuracy
- yolo11_seg (auto): Automatically selects based on base parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional
import warnings

from . import register_model

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class SimpleDecoder(nn.Module):
    """
    Simple decoder for semantic segmentation.
    Upsamples features and produces class logits.
    """
    def __init__(self, feature_channels: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()

        # Progressive upsampling + refinement
        self.up1 = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # Final segmentation head
        self.seg_head = nn.Conv2d(hidden_dim // 4, n_classes, 1)

    def forward(self, x):
        x = self.up1(x)  # 2x
        x = self.up2(x)  # 4x
        x = self.up3(x)  # 8x
        return self.seg_head(x)


class YOLO11BackboneSegmentation(nn.Module):
    """
    YOLO11 backbone + custom semantic segmentation head.

    Uses YOLO11 as a pretrained feature extractor and adds a simple
    decoder to produce semantic segmentation logits compatible with
    IrisPupilNet's training pipeline.

    Args:
        variant: YOLO11 variant ('n', 's', 'm', 'l')
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        n_classes: Number of output classes (typically 3: background, iris, pupil)
        pretrained: Whether to load pretrained COCO weights
        freeze_backbone: Whether to freeze YOLO backbone weights
    """

    def __init__(
        self,
        variant: Literal["n", "s", "m", "l"],
        in_channels: int = 3,
        n_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is required for YOLO11 models. "
                "Install with: pip install ultralytics>=8.3.0"
            )

        self.variant = variant
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.freeze_backbone = freeze_backbone

        # Load YOLO model
        model_name = f"yolo11{variant}-seg.pt" if pretrained else f"yolo11{variant}-seg.yaml"

        print(f"Loading YOLO11{variant}-seg model (pretrained={pretrained})...")
        yolo = YOLO(model_name)
        pytorch_model = yolo.model

        # Keep the full YOLO model but we'll intercept features before the Segment head
        self.yolo_model = pytorch_model

        # Add input adapter for non-RGB inputs
        # YOLO's architecture expects 3 channels internally, so we convert
        self.input_adapter = None
        if in_channels != 3:
            self.input_adapter = self._create_input_adapter(in_channels)

        # Use a forward hook to capture features from layer 22 (last before Segment head)
        self.features = None

        def hook_fn(module, input, output):
            self.features = output

        # Register hook on layer 22 (last backbone layer before detection)
        self.yolo_model.model[22].register_forward_hook(hook_fn)

        # Determine output channels from backbone by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 160, 160)  # YOLO always expects 3 channels
            self.yolo_model.eval()
            _ = self.yolo_model(dummy_input)  # Run forward to trigger hook
            feature_channels = self.features.shape[1]
            feature_spatial = self.features.shape[2]

        print(f"  Backbone output: {feature_channels} channels, {feature_spatial}x{feature_spatial} spatial")

        # Create custom decoder
        # Calculate hidden dim based on variant
        hidden_dim_map = {'n': 64, 's': 96, 'm': 128, 'l': 160}
        hidden_dim = hidden_dim_map.get(variant, 96)

        self.decoder = SimpleDecoder(feature_channels, n_classes, hidden_dim)

        # Optionally freeze backbone
        if freeze_backbone:
            print(f"  Freezing YOLO11 backbone weights")
            for param in self.yolo_model.parameters():
                param.requires_grad = False

        print(f"✓ YOLO11{variant} backbone + semantic head ready")

    def _create_input_adapter(self, in_channels: int):
        """
        Create adapter to convert non-RGB inputs to 3-channel format expected by YOLO.

        YOLO's architecture has hardcoded expectations for 3-channel inputs in various
        places (concatenations, etc.), so instead of modifying the first conv layer,
        we add a preprocessing step that converts grayscale to RGB-like format.
        """
        if in_channels == 1:
            # For grayscale: use a 1x1 conv to learn the optimal channel expansion
            # Initialize with identity-like weights (repeat grayscale 3 times)
            adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            # Initialize to repeat the channel 3 times
            with torch.no_grad():
                adapter.weight.fill_(1.0 / 3.0)  # Average of 3 channels will equal original
            print(f"  Created input adapter: {in_channels} channels → 3 channels (learnable)")
            return adapter
        else:
            # For other channel counts, use a simple conv
            adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
            print(f"  Created input adapter: {in_channels} channels → 3 channels")
            return adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning semantic segmentation logits.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            logits: Output tensor (B, n_classes, H, W) with raw logits
        """
        # Apply input adapter if needed (convert to 3 channels for YOLO)
        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # Run through YOLO backbone (hook will capture features)
        _ = self.yolo_model(x)

        # Use captured features from hook
        features = self.features

        # Decode to semantic segmentation logits
        logits = self.decoder(features)

        # Ensure output matches input spatial dimensions
        # The decoder upsamples from the backbone output, but we need to match original input
        target_size = x.shape[2:]  # Match input H, W
        if logits.shape[2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

        return logits

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.yolo_model.parameters():
            param.requires_grad = True
        print("✓ YOLO11 backbone unfrozen")

    def freeze_backbone(self):
        """Freeze backbone weights."""
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        print("✓ YOLO11 backbone frozen")


# ============================================================================
# Model Registry Functions
# ============================================================================

@register_model("yolo11n_seg")
def yolo11n_seg(in_channels: int = 3, n_classes: int = 3, base: int = 16):
    """
    YOLO11 Nano segmentation (smallest, fastest).

    Best for: Quick experiments, limited compute
    Params: ~2.7M (vs ~5.9M for full YOLO11n-seg with detection head)
    """
    return YOLO11BackboneSegmentation(
        variant="n",
        in_channels=in_channels,
        n_classes=n_classes,
        pretrained=True,
        freeze_backbone=False
    )


@register_model("yolo11s_seg")
def yolo11s_seg(in_channels: int = 3, n_classes: int = 3, base: int = 32):
    """
    YOLO11 Small segmentation (balanced speed/accuracy).

    Best for: General use, good balance
    Params: ~9.4M
    """
    return YOLO11BackboneSegmentation(
        variant="s",
        in_channels=in_channels,
        n_classes=n_classes,
        pretrained=True,
        freeze_backbone=False
    )


@register_model("yolo11m_seg")
def yolo11m_seg(in_channels: int = 3, n_classes: int = 3, base: int = 64):
    """
    YOLO11 Medium segmentation (higher accuracy).

    Best for: When accuracy is priority, ample compute
    Params: ~20.1M
    """
    return YOLO11BackboneSegmentation(
        variant="m",
        in_channels=in_channels,
        n_classes=n_classes,
        pretrained=True,
        freeze_backbone=False
    )


@register_model("yolo11l_seg")
def yolo11l_seg(in_channels: int = 3, n_classes: int = 3, base: int = 128):
    """
    YOLO11 Large segmentation (best accuracy).

    Best for: Maximum accuracy, substantial compute
    Params: ~25.3M
    """
    return YOLO11BackboneSegmentation(
        variant="l",
        in_channels=in_channels,
        n_classes=n_classes,
        pretrained=True,
        freeze_backbone=False
    )


@register_model("yolo11_seg")
def yolo11_seg(in_channels: int = 3, n_classes: int = 3, base: int = 32):
    """
    YOLO11 segmentation with auto-selected variant based on base parameter.

    Base parameter mapping:
    - base=16  → nano (fastest)
    - base=32  → small (default, balanced)
    - base=64  → medium (more accurate)
    - base=128 → large (most accurate)
    """
    variant_map = {
        16: "n",   # nano
        32: "s",   # small (default)
        64: "m",   # medium
        128: "l",  # large
    }

    variant = variant_map.get(base, "s")  # Default to small if unknown base

    if base not in variant_map:
        warnings.warn(
            f"Unknown base={base} for YOLO11, defaulting to 's' (small). "
            f"Supported values: {list(variant_map.keys())}"
        )

    return YOLO11BackboneSegmentation(
        variant=variant,
        in_channels=in_channels,
        n_classes=n_classes,
        pretrained=True,
        freeze_backbone=False
    )
