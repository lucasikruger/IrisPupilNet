"""
Tests for YOLO11 segmentation models integration.

These tests verify that YOLO11 models:
1. Are properly registered in the model registry
2. Produce correct output shapes
3. Work with both grayscale and RGB inputs
4. Are compatible with CrossEntropyLoss
5. Handle different input sizes correctly
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from irispupilnet.models import MODEL_REGISTRY


def test_yolo11_models_registered():
    """Test that all YOLO11 variants are registered."""
    expected_models = [
        "yolo11n_seg",
        "yolo11s_seg",
        "yolo11m_seg",
        "yolo11l_seg",
        "yolo11_seg"  # auto-select variant
    ]
    for model_name in expected_models:
        assert model_name in MODEL_REGISTRY, f"{model_name} not registered"
        print(f"✓ {model_name} is registered")


@pytest.mark.parametrize("model_name,expected_variant", [
    ("yolo11n_seg", "n"),
    ("yolo11s_seg", "s"),
    ("yolo11m_seg", "m"),
    ("yolo11l_seg", "l"),
])
def test_yolo11_output_shape(model_name, expected_variant):
    """Test YOLO11 output shape matches expected format."""
    model_fn = MODEL_REGISTRY[model_name]

    # Use small input size for faster testing
    batch_size = 2
    img_size = 160  # Must be multiple of 32 for YOLO

    # Test with grayscale (faster)
    model = model_fn(in_channels=1, n_classes=3, base=32)

    # Test input
    x = torch.randn(batch_size, 1, img_size, img_size)

    # Forward pass in eval mode
    model.eval()
    with torch.no_grad():
        logits = model(x)

    # Verify output shape
    expected_shape = (batch_size, 3, img_size, img_size)
    assert logits.shape == expected_shape, \
        f"{model_name}: Expected shape {expected_shape}, got {logits.shape}"

    # Verify output is logits (not probabilities)
    # Logits can be any real number, probabilities are [0, 1]
    # If all values are in [0,1], it's likely probabilities (bad)
    has_negative = (logits < 0).any().item()
    has_large = (logits > 1).any().item()
    assert has_negative or has_large, \
        f"{model_name}: Output appears to be probabilities, expected raw logits"

    print(f"✓ {model_name} produces correct output shape: {logits.shape}")


def test_yolo11_grayscale_input():
    """Test YOLO11 works with grayscale input."""
    model = MODEL_REGISTRY["yolo11n_seg"](in_channels=1, n_classes=3, base=16)
    x = torch.randn(2, 1, 160, 160)

    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 3, 160, 160), \
        f"Expected shape (2, 3, 160, 160), got {output.shape}"
    assert output.shape[1] == 3, "Expected 3 output classes"

    print(f"✓ Grayscale input works correctly: {output.shape}")


def test_yolo11_rgb_input():
    """Test YOLO11 works with RGB input."""
    model = MODEL_REGISTRY["yolo11s_seg"](in_channels=3, n_classes=3, base=32)
    x = torch.randn(2, 3, 160, 160)

    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 3, 160, 160), \
        f"Expected shape (2, 3, 160, 160), got {output.shape}"
    assert output.shape[1] == 3, "Expected 3 output classes"

    print(f"✓ RGB input works correctly: {output.shape}")


@pytest.mark.parametrize("size", [128, 160, 192, 224, 256])
def test_yolo11_different_sizes(size):
    """Test YOLO11 with different input sizes (multiples of 32)."""
    model = MODEL_REGISTRY["yolo11n_seg"](in_channels=1, n_classes=3, base=16)

    x = torch.randn(1, 1, size, size)
    model.eval()
    with torch.no_grad():
        output = model(x)

    expected_shape = (1, 3, size, size)
    assert output.shape == expected_shape, \
        f"Size {size}: expected {expected_shape}, got {output.shape}"

    print(f"✓ Size {size}x{size} works: output {output.shape}")


def test_yolo11_base_parameter_mapping():
    """Test that base parameter correctly maps to YOLO variants."""
    test_cases = [
        (16, "yolo11n_seg"),   # nano
        (32, "yolo11s_seg"),   # small
        (64, "yolo11m_seg"),   # medium
        (128, "yolo11l_seg"),  # large
    ]

    for base, expected_model in test_cases:
        model_fn = MODEL_REGISTRY[expected_model]
        model = model_fn(in_channels=1, n_classes=3, base=base)
        assert model is not None, f"Failed to create {expected_model} with base={base}"

        # Verify the variant is correct
        if hasattr(model, 'variant'):
            variant_map = {16: 'n', 32: 's', 64: 'm', 128: 'l'}
            expected_variant = variant_map[base]
            assert model.variant == expected_variant, \
                f"Base {base}: expected variant '{expected_variant}', got '{model.variant}'"

        print(f"✓ base={base} correctly creates {expected_model}")


def test_yolo11_auto_variant_selection():
    """Test auto variant selection based on base parameter."""
    # Test the generic yolo11_seg model with different base values
    base_to_variant = {
        16: 'n',
        32: 's',
        64: 'm',
        128: 'l',
    }

    for base, expected_variant in base_to_variant.items():
        model = MODEL_REGISTRY["yolo11_seg"](in_channels=1, n_classes=3, base=base)
        assert hasattr(model, 'variant'), "Model should have variant attribute"
        assert model.variant == expected_variant, \
            f"Base {base}: expected variant '{expected_variant}', got '{model.variant}'"

        print(f"✓ Auto-select with base={base} → variant '{expected_variant}'")


def test_yolo11_integration_with_loss():
    """Test YOLO11 output is compatible with CrossEntropyLoss."""
    model = MODEL_REGISTRY["yolo11n_seg"](in_channels=1, n_classes=3, base=16)
    loss_fn = nn.CrossEntropyLoss()

    # Create dummy data
    x = torch.randn(2, 1, 160, 160)
    y = torch.randint(0, 3, (2, 160, 160))  # Class indices

    # Forward pass in train mode
    model.train()
    logits = model(x)

    # Compute loss (should not raise)
    loss = loss_fn(logits, y)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be inf"

    # Backward pass (should not raise)
    loss.backward()

    # Verify gradients exist
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "Model should have gradients after backward pass"

    print(f"✓ CrossEntropyLoss integration works (loss={loss.item():.4f})")


def test_yolo11_parameter_count():
    """Test that YOLO11 models have reasonable parameter counts."""
    model = MODEL_REGISTRY["yolo11n_seg"](in_channels=1, n_classes=3, base=16)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0, "Model should have parameters"
    assert trainable_params > 0, "Model should have trainable parameters"
    assert trainable_params <= total_params, "Trainable params should <= total params"

    # YOLO11n should have roughly 2-3M parameters (backbone only, no detection head)
    # Allow range of 1M-10M for flexibility
    assert 1_000_000 <= total_params <= 10_000_000, \
        f"YOLO11n parameter count {total_params} seems unusual"

    print(f"✓ Parameter count: {total_params:,} total, {trainable_params:,} trainable")


def test_yolo11_freeze_backbone():
    """Test backbone freezing functionality."""
    model = MODEL_REGISTRY["yolo11n_seg"](in_channels=1, n_classes=3, base=16)

    # Initially all params should be trainable
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    assert trainable_before == total, "Initially all params should be trainable"

    # Freeze backbone
    if hasattr(model, 'freeze_backbone'):
        model.freeze_backbone()

        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # After freezing, some params should be frozen
        assert trainable_after < trainable_before, \
            "After freezing, trainable params should decrease"

        # Decoder params should still be trainable
        decoder_trainable = sum(
            p.numel() for p in model.decoder.parameters() if p.requires_grad
        )
        assert decoder_trainable > 0, "Decoder should remain trainable"

        print(f"✓ Backbone freezing works: {trainable_before:,} → {trainable_after:,} trainable")

        # Test unfreezing
        model.unfreeze_backbone()
        trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_unfrozen == total, "After unfreezing, all params should be trainable"

        print(f"✓ Backbone unfreezing works: {trainable_after:,} → {trainable_unfrozen:,} trainable")


@pytest.mark.slow
def test_yolo11_pretrained_weights_load():
    """
    Test that pretrained weights are loaded correctly.
    Marked as slow since it downloads weights (~6MB for nano).
    """
    # This should download pretrained YOLO11n-seg weights
    model = MODEL_REGISTRY["yolo11n_seg"](in_channels=3, n_classes=3, base=16)

    # Verify model has been initialized with non-zero weights
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model has no parameters"

    # Check that backbone weights are not all zeros (indicating pretrained weights loaded)
    if hasattr(model, 'backbone'):
        backbone_params = list(model.backbone.parameters())
        if backbone_params:
            first_param = backbone_params[0]
            assert not torch.allclose(first_param, torch.zeros_like(first_param)), \
                "Backbone params appear to be all zeros (pretrained weights may not have loaded)"

    print(f"✓ Pretrained weights loaded successfully ({total_params:,} params)")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
