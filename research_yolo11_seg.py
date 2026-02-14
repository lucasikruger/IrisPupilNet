"""
Research script to explore YOLO11 segmentation model structure.
This helps us understand how to extract raw segmentation logits for integration.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO

print("=" * 80)
print("YOLO11 Segmentation Model Research")
print("=" * 80)

# Load YOLO11 nano segmentation model
print("\n1. Loading YOLO11n-seg model...")
try:
    model = YOLO("yolo11n-seg.pt")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Examine model structure
print("\n2. Model structure:")
print(f"   Type: {type(model)}")
print(f"   Has .model attribute: {hasattr(model, 'model')}")

if hasattr(model, 'model'):
    pytorch_model = model.model
    print(f"   PyTorch model type: {type(pytorch_model)}")
    print(f"   Has .model attribute: {hasattr(pytorch_model, 'model')}")

    if hasattr(pytorch_model, 'model'):
        print(f"\n3. Model layers (pytorch_model.model):")
        for i, layer in enumerate(pytorch_model.model):
            print(f"   [{i}] {type(layer).__name__}")
            if hasattr(layer, 'nc'):
                print(f"       ├─ nc (num_classes): {layer.nc}")
            if hasattr(layer, 'nm'):
                print(f"       ├─ nm (num_masks): {layer.nm}")
            if hasattr(layer, 'npr'):
                print(f"       └─ npr (num_protos): {layer.npr}")

# Test forward pass
print("\n4. Testing forward pass:")
test_input = torch.randn(1, 3, 160, 160)
print(f"   Input shape: {test_input.shape}")

# Try different modes
print("\n   Testing train mode:")
pytorch_model.train()
try:
    output_train = pytorch_model(test_input)
    print(f"   ✓ Train output type: {type(output_train)}")
    if isinstance(output_train, (list, tuple)):
        print(f"     Number of outputs: {len(output_train)}")
        for i, out in enumerate(output_train):
            if isinstance(out, torch.Tensor):
                print(f"     Output[{i}] shape: {out.shape}")
            else:
                print(f"     Output[{i}] type: {type(out)}")
    elif isinstance(output_train, torch.Tensor):
        print(f"     Output shape: {output_train.shape}")
except Exception as e:
    print(f"   ✗ Error in train mode: {e}")

print("\n   Testing eval mode:")
pytorch_model.eval()
try:
    with torch.no_grad():
        output_eval = pytorch_model(test_input)
    print(f"   ✓ Eval output type: {type(output_eval)}")
    if isinstance(output_eval, (list, tuple)):
        print(f"     Number of outputs: {len(output_eval)}")
        for i, out in enumerate(output_eval):
            if isinstance(out, torch.Tensor):
                print(f"     Output[{i}] shape: {out.shape}")
            else:
                print(f"     Output[{i}] type: {type(out)}")
    elif isinstance(output_eval, torch.Tensor):
        print(f"     Output shape: {output_eval.shape}")
except Exception as e:
    print(f"   ✗ Error in eval mode: {e}")

# Examine the segmentation head in detail
print("\n5. Examining segmentation head (last layer):")
if hasattr(pytorch_model, 'model') and len(pytorch_model.model) > 0:
    seg_head = pytorch_model.model[-1]
    print(f"   Type: {type(seg_head).__name__}")
    print(f"   Attributes:")
    for attr in dir(seg_head):
        if not attr.startswith('_'):
            val = getattr(seg_head, attr)
            if not callable(val):
                print(f"     {attr}: {val if not isinstance(val, torch.Tensor) else f'Tensor{tuple(val.shape)}'}")

# Check for forward hooks capability
print("\n6. Checking intermediate layers:")
print(f"   Total layers: {len(pytorch_model.model)}")
print(f"   Can use forward hooks: Yes (standard PyTorch nn.Module)")

# Test if we can access proto masks (for segmentation)
print("\n7. Segmentation-specific components:")
if hasattr(pytorch_model, 'model'):
    for i, layer in enumerate(pytorch_model.model):
        layer_name = type(layer).__name__
        if 'Segment' in layer_name or 'Proto' in layer_name or 'Detect' in layer_name:
            print(f"   [{i}] {layer_name}")
            if hasattr(layer, 'cv2'):
                print(f"       ├─ cv2 (class prediction convs)")
            if hasattr(layer, 'cv3'):
                print(f"       ├─ cv3 (box prediction convs)")
            if hasattr(layer, 'cv4'):
                print(f"       └─ cv4 (mask/proto convs)")

print("\n" + "=" * 80)
print("Research complete!")
print("=" * 80)
