"""
Detailed research to understand YOLO11 segmentation mask generation.
Goal: Find how to extract raw (B, num_classes, H, W) logits.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
import torch.nn.functional as F

print("=" * 80)
print("YOLO11 Segmentation - Detailed Mask Extraction Research")
print("=" * 80)

# Load model
model = YOLO("yolo11n-seg.pt")
pytorch_model = model.model
seg_head = pytorch_model.model[-1]

print(f"\n1. Segment head details:")
print(f"   nc (num_classes): {seg_head.nc}")
print(f"   nm (num_masks): {seg_head.nm}")
print(f"   npr (num_protos): {seg_head.npr} (proto resolution)")
print(f"   nl (num_layers): {seg_head.nl}")

# Examine the convolution layers
print(f"\n2. Convolution layers in Segment head:")
if hasattr(seg_head, 'cv2'):
    print(f"   cv2 (class prediction): {len(seg_head.cv2)} layers")
    for i, cv in enumerate(seg_head.cv2):
        if hasattr(cv, 'weight'):
            print(f"      [{i}] out_channels: {cv.weight.shape[0]}")

if hasattr(seg_head, 'cv3'):
    print(f"   cv3 (box prediction): {len(seg_head.cv3)} layers")
    for i, cv in enumerate(seg_head.cv3):
        if hasattr(cv, 'weight'):
            print(f"      [{i}] out_channels: {cv.weight.shape[0]}")

if hasattr(seg_head, 'cv4'):
    print(f"   cv4 (mask coefficients): {len(seg_head.cv4)} layers")
    for i, cv in enumerate(seg_head.cv4):
        if hasattr(cv, 'weight'):
            print(f"      [{i}] out_channels: {cv.weight.shape[0]}")

# Check for proto layer (generates mask prototypes)
print(f"\n3. Prototype layer:")
if hasattr(seg_head, 'proto'):
    print(f"   Type: {type(seg_head.proto).__name__}")
    if hasattr(seg_head.proto, 'cv1'):
        print(f"   Has convolution layers")
    print(f"   Expected output: ({seg_head.nm} prototypes at {seg_head.npr}x{seg_head.npr})")

# Test forward pass with hooks
print(f"\n4. Testing with forward hooks to capture intermediate outputs:")

captured_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        captured_outputs[name] = output
    return hook

# Register hooks
if hasattr(seg_head, 'proto'):
    seg_head.proto.register_forward_hook(make_hook('proto'))

# Forward pass
test_input = torch.randn(1, 3, 160, 160)
pytorch_model.eval()
with torch.no_grad():
    outputs = pytorch_model(test_input)

print(f"   Captured outputs:")
for name, output in captured_outputs.items():
    if isinstance(output, torch.Tensor):
        print(f"     {name}: {output.shape}")
    elif isinstance(output, (list, tuple)):
        print(f"     {name}: tuple/list of {len(output)} elements")

# Analyze main outputs
print(f"\n5. Main forward output structure:")
if isinstance(outputs, tuple):
    print(f"   Tuple with {len(outputs)} elements")
    for i, out in enumerate(outputs):
        if isinstance(out, torch.Tensor):
            print(f"     [{i}] Tensor: {out.shape}")
        elif isinstance(out, (list, tuple)):
            print(f"     [{i}] List/Tuple: {len(out)} elements")
            for j, subout in enumerate(out):
                if isinstance(subout, torch.Tensor):
                    print(f"         [{j}] Tensor: {subout.shape}")

# Try to understand YOLO's segmentation approach
print(f"\n6. Understanding YOLO segmentation approach:")
print(f"   YOLO uses instance segmentation (not semantic segmentation):")
print(f"   - Detects objects with bounding boxes (nc={seg_head.nc} classes)")
print(f"   - For each detection, predicts mask coefficients (nm={seg_head.nm})")
print(f"   - Combines coefficients with prototypes (npr={seg_head.npr}x{seg_head.npr})")
print(f"   - Result: one mask per detected instance, not per class")

print(f"\n   THIS IS A PROBLEM:")
print(f"   - IrisPupilNet needs semantic segmentation: (B, 3, H, W) class logits")
print(f"   - YOLO provides instance segmentation: masks for detected objects")
print(f"   - These are fundamentally different tasks!")

# Check if we can adapt
print(f"\n7. Potential adaptation strategies:")
print(f"   Strategy 1: Use YOLO as backbone + custom segmentation head")
print(f"      - Extract features from layer ~10-16 (before detection head)")
print(f"      - Add custom decoder to produce (B, 3, H, W) logits")
print(f"      - Discard YOLO's Segment head")
print(f"")
print(f"   Strategy 2: Modify YOLO's Segment head for semantic segmentation")
print(f"      - Replace cv2/cv3 (detection) with semantic segmentation convs")
print(f"      - Use proto layer directly for semantic masks")
print(f"      - Heavily modify YOLO architecture")
print(f"")
print(f"   Strategy 3: Convert instance masks to semantic masks")
print(f"      - Run YOLO detection+segmentation")
print(f"      - Merge instance masks into class-based semantic masks")
print(f"      - Complex post-processing, loses gradient flow")

# Let's check the backbone features
print(f"\n8. Checking backbone features (for Strategy 1):")
feature_layers = [10, 13, 16, 19, 22]  # Before Segment head
for layer_idx in feature_layers:
    layer = pytorch_model.model[layer_idx]
    print(f"   Layer [{layer_idx}] {type(layer).__name__}")

# Hook to capture feature maps
feature_maps = {}
def feature_hook(name):
    def hook(module, input, output):
        feature_maps[name] = output
    return hook

# Register hooks on potential feature layers
for idx in [16, 19, 22]:  # These feed into Segment head
    pytorch_model.model[idx].register_forward_hook(feature_hook(f'layer_{idx}'))

# Forward pass
with torch.no_grad():
    _ = pytorch_model(test_input)

print(f"\n   Captured feature maps:")
for name, fmap in feature_maps.items():
    if isinstance(fmap, torch.Tensor):
        print(f"     {name}: {fmap.shape}")

print(f"\n   These could be used for custom segmentation head!")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("YOLO11-seg performs INSTANCE segmentation, not SEMANTIC segmentation.")
print("We need to use Strategy 1: YOLO as backbone + custom semantic head.")
print("")
print("Recommended implementation:")
print("1. Load YOLO11 model")
print("2. Extract backbone features from intermediate layers (e.g., layer 16, 19, 22)")
print("3. Add custom U-Net-style decoder")
print("4. Add final conv layer for (B, 3, H, W) semantic segmentation logits")
print("5. Optionally freeze YOLO backbone for faster training")
print("=" * 80)
