from typing import Callable, Dict
import cv2
import numpy as np

# --- Registry ---
_MASK_CONVERTERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

def register_mask_format(name: str):
    """Decorator to register a mask conversion function."""
    def deco(fn: Callable[[np.ndarray], np.ndarray]):
        _MASK_CONVERTERS[name.lower()] = fn
        return fn
    return deco

def get_mask_converter(name: str) -> Callable[[np.ndarray], np.ndarray]:
    key = name.lower()
    if key not in _MASK_CONVERTERS:
        raise KeyError(f"Unknown dataset_format '{name}'. Registered: {list(_MASK_CONVERTERS.keys())}")
    return _MASK_CONVERTERS[key]

# --- Built-ins ---

@register_mask_format("mobius_3c")  # 0=bg, 1=iris, 2=pupil
def mobius_3c(mask_bgr: np.ndarray) -> np.ndarray:
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    red   = (mask_rgb[...,0]==255) & (mask_rgb[...,1]==0)   & (mask_rgb[...,2]==0)   # sclera -> bg(0)
    green = (mask_rgb[...,0]==0)   & (mask_rgb[...,1]==255) & (mask_rgb[...,2]==0)   # iris  -> 1
    blue  = (mask_rgb[...,0]==0)   & (mask_rgb[...,1]==0)   & (mask_rgb[...,2]==255) # pupil -> 2
    out = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    out[green] = 1
    out[blue]  = 2
    return out.astype(np.int64)


@register_mask_format("iris_pupil_eye_cls")  # Alias for iris-pupil masks using MOBIUS colors
def iris_pupil_eye_cls(mask_bgr: np.ndarray) -> np.ndarray:
    return mobius_3c(mask_bgr)


@register_mask_format("mobius")
def mobius_alias(mask_bgr: np.ndarray) -> np.ndarray:
    return mobius_3c(mask_bgr)

@register_mask_format("mobius_2c_pupil_only")  # 0=bg, 1=pupil (iris folded into bg)
def mobius_2c_pupil_only(mask_bgr: np.ndarray) -> np.ndarray:
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    blue  = (mask_rgb[...,0]==0) & (mask_rgb[...,1]==0) & (mask_rgb[...,2]==255)
    out = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    out[blue] = 1
    return out.astype(np.int64)

@register_mask_format("pascal_indexed")  # example for indexed PNG where values are already classes
def pascal_indexed(mask_bgr: np.ndarray) -> np.ndarray:
    # If mask is paletted / single-channel but cv2 read as BGR (3ch), take one channel.
    # In real code, you may want to read with PIL in "P" mode; this keeps it simple.
    return mask_bgr[...,0].astype(np.int64)

@register_mask_format("unity_eyes_3c")  # Unity Eyes synthetic dataset (same as mobius_3c)
def unity_eyes_3c(mask_bgr: np.ndarray) -> np.ndarray:
    """
    Unity Eyes synthetic dataset uses MOBIUS format:
    - Red (255,0,0): Background/sclera -> class 0
    - Green (0,255,0): Iris -> class 1
    - Blue (0,0,255): Pupil -> class 2
    """
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    red   = (mask_rgb[...,0]==255) & (mask_rgb[...,1]==0)   & (mask_rgb[...,2]==0)
    green = (mask_rgb[...,0]==0)   & (mask_rgb[...,1]==255) & (mask_rgb[...,2]==0)
    blue  = (mask_rgb[...,0]==0)   & (mask_rgb[...,1]==0)   & (mask_rgb[...,2]==255)
    out = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    out[green] = 1
    out[blue]  = 2
    return out.astype(np.int64)

@register_mask_format("tayed_3c")  # TayedEyes dataset (same as mobius_3c)
def tayed_3c(mask_bgr: np.ndarray) -> np.ndarray:
    """
    TayedEyes dataset uses MOBIUS format:
    - Red (255,0,0) in RGB / (0,0,255) in BGR: Sclera -> class 0
    - Green (0,255,0): Iris -> class 1
    - Blue (0,0,255) in RGB / (255,0,0) in BGR: Pupil -> class 2
    """
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    red   = (mask_rgb[...,0]==255) & (mask_rgb[...,1]==0)   & (mask_rgb[...,2]==0)
    green = (mask_rgb[...,0]==0)   & (mask_rgb[...,1]==255) & (mask_rgb[...,2]==0)
    blue  = (mask_rgb[...,0]==0)   & (mask_rgb[...,1]==0)   & (mask_rgb[...,2]==255)
    out = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    out[green] = 1
    out[blue]  = 2
    return out.astype(np.int64)
