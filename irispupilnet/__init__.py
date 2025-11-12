"""
IrisPupilNet: Iris and Pupil Segmentation with UNet-SE

A PyTorch-based training and deployment pipeline for iris/pupil segmentation.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from .models import MODEL_REGISTRY
from .datasets import DATASET_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
]
