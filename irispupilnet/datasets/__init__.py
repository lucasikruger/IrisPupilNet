from typing import Dict, Type, Callable

DATASET_REGISTRY: Dict[str, Callable] = {}
def register_dataset(name: str):
    def deco(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return deco

# Import dataset modules to trigger registration
from . import csv_seg
