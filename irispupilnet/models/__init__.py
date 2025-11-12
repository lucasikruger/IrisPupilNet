from typing import Dict, Callable

MODEL_REGISTRY: Dict[str, Callable] = {}
def register_model(name: str):
    def deco(fn_or_cls):
        MODEL_REGISTRY[name] = fn_or_cls
        return fn_or_cls
    return deco

# Import model modules to trigger registration
from . import unet_se
