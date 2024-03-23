import torch
from functools import lru_cache


@lru_cache(maxsize=5)
def get_hardware_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"
