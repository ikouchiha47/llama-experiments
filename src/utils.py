import torch
from functools import lru_cache


@lru_cache(maxsize=2)
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
