"""Device selection for inference."""

import torch
from functools import wraps


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cuda_autocast(function):
    """Use the original BF16 autocast only when running on CUDA."""

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return function(self, *args, **kwargs)
        return function(self, *args, **kwargs)

    return wrapped
