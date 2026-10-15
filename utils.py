"""
Shared utilities: global seeding for reproducibility and small helpers.
"""

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seeds Python, NumPy and PyTorch (CPU/CUDA/MPS) RNGs for reproducible runs.

    Args:
        seed: The seed value applied to every RNG.
        deterministic: If True, request deterministic kernels where available.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            # warn_only avoids hard failures on ops lacking deterministic kernels (e.g. some MPS ops)
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
