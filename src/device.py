"""Select the fastest hardware backend available to PyTorch."""

from __future__ import annotations

import torch


def select_device() -> torch.device:
    """Select CUDA first, then Apple MPS, and otherwise CPU.

    The order is intentional: CUDA is checked first on machines that expose
    both CUDA and another backend. The selected device is printed so a run's
    hardware choice is visible in its logs.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Selected device: {device}")
    return device
