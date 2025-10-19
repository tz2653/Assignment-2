# helper_lib/utils.py
import os
import random
import torch
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Make experiments reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick GPU if available; otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model state dict to a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location=None) -> None:
    """Load model state dict from a file."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
