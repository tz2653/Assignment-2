# helper_lib/evaluator.py
from typing import Tuple, List
import torch
from torch import nn
from torch.utils.data import DataLoader

@torch.no_grad()
def collect_preds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[List[int], List[int]]:
    """
    Run the model over a loader and collect predictions and targets.
    Returns:
        (y_true, y_pred) as Python lists of ints.
    """
    model.eval()
    y_true, y_pred = [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(targets.tolist())
    return y_true, y_pred


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / max(len(y_true), 1)
