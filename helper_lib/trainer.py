# helper_lib/trainer.py
from typing import Dict, List, Tuple, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on validation/test loader.

    Returns:
        (avg_loss, avg_accuracy)
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Full training loop with optional validation.

    Returns:
        history dict with keys: ["train_loss", "train_acc", "val_loss", "val_acc"]
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"[{epoch:03d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f}")
        else:
            print(f"[{epoch:03d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f}")

    return history

# ======== VAE training utilities ========
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

def vae_loss(recon_logits: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = reconstruction + beta * KL.
    - recon_logits: decoder outputs (logits)
    - x: input images in [0,1]
    - mu, logvar: parameters of q(z|x)
    Returns: (total_loss, recon_loss, kl_loss), each averaged per batch.
    """
    # BCE with logits expects float targets in [0,1]
    recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum")
    # KL divergence between q(z|x) ~ N(mu, sigma^2) and p(z) ~ N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = (recon + beta * kl) / x.size(0)  # normalize by batch size
    return total, recon / x.size(0), kl / x.size(0)


def train_vae_model(
    model: torch.nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    device: torch.device,
    beta: float = 1.0,
) -> Dict[str, List[float]]:
    """
    Generic VAE training loop.
    Returns history with keys: ['total_loss', 'recon_loss', 'kl_loss']
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"total_loss": [], "recon_loss": [], "kl_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tot_sum = rec_sum = kl_sum = n_batches = 0

        for x, _ in train_loader:
            # Expect images scaled to [0,1]; if tensors are [0,255], normalize before loading.
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, mu, logvar = model(x)
            total, recon, kl = vae_loss(logits, x, mu, logvar, beta=beta)
            total.backward()
            optimizer.step()

            tot_sum += total.item()
            rec_sum += recon.item()
            kl_sum  += kl.item()
            n_batches += 1

        avg_total = tot_sum / max(n_batches, 1)
        avg_recon = rec_sum / max(n_batches, 1)
        avg_kl    = kl_sum  / max(n_batches, 1)
        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        print(f"[{epoch:03d}] vae_total={avg_total:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f} (beta={beta})")

    return history
