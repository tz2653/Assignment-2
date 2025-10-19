# helper_lib/generator.py
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

@torch.no_grad()
def generate_samples(model, num_samples: int, device: torch.device,
                     latent_dim: Optional[int] = None):
    """
    Sample random latent vectors and decode to images in [0,1].
    Returns a tensor of shape (N, 3, 128, 128) in [0,1].
    """
    model.eval()
    model.to(device)

    # Infer latent_dim if not provided
    if latent_dim is None:
        latent_dim = getattr(model, "latent_dim", 64)

    z = torch.randn(num_samples, latent_dim, device=device)
    logits = model.decode(z)  # (N, 3, 128, 128) logits
    imgs = torch.sigmoid(logits)  # convert to [0,1]
    return imgs.cpu()

@torch.no_grad()
def save_grid(images: torch.Tensor, path: str, nrow: int = 4):
    """
    Save a grid image to the given path. `images` should be in [0,1].
    """
    grid = make_grid(images, nrow=nrow)
    save_image(grid, path)
