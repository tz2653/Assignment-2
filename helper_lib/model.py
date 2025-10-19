# helper_lib/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Add back SimpleCNN definition ---
class SimpleCNN(nn.Module):
    """
    A small CNN suitable for 128x128 RGB images.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VAE(nn.Module):
    """
    A simple convolutional VAE for 128x128 RGB images.
    Decoder outputs logits; use BCEWithLogits for recon loss.
    """
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (B,3,128,128) -> (B,128,16,16)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64x64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16
            nn.ReLU(True),
        )
        enc_out_dim = 128 * 16 * 16
        self.fc_mu    = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar= nn.Linear(enc_out_dim, latent_dim)

        # Decoder: z -> (B,128,16,16) -> (B,3,128,128)
        self.fc_dec = nn.Linear(latent_dim, enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 128x128
            # NOTE: logits output; apply sigmoid at inference if needed
        )

    def encode(self, x):
        h = self.enc(x)
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 128, 16, 16)
        return self.dec(h)  # logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


# ---- name-based factory ----
def get_model(model_name: str | None = None, **kwargs) -> nn.Module:
    """
    Build model by name. Supported: 'CNN'/'SimpleCNN', 'VAE'.
    Fallback to SimpleCNN if model_name is None and num_classes provided.
    """
    name = (model_name or "").lower() if isinstance(model_name, str) else None
    if name in {"cnn", "simplecnn"}:
        num_classes = kwargs.get("num_classes", 2)
        return SimpleCNN(num_classes=num_classes)
    if name == "vae":
        latent_dim = kwargs.get("latent_dim", 64)
        return VAE(latent_dim=latent_dim)
    # backward-compatible fallback
    if "num_classes" in kwargs:
        return SimpleCNN(num_classes=kwargs["num_classes"])
    raise ValueError("Please provide model_name in {'CNN','SimpleCNN','VAE'} or num_classes for SimpleCNN.")

# ---- Assignment2: exact CNN spec for 64x64 ----
import torch.nn as nn
import torch

class A2CNN(nn.Module):
    """Assignment 2 CNN: 64x64x3 -> 10 classes."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # -> (16,64,64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> (16,32,32)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> (32,32,32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> (32,16,16)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                           # -> 32*16*16
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---- extend factory ----
def get_model(model_name: str | None = None, **kwargs) -> nn.Module:
    name = (model_name or "").lower() if isinstance(model_name, str) else None
    if name in {"cnn", "simplecnn"}:
        return SimpleCNN(num_classes=kwargs.get("num_classes", 2))
    if name == "a2cnn":
        return A2CNN(num_classes=kwargs.get("num_classes", 10))
    if name == "vae":
        return VAE(latent_dim=kwargs.get("latent_dim", 64))
    if "num_classes" in kwargs:
        return SimpleCNN(num_classes=kwargs["num_classes"])
    raise ValueError("Use model_name in {'A2CNN','CNN','SimpleCNN','VAE'} or pass num_classes.")
