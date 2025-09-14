import os
import sys
import time
import math
from glob import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from PIL import Image

# Use non-interactive backend for HPC environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
class BrainMRIDataset(Dataset):
    """
    Dataset for preprocessed OASIS brain MR images stored as PNGs.

    Recursively scans one or more root directories for `.png` files and
    returns single-channel tensors in the range [0, 1]. By default, files
    with paths containing "seg" are excluded (to avoid segmentation masks).

    Args:
        roots: Directory path or list of directory paths to scan recursively
            for PNG images. Example: [
            "/home/groups/comp3710/OASIS/keras_png_slices_train",
            "/home/groups/comp3710/OASIS/keras_png_slices_validate",
            "/home/groups/comp3710/OASIS/keras_png_slices_test",
            ].
        transform: Optional torchvision transform to apply to each PIL image
            (e.g., grayscale conversion, resize to `IMG_SIZE`, `ToTensor`).
        limit: Optional maximum number of images to load (useful for quick
            tests). If None or <= 0, uses all discovered images.
        include_seg: If True, does not filter out files whose paths contain
            "seg"; if False (default), such files are excluded.

    Returns:
        Each `__getitem__` returns a tuple `(image_tensor, 0)`, where the
        label placeholder `0` is unused (unsupervised setting).
    """

    def __init__(self, roots, transform: T.Compose | None = None, limit: int | None = None, include_seg: bool = False):
        self.roots = [roots] if isinstance(roots, str) else list(roots)
        # Recursively find PNG files across roots
        files = []
        for root in self.roots:
            patterns = [
                os.path.join(root, "**", "*.png"),
                os.path.join(root, "**", "*.PNG"),
            ]
            for p in patterns:
                # extend all matching paths
                files.extend(glob(p, recursive=True))
        # Exclude segmentation files unless requested
        if not include_seg:
            files = [f for f in files if 'seg' not in f.lower()]
        self.files = sorted(files)
        if limit is not None and limit > 0:
            self.files = self.files[:limit]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No PNG files found under: {self.roots}")

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("L")  # force 1-channel grayscale
        if self.transform:
            # convert it to tensor that can be used for training
            img = self.transform(img)
        return img, 0  # label unused (unsupervised)
    
# -----------------------------
# Model (VAE)
# -----------------------------
class CNNVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for 1*H*W brain MR images.

    The encoder downsamples the image by 2* four times (H/16 * W/16), then
    maps to a latent Gaussian with `latent_dim` mean/log-variance. The decoder
    mirrors this with transposed convolutions back to the input resolution.

    Args:
        latent_dim: Size of the latent code z (e.g., 16-128). Larger values
            typically capture more variation but may be harder to regularize.
        img_size: Square image size expected by the model (must be divisible
            by 16, e.g., 128, 256). Input images are resized to `(img_size, img_size)`.

    Input/Output:
        Input tensors have shape `(N, 1, img_size, img_size)` and values in
        [0, 1]. The decoder outputs the same shape with a sigmoid activation.
    """
    def __init__(self, latent_dim: int = 32, img_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Encoder: input 1x128x128 -> 256x8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        enc_out_spatial = img_size // 16  # 128 -> 8
        self.enc_flat_dim = 256 * enc_out_spatial * enc_out_spatial # total number of features 
        self.fc_mu = nn.Linear(self.enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.enc_flat_dim)
        self.dec_channels = 256
        self.dec_spatial = enc_out_spatial

        # Decoder: input 256x8x8 -> 1x128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1) # convert 4D tensor (batch, channels, h, w) -> (batch, features)
        mu = self.fc_mu(h) # apply fully connected layer
        logvar = self.fc_logvar(h) # apply fully connected layer
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar) # convert to actual standard deviation
            eps = torch.randn_like(std) # draws random samples from standard normal distribution
            return mu + eps * std # return latent variable
        else:
            return mu

    def decode(self, z):
        h = self.fc_dec(z) # apply fully connected layer
        h = h.view(-1, self.dec_channels, self.dec_spatial, self.dec_spatial) # reshape to (batch, channels, h, w)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute VAE loss = reconstruction loss + beta * KL divergence.

    Args:
        recon_x: Reconstructed images from the decoder, shape `(N, 1, H, W)`, values in [0, 1].
        x: Original images, same shape and range as `recon_x`.
        mu: Latent mean tensor `(N, latent_dim)`.
        logvar: Latent log-variance tensor `(N, latent_dim)`.
        beta: Weight for the KL divergence term (beta-VAE). Use 1.0 for a
            standard VAE; >1.0 for stronger disentanglement pressure.

    Returns:
        total_loss, bce, kld: Scalars (summed over batch and pixels for BCE/KLD);
        typically divide by dataset size per epoch for reporting.
    """
    # Reconstruction loss (BCE expects inputs in [0,1])
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum') # Compare between decoded image and actual image
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # Measures how far the learned latent distribution is from prior
    return bce + beta * kld, bce, kld