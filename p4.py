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

# -----------------------------
# Visualization helpers
# -----------------------------
def save_reconstructions(vae, data_loader, device, out_path, n=8):
    """
    Save a grid comparing original vs reconstructed validation images.

    Args:
        vae: Trained VAE model.
        data_loader: DataLoader used for sampling images to reconstruct.
        device: Torch device to run inference on.
        out_path: Filepath to save the PNG figure.
        n: Number of columns (image pairs) to render.
    """
    vae.eval()
    with torch.no_grad():
        images, _ = next(iter(data_loader))
        images = images[: n * 2].to(device)
        recon, _, _ = vae(images)

        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
        for i in range(n):
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
        axes[0, 0].set_ylabel('Orig')
        axes[1, 0].set_ylabel('Recon')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)


def save_generated_samples(vae, device, out_path, nrow=8):
    """
    Sample random latent vectors and save a grid of generated images.

    Args:
        vae: Trained VAE model.
        device: Torch device to run inference on.
        out_path: Filepath to save the PNG figure.
        nrow: Grid size (nrow × nrow samples).
    """
    vae.eval()
    with torch.no_grad():
        z = torch.randn(nrow * nrow, vae.latent_dim).to(device)
        samples = vae.decode(z)
        fig, axes = plt.subplots(nrow, nrow, figsize=(nrow * 1.5, nrow * 1.5))
        for i in range(nrow):
            for j in range(nrow):
                k = i * nrow + j
                axes[i, j].imshow(samples[k].cpu().squeeze(), cmap='gray')
                axes[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)


def save_latent_tsne(vae, data_loader, device, out_path, max_batches=20):
    """
    Compute 2D t-SNE embedding of validation latent means and save a scatter.

    Args:
        vae: Trained VAE model.
        data_loader: Validation DataLoader.
        device: Torch device to run inference on.
        out_path: Filepath to save the PNG figure.
        max_batches: Optional cap on number of batches to embed for speed.
    """
    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        print(f"t-SNE unavailable: {e}")
        return

    vae.eval()
    latents = []
    with torch.no_grad():
        for b_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)
            mu, _ = vae.encode(x)
            latents.append(mu.cpu())
            if max_batches and (b_idx + 1) >= max_batches:
                break
    latents = torch.cat(latents, dim=0).numpy()
    print(f"t-SNE on {latents.shape[0]} embeddings ...")
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate='auto')
    lat2d = tsne.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    plt.scatter(lat2d[:, 0], lat2d[:, 1], s=3, alpha=0.6, c='steelblue')
    plt.title('VAE Latent Space (t-SNE)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_manifold_grid(vae, device, out_path, n_grid=15, dims: tuple[int, int] | None = None, span: float = 3.0):
    """
    Visualize a 2D manifold by fixing two latent dimensions on a grid.

    Args:
        vae: Trained VAE model.
        device: Torch device to run inference on.
        out_path: Filepath to save the PNG figure.
        n_grid: Number of steps per axis on the grid.
        dims: Tuple of (dim1, dim2) latent dimensions to vary. If None,
            uses (0, 1) or a fallback when latent_dim < 2.
        span: Range for each axis, sampled from [-span, span].
    """
    vae.eval()
    if dims is None:
        dim1, dim2 = 0, 1 if vae.latent_dim > 1 else (0, 0)
    else:
        dim1, dim2 = dims

    grid_x = np.linspace(-span, span, n_grid)
    grid_y = np.linspace(-span, span, n_grid)
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(n_grid, n_grid))

    with torch.no_grad():
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                z = torch.zeros(1, vae.latent_dim, device=device)
                if vae.latent_dim >= 2:
                    z[0, dim1] = x
                    z[0, dim2] = y
                else:
                    z[0, 0] = x  # single-dim fallback
                img = vae.decode(z)
                axes[i, j].imshow(img.cpu().squeeze(), cmap='gray')
                axes[i, j].axis('off')
    plt.suptitle(f'Manifold grid (dims {dim1},{dim2})', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_circular_walk(vae, device, out_path, n_samples=24, radius=2.0, dims: tuple[int, int] | None = None):
    """
    Visualize traversal by walking along a circle in a 2D latent subspace.

    Args:
        vae: Trained VAE model.
        device: Torch device to run inference on.
        out_path: Filepath to save the PNG figure.
        n_samples: Number of points along the circle to decode.
        radius: Circle radius in latent units for the chosen dims.
        dims: Tuple of (dim1, dim2) latent dimensions to vary. If None,
            uses (0, 1) or a fallback when latent_dim < 2.
    """
    vae.eval()
    if dims is None:
        dim1, dim2 = 0, 1 if vae.latent_dim > 1 else (0, 0)
    else:
        dim1, dim2 = dims

    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    fig, axes = plt.subplots(3, math.ceil(n_samples / 3), figsize=(16, 6))
    axes = axes.flatten()
    with torch.no_grad():
        for i, ang in enumerate(angles):
            z = torch.zeros(1, vae.latent_dim, device=device)
            z[0, dim1] = radius * math.cos(ang)
            if vae.latent_dim >= 2:
                z[0, dim2] = radius * math.sin(ang)
            img = vae.decode(z)
            axes[i].imshow(img.cpu().squeeze(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"{ang:.2f}", fontsize=8)
    plt.suptitle(f'Circular walk in latent space (dims {dim1},{dim2})', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------------
# Training
# -----------------------------
# Hardcoded configuration for SLURM execution
OUTPUT_DIR = "./outputs_p4"


def train():
    """
    Train the VAE on OASIS PNG images using hardcoded configuration.

    Uses global constants for data directories, hyperparameters, and output
    locations. Saves checkpoints and several visualization artifacts to
    `OUTPUT_DIR`.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Using CPU.")
    else:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    ensure_dir(OUTPUT_DIR)

    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((128, 128)),
        T.ToTensor(),  # [0,1]
    ])

    # Explicit train/val directories (exclude test set)
    train_dirs = "/home/groups/comp3710/OASIS/keras_png_slices_train"
    val_dirs = "/home/groups/comp3710/OASIS/keras_png_slices_validate"
    train_ds = BrainMRIDataset(train_dirs, transform=transform, limit=None, include_seg=False)
    val_ds = BrainMRIDataset(val_dirs, transform=transform, limit=None, include_seg=False)
    print(f"Dataset size: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=max(0, 8 // 2),
        pin_memory=True,
    )

    # Model
    vae = CNNVAE(latent_dim=32, img_size=128).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    best_val = float('inf')
    start_time = time.time()

    print("Training VAE ...")
    for epoch in range(1, 21):
        vae.train()
        running_loss = 0.0
        running_bce = 0.0
        running_kld = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss, bce, kld = vae_loss_function(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            running_bce += bce.item()
            running_kld += kld.item()

        n_train_elems = len(train_loader.dataset)
        train_loss = running_loss / n_train_elems
        train_bce = running_bce / n_train_elems
        train_kld = running_kld / n_train_elems

        # Validation
        vae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, mu, logvar = vae(x)
                loss, _, _ = vae_loss_function(recon, x, mu, logvar)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch}/{20}] "
              f"Train: loss={train_loss:.4f}, bce={train_bce:.4f}, kld={train_kld:.4f} | "
              f"Val: loss={val_loss:.4f}")

        # Save periodic artifacts
        if epoch % 5 == 0 or epoch == 20:
            save_reconstructions(vae, val_loader, device, os.path.join(OUTPUT_DIR, f"recon_epoch{epoch}.png"))

        # Track best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "vae_best.pth"))

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.1f} min. Best val loss: {best_val:.4f}")

    # Final artifacts
    torch.save(vae.state_dict(), os.path.join(OUTPUT_DIR, "vae_last.pth"))
    save_reconstructions(vae, val_loader, device, os.path.join(OUTPUT_DIR, "recon_final.png"))
    save_generated_samples(vae, device, os.path.join(OUTPUT_DIR, "samples.png"))
    save_latent_tsne(vae, val_loader, device, os.path.join(OUTPUT_DIR, "latent_tsne.png"), max_batches=30)
    save_manifold_grid(vae, device, os.path.join(OUTPUT_DIR, "manifold_grid.png"), n_grid=15)
    save_circular_walk(vae, device, os.path.join(OUTPUT_DIR, "manifold_circular.png"), n_samples=24)

if __name__ == '__main__':
    train()
