#!/usr/bin/env python3
"""
Part 4.3.1 — Variational Autoencoder (VAE) for MR brain images

This script implements a VAE tailored for the Preprocessed OASIS dataset
stored on the Rangpur cluster under /home/groups/comp3710/.

It is designed to be robust to local development: you can specify a dataset
path via --data_dir. If not found, it can fall back to a small synthetic
dataset for quick checks. On the cluster, point --data_dir to the OASIS path
or an extracted subset.

Features:
- Configurable latent dimension (default 2 for direct manifold plotting)
- 2D slice extraction from volumes, normalization, and resizing
- Training loop with KL and reconstruction losses
- Optional UMAP/t-SNE latent visualization if latent_dim > 2
- Reconstruction and random-sample grids for qualitative assessment

Example (cluster):
  python part4.3.1.py \
    --data_dir /home/groups/comp3710/oasis_preprocessed \
    --epochs 25 --batch_size 128 --img_size 128 --latent_dim 2

Example (local quick run):
  python part4.3.1.py --synthetic --epochs 1 --batch_size 32
"""

import argparse
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision
    from torchvision.utils import make_grid, save_image
    from torchvision import transforms as T
except Exception:  # pragma: no cover
    torchvision = None
    T = None

# Optional: nibabel for NIfTI support if available
try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover
    nib = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MRISliceDataset(Dataset):
    """Dataset that loads 2D slices from MR volumes or PNGs.

    Supports:
    - Directory of NIfTI: *.nii or *.nii.gz (requires nibabel)
    - Directory of PNG/JPG images (uses PIL via torchvision)
    """

    def __init__(
        self,
        data_dir: Path,
        img_size: int = 128,
        max_slices_per_volume: Optional[int] = 64,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.max_slices_per_volume = max_slices_per_volume
        self.samples = []  # list of (path, slice_idx or None, kind)

        nii_exts = {".nii", ".nii.gz"}
        img_exts = {".png", ".jpg", ".jpeg", ".bmp"}

        for p in sorted(self.data_dir.rglob("*")):
            if p.is_file():
                lower = p.name.lower()
                if any(lower.endswith(ext) for ext in [".nii", ".nii.gz"]):
                    self.samples.append((p, None, "nii"))
                elif any(lower.endswith(ext) for ext in img_exts):
                    self.samples.append((p, None, "img"))

        if not self.samples:
            raise FileNotFoundError(
                f"No supported files found under {self.data_dir}. "
                "Expecting NIfTI volumes (.nii/.nii.gz) or images (.png/.jpg)."
            )

        # Preload for images when not using NIfTI
        self.to_tensor = (
            T.Compose([
                T.Grayscale(num_output_channels=1),
                T.Resize((img_size, img_size)),
                T.ToTensor(),  # 0..1
            ])
            if T is not None
            else None
        )

    def __len__(self) -> int:
        # For NIfTI, we defer actual indexing to __getitem__ by sampling slices
        # on the fly to keep it simple. We approximate length as num_files * max_slices_per_volume
        nii_count = sum(1 for _, _, k in self.samples if k == "nii")
        img_count = sum(1 for _, _, k in self.samples if k == "img")
        return img_count + (nii_count * (self.max_slices_per_volume or 1))

    def _load_nifti_slice(self, path: Path) -> np.ndarray:
        if nib is None:
            raise RuntimeError("nibabel is required to read NIfTI files but is not installed.")
        vol = nib.load(str(path)).get_fdata()
        # Choose a middle slice along the last axis and optionally jitter
        axis = 2 if vol.ndim == 3 else -1
        mid = vol.shape[axis] // 2
        if self.max_slices_per_volume and self.max_slices_per_volume > 1:
            # random offset within +/- 20 slices if available
            span = min(20, vol.shape[axis] // 2)
            offset = random.randint(-span, span)
            idx = int(np.clip(mid + offset, 0, vol.shape[axis] - 1))
        else:
            idx = mid
        slc = vol[:, :, idx] if axis == 2 else np.take(vol, idx, axis=axis)
        # Min-max normalize per-slice to 0..1
        slc = slc.astype(np.float32)
        mn, mx = np.percentile(slc, 1), np.percentile(slc, 99)
        slc = np.clip((slc - mn) / (mx - mn + 1e-6), 0, 1)
        # Resize to target using torchvision if available else numpy
        if T is not None:
            pil = torchvision.transforms.functional.to_pil_image(slc)
            pil = pil.resize((self.img_size, self.img_size))
            tensor = torchvision.transforms.functional.to_tensor(pil)  # 1xHxW
        else:
            from skimage.transform import resize  # type: ignore

            tensor = resize(slc, (self.img_size, self.img_size), anti_aliasing=True)
            tensor = torch.from_numpy(tensor[None, ...].astype(np.float32))
        return tensor  # 1xHxW

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Map idx to an underlying file deterministically
        nii_files = [p for p, _, k in self.samples if k == "nii"]
        img_files = [p for p, _, k in self.samples if k == "img"]
        nii_count = len(nii_files)
        img_count = len(img_files)

        if idx < img_count and self.to_tensor is not None:
            # Direct image path
            path = img_files[idx]
            img = torchvision.io.read_image(str(path)) if hasattr(torchvision.io, "read_image") else None
            if img is not None:
                if img.shape[0] == 3:
                    img = img.mean(dim=0, keepdim=True)  # crude grayscale
                img = F.interpolate(img.unsqueeze(0).float() / 255.0, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
                return img
            else:
                # Fallback via PIL
                from PIL import Image  # type: ignore

                pil = Image.open(path).convert("L").resize((self.img_size, self.img_size))
                return torchvision.transforms.functional.to_tensor(pil)

        # For NIfTI, sample a slice from a volume chosen by idx
        if nii_count == 0:
            raise IndexError("Index exceeds dataset size and no NIfTI files available.")
        vol_idx = (idx - img_count) % nii_count
        path = nii_files[vol_idx]
        return self._load_nifti_slice(path)


class SyntheticBlobs(Dataset):
    """Small synthetic dataset for local sanity checks."""

    def __init__(self, n: int = 1024, img_size: int = 64) -> None:
        self.n = n
        self.img_size = img_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        canvas = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        # Random circle
        r = random.randint(self.img_size // 12, self.img_size // 6)
        cx = random.randint(r + 1, self.img_size - r - 1)
        cy = random.randint(r + 1, self.img_size - r - 1)
        yy, xx = np.ogrid[: self.img_size, : self.img_size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        canvas[mask] = 1.0
        canvas += 0.05 * np.random.randn(*canvas.shape).astype(np.float32)
        canvas = np.clip(canvas, 0, 1)
        return torch.from_numpy(canvas)[None, ...]


class ConvVAE(nn.Module):
    def __init__(self, img_size: int = 128, latent_dim: int = 2) -> None:
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Encoder: 1xHxW -> latent
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # H/8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # H/16
            nn.ReLU(inplace=True),
        )
        feat_size = img_size // 16
        self.enc_out = 256 * feat_size * feat_size
        self.fc_mu = nn.Linear(self.enc_out, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out, latent_dim)

        # Decoder: latent -> 1xHxW
        self.fc_dec = nn.Linear(latent_dim, self.enc_out)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),  # 0..1
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), 256, self.img_size // 16, self.img_size // 16)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def loss_fn(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Reconstruction (BCE) + KL divergence
    rec = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return rec + kld, rec, kld


def get_dataloader(
    data_dir: Optional[str], img_size: int, batch_size: int, synthetic: bool
) -> DataLoader:
    if synthetic:
        ds = SyntheticBlobs(n=2048, img_size=img_size)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    if data_dir is None:
        raise ValueError("--data_dir must be provided unless --synthetic is set.")
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    ds = MRISliceDataset(root, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def maybe_umap(x: np.ndarray) -> Optional[np.ndarray]:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(x)
    except Exception:
        return None


@torch.no_grad()
def visualize(model: ConvVAE, device: torch.device, outdir: Path, loader: DataLoader, latent_dim: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    # Collect a batch
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    batch = batch.to(device)

    # Reconstructions
    x_hat, mu, logvar = model(batch)
    both = torch.cat([batch[:8], x_hat[:8]], dim=0)
    save_image(both, outdir / "recon_grid.png", nrow=8)

    # Random samples from prior
    z = torch.randn(64, latent_dim, device=device)
    samples = model.decode(z)
    save_image(samples, outdir / "samples_grid.png", nrow=8)

    # Latent scatter
    all_mu = []
    max_batches = 16
    for i, xb in enumerate(loader):
        if i >= max_batches:
            break
        xb = xb.to(device)
        _, mu_b, _ = model(xb)
        all_mu.append(mu_b.cpu().numpy())
    Z = np.concatenate(all_mu, axis=0)
    if latent_dim == 2:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(5, 5))
            plt.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.5)
            plt.title("Latent space (mu)")
            plt.tight_layout()
            plt.savefig(outdir / "latent_scatter.png", dpi=150)
            plt.close()
        except Exception:
            pass
    else:
        Z2 = maybe_umap(Z)
        if Z2 is None:
            # Fallback to t-SNE if available
            try:
                from sklearn.manifold import TSNE  # type: ignore

                Z2 = TSNE(n_components=2, init="pca", random_state=42).fit_transform(Z)
            except Exception:
                Z2 = None
        if Z2 is not None:
            try:
                import matplotlib.pyplot as plt  # type: ignore

                plt.figure(figsize=(5, 5))
                plt.scatter(Z2[:, 0], Z2[:, 1], s=4, alpha=0.5)
                plt.title("Latent space (2D embedding)")
                plt.tight_layout()
                plt.savefig(outdir / "latent_embedding.png", dpi=150)
                plt.close()
            except Exception:
                pass


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader = get_dataloader(args.data_dir, args.img_size, args.batch_size, args.synthetic)
    model = ConvVAE(img_size=args.img_size, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_interval = max(1, len(loader) // 10)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, total_rec, total_kld = 0.0, 0.0, 0.0
        for i, batch in enumerate(loader, start=1):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(batch)
            loss, rec, kld = loss_fn(x_hat, batch, mu, logvar)
            loss.backward()
            opt.step()
            total += float(loss)
            total_rec += float(rec)
            total_kld += float(kld)
            if i % log_interval == 0:
                print(
                    f"Epoch {epoch} [{i}/{len(loader)}] loss={loss.item():.4f} rec={rec.item():.4f} kld={kld.item():.4f}",
                    flush=True,
                )
        print(
            f"Epoch {epoch} done: loss={total/len(loader):.4f} rec={total_rec/len(loader):.4f} kld={total_kld/len(loader):.4f}",
            flush=True,
        )

        # Periodic visualization
        if epoch % args.viz_every == 0 or epoch == args.epochs:
            outdir = Path(args.out_dir)
            visualize(model, device, outdir, loader, args.latent_dim)

    # Final save
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, Path(args.out_dir) / "vae.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE on OASIS brain MRI slices")
    p.add_argument("--data_dir", type=str, default=None, help="Path to OASIS preprocessed root")
    p.add_argument("--out_dir", type=str, default=".dist/vae", help="Output directory for artifacts")
    p.add_argument("--img_size", type=int, default=128, help="Image size (square)")
    p.add_argument("--latent_dim", type=int, default=2, help="Latent dimension")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--viz_every", type=int, default=5, help="Visualize every N epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic blobs dataset for local checks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Guidance: On Rangpur, set --data_dir to /home/groups/comp3710/... path
    if not args.synthetic and args.data_dir is None:
        print("Warning: --data_dir is not set; use --synthetic for a local dry run.")
    train(args)


if __name__ == "__main__":
    main()
