"""PyTorch training loop for ResNet-MLP value model."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flipstack.models.resnet_mlp import ResNetMLP
from flipstack.training.data_gen import generate_random_walks

logger = logging.getLogger(__name__)


def train_resnet_mlp(
    n: int,
    num_samples: int = 100000,
    epochs: int = 1000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    data_regen_interval: int = 300,
    checkpoint_dir: str = "experiments/checkpoints",
    device: str = "cpu",
    seed: int = 42,
) -> ResNetMLP:
    """Train a ResNet-MLP value model for a given permutation size.

    Args:
        n: Permutation size.
        num_samples: Training samples per data generation.
        epochs: Total training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate (constant).
        data_regen_interval: Regenerate data every N epochs.
        checkpoint_dir: Directory for checkpoints.
        device: Training device ('cpu', 'cuda', 'mps').
        seed: Random seed.

    Returns:
        Trained model.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    model = ResNetMLP(max_n=n, embed_dim=64, width=2048, num_blocks=10).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model for n=%d: %d params, device=%s", n, param_count, device)

    start_time = time.monotonic()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Regenerate data periodically
        if epoch == 1 or epoch % data_regen_interval == 0:
            x_np, y_np = generate_random_walks(n, num_samples=num_samples, seed=seed + epoch)
            x_tensor = torch.from_numpy(x_np.astype(np.int64)).to(dev)
            y_tensor = torch.from_numpy(y_np).unsqueeze(1).to(dev)
            dataset = TensorDataset(x_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            logger.info("Epoch %d: regenerated %d samples", epoch, num_samples)

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if epoch % 100 == 0 or epoch == 1:
            elapsed = time.monotonic() - start_time
            logger.info(
                "Epoch %d/%d: loss=%.4f, best=%.4f, time=%.0fs",
                epoch,
                epochs,
                avg_loss,
                best_loss,
                elapsed,
            )

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Checkpoint every 100 epochs
        if epoch % 100 == 0:
            import pathlib

            ckpt_path = pathlib.Path(checkpoint_dir) / f"resnet_n{n}_ep{epoch}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(ckpt_path))

    # Save final
    import pathlib

    final_path = pathlib.Path(checkpoint_dir) / f"resnet_n{n}_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))
    logger.info("Training complete for n=%d, final loss=%.4f", n, best_loss)

    return model
