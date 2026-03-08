"""ResNet-MLP value network for pancake sorting distance prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with two linear layers."""

    def __init__(self, width: int) -> None:
        """Initialize residual block.

        Args:
            width: Hidden dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, width).

        Returns:
            Output tensor of same shape.
        """
        return x + self.net(x)


class ResNetMLP(nn.Module):
    """ResNet-MLP value network for predicting sorting distance.

    Architecture: position embeddings -> projection -> ResNet blocks -> scalar output.
    """

    def __init__(
        self,
        max_n: int = 100,
        embed_dim: int = 64,
        width: int = 2048,
        num_blocks: int = 10,
    ) -> None:
        """Initialize the model.

        Args:
            max_n: Maximum permutation size.
            embed_dim: Embedding dimension per position.
            width: Hidden layer width.
            num_blocks: Number of residual blocks.
        """
        super().__init__()
        self.max_n = max_n
        self.embed_dim = embed_dim

        # Learnable embeddings for (position, value) pairs
        self.pos_embed = nn.Embedding(max_n, embed_dim)
        self.val_embed = nn.Embedding(max_n, embed_dim)

        # Project concatenated embeddings to hidden width
        self.input_proj = nn.Sequential(
            nn.Linear(max_n * embed_dim * 2, width),
            nn.ReLU(),
        )

        # Residual blocks
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(num_blocks)])

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, perm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            perm: Permutation tensor of shape (batch, n), dtype long.

        Returns:
            Predicted distance of shape (batch, 1).
        """
        batch_size, n = perm.shape

        positions = torch.arange(n, device=perm.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)  # (batch, n, embed_dim)
        val_emb = self.val_embed(perm)  # (batch, n, embed_dim)

        # Concatenate position and value embeddings
        combined = torch.cat([pos_emb, val_emb], dim=-1)  # (batch, n, 2*embed_dim)

        # Pad to max_n if needed
        if n < self.max_n:
            pad = torch.zeros(batch_size, self.max_n - n, 2 * self.embed_dim, device=perm.device)
            combined = torch.cat([combined, pad], dim=1)

        # Flatten and project
        x = combined.reshape(batch_size, -1)
        x = self.input_proj(x)

        # Residual blocks
        x = self.blocks(x)

        # Output
        return self.head(x)

    def save(self, path: str) -> None:
        """Save model weights.

        Args:
            path: Output path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> None:
        """Load model weights.

        Args:
            path: Path to saved weights.
            device: Device to load to.
        """
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))
