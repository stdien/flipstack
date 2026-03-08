"""Learned move filtering: predict which flips to try."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MovePredictor(nn.Module):
    """Predict probability of each flip being useful.

    Input: permutation (as long tensor).
    Output: logits for each possible flip k=2..n.
    """

    def __init__(
        self,
        max_n: int = 100,
        embed_dim: int = 32,
        hidden_dim: int = 512,
    ) -> None:
        """Initialize move predictor.

        Args:
            max_n: Maximum permutation size.
            embed_dim: Embedding dimension.
            hidden_dim: Hidden layer width.
        """
        super().__init__()
        self.max_n = max_n

        self.embed = nn.Embedding(max_n, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(max_n * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_n - 1),  # k=2..n -> n-1 outputs
        )

    def forward(self, perm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            perm: Permutation tensor of shape (batch, n), dtype long.

        Returns:
            Logits of shape (batch, max_n-1) for flips k=2..max_n.
        """
        batch_size, n = perm.shape
        emb = self.embed(perm)  # (batch, n, embed_dim)

        # Pad to max_n
        if n < self.max_n:
            pad = torch.zeros(batch_size, self.max_n - n, emb.shape[-1], device=perm.device)
            emb = torch.cat([emb, pad], dim=1)

        x = emb.reshape(batch_size, -1)
        return self.net(x)

    def predict_moves(self, perm: torch.Tensor, top_k: int = 4) -> list[list[int]]:
        """Predict top-k moves for a batch of permutations.

        Args:
            perm: Permutation tensor of shape (batch, n), dtype long.
            top_k: Number of moves to return.

        Returns:
            List of lists of flip values (k=2..n).
        """
        n = perm.shape[1]
        with torch.no_grad():
            logits = self.forward(perm)
            # Only consider valid flips k=2..n (indices 0..n-2)
            valid_logits = logits[:, : n - 1]
            _, indices = valid_logits.topk(min(top_k, n - 1), dim=-1)
            # Convert indices to flip values (index 0 -> k=2)
            return (indices + 2).tolist()

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
