"""PyTorch Dataset and DataLoader for mmap trie files (v2 format).

Provides efficient batched access to optimal solution data for training
policy and value networks. Decodes Lehmer code indices to permutations
on-the-fly.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from flipstack.training.trie_builder import TrieArrays, load_trie

logger = logging.getLogger(__name__)


def _get_forward_policy(trie: TrieArrays, node_id: int) -> np.ndarray:
    """Extract normalized forward policy vector for a node.

    Args:
        trie: Loaded trie data (v2 format).
        node_id: Node index.

    Returns:
        Float32 vector of length n-1, where policy[m-2] = normalized count
        for move m. Zero vector if no forward edges.
    """
    off = int(trie.fwd_offsets[node_id])
    nc = int(trie.fwd_num[node_id])
    policy = np.zeros(trie.n - 1, dtype=np.float32)
    total = 0
    for i in range(nc):
        move = int(trie.fwd_moves[off + i])
        count = int(trie.fwd_counts[off + i])
        policy[move - 2] = count
        total += count
    if total > 0:
        policy /= total
    return policy


class TrieDataset(Dataset):
    """PyTorch Dataset over a mmap trie file (v2 format).

    Each sample is (state, forward_policy, distance) where:
    - state: int64 tensor of shape (n,) for embedding layer
    - forward_policy: float32 tensor of shape (n-1,), normalized move counts
    - distance: float32 scalar, optimal distance from identity
    """

    def __init__(
        self,
        trie_path: str | Path,
        split: str = "train",
        test_frac: float = 0.2,
        seed: int = 42,
    ) -> None:
        """Initialize dataset from trie file.

        Args:
            trie_path: Path to .trie binary file (v2 format).
            split: "train" or "test".
            test_frac: Fraction of data for test split.
            seed: Random seed for split.
        """
        self.trie = load_trie(trie_path)
        self.n = self.trie.n
        self._split = split

        # Exclude identity node (node 0): distance=0, no forward edges
        all_ids = np.arange(1, self.trie.num_nodes)

        if len(all_ids) == 0:
            self._ids = np.array([], dtype=np.int64)
            return

        rng = np.random.default_rng(seed)
        depths = np.array(self.trie.depths[all_ids])

        # Split strategy: random fraction per depth bucket
        # Shallow (depth 1-5): random test_frac -> test
        # Deep (depth 6+): random test_frac of leaves + non-leaves -> test
        shallow_mask = depths <= 5
        deep_mask = ~shallow_mask

        test_mask = np.zeros(len(all_ids), dtype=bool)

        # Shallow: simple random split
        shallow_ids = np.where(shallow_mask)[0]
        if len(shallow_ids) > 1:
            n_test = max(1, int(len(shallow_ids) * test_frac))
            test_indices = rng.choice(shallow_ids, size=n_test, replace=False)
            test_mask[test_indices] = True

        # Deep: find leaves (nodes with no children = fwd_num == 0 doesn't work,
        # leaves have fwd edges but no children in trie).
        # Without backward edges, use simple random split for deep nodes too.
        deep_ids = np.where(deep_mask)[0]
        if len(deep_ids) > 1:
            n_test_deep = max(1, int(len(deep_ids) * test_frac))
            test_deep = rng.choice(deep_ids, size=n_test_deep, replace=False)
            test_mask[test_deep] = True

        if split == "test":
            self._ids = all_ids[test_mask]
        else:
            self._ids = all_ids[~test_mask]

        logger.info(
            "TrieDataset(%s): %d samples (total=%d, n=%d)",
            split, len(self._ids), len(all_ids), self.n,
        )

    def __len__(self) -> int:
        """Number of samples in this split."""
        return len(self._ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[invalid-method-override]
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            (state, policy, distance) tuple of tensors.
        """
        node_id = self._ids[idx]
        state = torch.from_numpy(
            np.array(self.trie.nodes[node_id], dtype=np.int64),
        )
        policy = torch.from_numpy(_get_forward_policy(self.trie, int(node_id)))
        distance = torch.tensor(float(self.trie.depths[node_id]), dtype=torch.float32)
        return state, policy, distance


def create_dataloader(
    trie_path: str | Path,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
    split: str = "train",
    test_frac: float = 0.2,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader from a trie file.

    Args:
        trie_path: Path to .trie binary file (v2 format).
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: Number of data loading workers.
        split: "train" or "test".
        test_frac: Fraction for test split.
        seed: Random seed.

    Returns:
        PyTorch DataLoader yielding (states, policies, distances) batches.
    """
    dataset = TrieDataset(
        trie_path, split=split, test_frac=test_frac, seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and len(dataset) > 0,
        num_workers=num_workers,
        pin_memory=True,
    )
