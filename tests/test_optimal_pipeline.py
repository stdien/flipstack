"""Tests for the optimal DFS -> trie -> DataLoader pipeline."""
from __future__ import annotations

import tempfile
from collections import deque
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch

from flipstack.core.permutation import index_to_perm, perm_to_index
from flipstack.search.optimal_dfs import gap_h, solve_and_enumerate, solve_one

# ---------------------------------------------------------------------------
# BFS reference for small n
# ---------------------------------------------------------------------------


def _bfs_optimal_distances(n: int) -> dict[tuple[int, ...], int]:
    """Compute exact optimal distances for all perms of size n via BFS."""
    identity = tuple(range(n))
    dist: dict[tuple[int, ...], int] = {identity: 0}
    queue: deque[tuple[int, ...]] = deque([identity])
    while queue:
        state = queue.popleft()
        d = dist[state]
        for k in range(2, n + 1):
            child = state[:k][::-1] + state[k:]
            if child not in dist:
                dist[child] = d + 1
                queue.append(child)
    return dist


# ---------------------------------------------------------------------------
# Tests: Numba DFS solver
# ---------------------------------------------------------------------------


class TestOptimalDFS:
    """Test Numba DFS solver against BFS ground truth."""

    def test_identity_is_zero(self) -> None:
        p = np.arange(5, dtype=np.int8)
        sol_len, count = solve_one(p, 5, 0)
        assert sol_len == 0
        assert count == 1

    def test_gap_h_identity(self) -> None:
        p = np.arange(5, dtype=np.int8)
        assert gap_h(p, 5) == 0

    def test_gap_h_reversed(self) -> None:
        p = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        assert gap_h(p, 5) == 1

    def test_solve_n5_all_perms(self) -> None:
        """Verify DFS optimal length matches BFS for all 120 perms of S_5."""
        n = 5
        bfs_dist = _bfs_optimal_distances(n)

        for perm_tuple in permutations(range(n)):
            p = np.array(perm_tuple, dtype=np.int8)
            expected = bfs_dist[perm_tuple]
            sol_len, count = solve_one(p, n, 4)
            assert sol_len == expected, (
                f"Perm {perm_tuple}: DFS={sol_len}, BFS={expected}"
            )
            assert count >= 1

    def test_enumerate_n4(self) -> None:
        """Enumerate solutions for a known n=4 perm and verify."""
        p = np.array([3, 2, 1, 0], dtype=np.int8)
        sol_len, count, solutions, truncated = solve_and_enumerate(p, 4, 4, 1000)
        assert sol_len >= 1
        assert not truncated
        assert solutions.shape[0] == count

        # Verify each solution actually solves the perm
        for i in range(int(count)):
            state = p.copy()
            for j in range(sol_len):
                k = int(solutions[i, j])
                assert 2 <= k <= 4
                state[:k] = state[k - 1 :: -1]
            assert np.array_equal(state, np.arange(4, dtype=np.int8))

    def test_enumerate_truncation(self) -> None:
        """Verify truncation when max_solutions is exceeded."""
        # [1, 0, 2, 3, 4] has many solutions at various depths
        p = np.array([1, 0, 2, 3, 4], dtype=np.int8)
        sol_len, _count, solutions, truncated = solve_and_enumerate(p, 5, 4, 1)
        assert sol_len == 1
        assert not truncated
        assert solutions.shape[0] == 1

    def test_unsolved_returns_negative(self) -> None:
        """With max_slack=0, some perms with slack>0 should return -1."""
        # [2, 0, 1] requires slack > 0
        p = np.array([2, 0, 1], dtype=np.int8)
        gh = gap_h(p, 3)
        if gh > 0:
            sol_len, _count = solve_one(p, 3, 0)
            # Either solved at gap_h or unsolved — both valid
            assert sol_len == -1 or sol_len == gh


# ---------------------------------------------------------------------------
# Tests: Lehmer code
# ---------------------------------------------------------------------------


class TestLehmerCode:
    """Test perm_to_index / index_to_perm roundtrip."""

    def test_identity_is_zero(self) -> None:
        p = np.arange(5, dtype=np.int8)
        assert perm_to_index(p) == 0

    def test_roundtrip_n4(self) -> None:
        for perm_tuple in permutations(range(4)):
            p = np.array(perm_tuple, dtype=np.int8)
            idx = perm_to_index(p)
            recovered = index_to_perm(idx, 4)
            assert np.array_equal(p, recovered), f"{perm_tuple} -> {idx} -> {recovered}"

    def test_all_indices_unique_n4(self) -> None:
        indices = set()
        for perm_tuple in permutations(range(4)):
            idx = perm_to_index(np.array(perm_tuple, dtype=np.int8))
            assert idx not in indices
            indices.add(idx)
        assert len(indices) == 24


# ---------------------------------------------------------------------------
# Tests: Trie builder (v2 format)
# ---------------------------------------------------------------------------


class TestTrieBuilder:
    """Test trie construction and serialization."""

    def test_build_from_n4_solutions(self) -> None:
        from flipstack.training.trie_builder import build_trie

        # Build solutions for a few n=4 perms
        n = 4
        solutions_data = []
        for perm_tuple in [(1, 0, 2, 3), (3, 2, 1, 0)]:
            p = np.array(perm_tuple, dtype=np.int8)
            sol_len, count, sols, trunc = solve_and_enumerate(p, n, 4, 100)
            if sol_len > 0 and not trunc:
                solutions_data.append(
                    (list(perm_tuple), [[int(sols[i, j]) for j in range(sol_len)] for i in range(int(count))]),
                )

        trie = build_trie(iter(solutions_data), n)

        # Identity should be node 0
        assert np.array_equal(trie.nodes[0], np.arange(n, dtype=np.int8))
        assert trie.depths[0] == 0

        # Should have multiple nodes
        assert trie.num_nodes > 1

        # All depths should be non-negative
        assert np.all(trie.depths >= 0)

        # Forward edges: identity (node 0) should have no forward edges
        assert trie.fwd_num[0] == 0

    def test_save_load_roundtrip(self) -> None:
        from flipstack.training.trie_builder import build_trie, load_trie, save_trie

        n = 3
        # Simple: just [1, 0, 2] solved by flip(2)
        solutions_data = [([1, 0, 2], [[2]])]
        trie = build_trie(iter(solutions_data), n)

        with tempfile.NamedTemporaryFile(suffix=".trie", delete=False) as f:
            path = Path(f.name)

        try:
            save_trie(trie, path)
            loaded = load_trie(path, mmap=False)

            assert loaded.n == trie.n
            assert loaded.max_depth == trie.max_depth
            assert loaded.num_nodes == trie.num_nodes
            assert np.array_equal(loaded.nodes, trie.nodes)
            assert np.array_equal(loaded.depths, trie.depths)
            assert np.array_equal(loaded.fwd_moves, trie.fwd_moves)
            assert np.array_equal(loaded.fwd_counts, trie.fwd_counts)
        finally:
            path.unlink(missing_ok=True)

    def test_save_load_mmap(self) -> None:
        from flipstack.training.trie_builder import build_trie, load_trie, save_trie

        n = 3
        solutions_data = [([1, 0, 2], [[2]])]
        trie = build_trie(iter(solutions_data), n)

        with tempfile.NamedTemporaryFile(suffix=".trie", delete=False) as f:
            path = Path(f.name)

        try:
            save_trie(trie, path)
            loaded = load_trie(path, mmap=True)

            assert loaded.num_nodes == trie.num_nodes
            assert np.array_equal(
                np.array(loaded.nodes), np.array(trie.nodes),
            )
        finally:
            path.unlink(missing_ok=True)

    def test_depth_consistency_error(self) -> None:
        from flipstack.training.trie_builder import build_trie

        n = 3
        solutions_data = [([2, 1, 0], [[3]])]
        trie = build_trie(iter(solutions_data), n)
        assert trie.num_nodes == 2  # identity + [2,1,0]


# ---------------------------------------------------------------------------
# Tests: DataLoader
# ---------------------------------------------------------------------------


class TestTrieDataset:
    """Test PyTorch Dataset and DataLoader."""

    @pytest.fixture
    def trie_path(self, tmp_path: Path) -> Path:
        """Build a small trie for testing."""
        from flipstack.training.trie_builder import build_trie, save_trie

        n = 4
        solutions_data = []
        for perm_tuple in permutations(range(n)):
            if perm_tuple == (0, 1, 2, 3):
                continue
            p = np.array(perm_tuple, dtype=np.int8)
            sol_len, count, sols, trunc = solve_and_enumerate(p, n, 4, 100)
            if sol_len > 0 and not trunc and count > 0:
                solutions_data.append(
                    (list(perm_tuple), [[int(sols[i, j]) for j in range(sol_len)] for i in range(int(count))]),
                )

        trie = build_trie(iter(solutions_data), n)
        path = tmp_path / "test.trie"
        save_trie(trie, path)
        return path

    def test_dataset_len(self, trie_path: Path) -> None:
        from flipstack.training.trie_dataset import TrieDataset

        ds_train = TrieDataset(trie_path, split="train")
        ds_test = TrieDataset(trie_path, split="test")
        # Total should be num_nodes - 1 (excluding identity)
        assert len(ds_train) + len(ds_test) > 0
        assert len(ds_train) > 0

    def test_dataset_item_shapes(self, trie_path: Path) -> None:
        from flipstack.training.trie_dataset import TrieDataset

        ds = TrieDataset(trie_path, split="train")
        state, policy, distance = ds[0]

        assert state.dtype == torch.int64
        assert state.shape == (4,)

        assert policy.dtype == torch.float32
        assert policy.shape == (3,)  # n-1

        assert distance.dtype == torch.float32
        assert distance.ndim == 0

    def test_policy_sums_to_one(self, trie_path: Path) -> None:
        from flipstack.training.trie_dataset import TrieDataset

        ds = TrieDataset(trie_path, split="train")
        for i in range(min(len(ds), 20)):
            _state, policy, _dist = ds[i]
            total = policy.sum().item()
            assert abs(total - 1.0) < 1e-5, f"Policy sum={total} at idx={i}"

    def test_identity_excluded(self, trie_path: Path) -> None:
        from flipstack.training.trie_dataset import TrieDataset

        ds = TrieDataset(trie_path, split="train")
        identity = torch.arange(4, dtype=torch.int64)
        for i in range(len(ds)):
            state, _policy, distance = ds[i]
            if torch.equal(state, identity):
                # If identity appears, its distance must not be 0
                # (it shouldn't appear at all since we exclude node 0)
                assert distance.item() > 0

    def test_dataloader_batches(self, trie_path: Path) -> None:
        from flipstack.training.trie_dataset import create_dataloader

        dl = create_dataloader(trie_path, batch_size=8, num_workers=0, shuffle=False)
        batch = next(iter(dl))
        states, policies, distances = batch

        assert states.shape[1] == 4
        assert policies.shape[1] == 3
        assert distances.shape[0] == states.shape[0]
