"""Tests for gap heuristic including BFS admissibility validation."""

from __future__ import annotations

from collections import deque
from itertools import permutations
from math import factorial

import numpy as np
import pytest

from flipstack.core.permutation import apply_flip
from flipstack.heuristics.gap import gap_features, gap_h, gap_h_batch, k_gap


def bfs_distances(n: int) -> dict[tuple[int, ...], int]:
    """Compute BFS distances from identity for all S_n permutations."""
    identity = tuple(range(n))
    distances: dict[tuple[int, ...], int] = {identity: 0}
    queue: deque[tuple[int, ...]] = deque([identity])
    while queue:
        state = queue.popleft()
        d = distances[state]
        perm = np.array(state, dtype=np.int8)
        for k in range(2, n + 1):
            child = tuple(apply_flip(perm, k).tolist())
            if child not in distances:
                distances[child] = d + 1
                queue.append(child)
    return distances


# Cache BFS results for small n
_bfs_cache: dict[int, dict[tuple[int, ...], int]] = {}


def get_bfs_distances(n: int) -> dict[tuple[int, ...], int]:
    """Get cached BFS distances."""
    if n not in _bfs_cache:
        _bfs_cache[n] = bfs_distances(n)
    return _bfs_cache[n]


class TestGapHeuristic:
    def test_identity_has_zero_gaps(self):
        perm = np.arange(5, dtype=np.int8)
        assert gap_h(perm) == 0

    def test_reversed_has_one_gap(self):
        # [4,3,2,1,0] with sentinel [4,3,2,1,0,5]
        # diffs: |4-3|=1, |3-2|=1, |2-1|=1, |1-0|=1, |0-5|=5
        # gap only at (0,5) -> 1 gap
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        assert gap_h(perm) == 1

    def test_known_example(self):
        # [3,2,0,1,4] with sentinel [3,2,0,1,4,5]
        # diffs: |3-2|=1, |2-0|=2, |0-1|=1, |1-4|=3, |4-5|=1
        # gaps at: (2,0) and (1,4) -> 2 gaps
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        assert gap_h(perm) == 2

    @pytest.mark.parametrize("n", [5, 6, 7])
    def test_admissibility_bfs(self, n):
        """Verify gap_h(state) <= true_distance(state) for all S_n."""
        distances = get_bfs_distances(n)
        violations = 0
        for state_tuple, true_dist in distances.items():
            perm = np.array(state_tuple, dtype=np.int8)
            h = gap_h(perm)
            if h > true_dist:
                violations += 1
        assert violations == 0, f"Found {violations} admissibility violations in S_{n}"

    @pytest.mark.parametrize("n", [5, 6, 7])
    def test_gap_change_bounded(self, n):
        """Verify a single flip changes gap count by at most 1."""
        for state_tuple in permutations(range(n)):
            perm = np.array(state_tuple, dtype=np.int8)
            g = gap_h(perm)
            for k in range(2, n + 1):
                child = apply_flip(perm, k)
                g_child = gap_h(child)
                diff = g_child - g
                assert diff >= -1, f"Gap decreased by {-diff} > 1: {state_tuple} flip {k}"

    @pytest.mark.slow
    def test_admissibility_s8(self):
        """BFS validation on S_8 (40320 states)."""
        distances = get_bfs_distances(8)
        assert len(distances) == factorial(8)
        for state_tuple, true_dist in distances.items():
            perm = np.array(state_tuple, dtype=np.int8)
            assert gap_h(perm) <= true_dist


class TestGapBatch:
    def test_batch_matches_individual(self):
        perms = np.array(
            [[3, 2, 0, 1, 4], [0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
            dtype=np.int8,
        )
        batch_result = gap_h_batch(perms)
        for i in range(3):
            assert batch_result[i] == gap_h(perms[i])


class TestKGap:
    def test_identity_zero(self):
        perm = np.arange(5, dtype=np.int8)
        assert k_gap(perm, 2) == 0
        assert k_gap(perm, 3) == 0

    def test_reversed(self):
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        # k=2: |4-2|=2, |3-1|=2, |2-0|=2 -> all == 2, so 0 k-gaps
        assert k_gap(perm, 2) == 0


class TestGapFeatures:
    def test_shape(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        feats = gap_features(perm)
        assert feats.shape == (3,)
        assert feats.dtype == np.float32
        assert feats[0] == gap_h(perm)
