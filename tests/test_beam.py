"""Tests for beam search."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from flipstack.core.permutation import apply_flip, is_sorted
from flipstack.search.beam import beam_search, iterated_beam_search
from tests.test_gap import get_bfs_distances


def _verify_solution(perm: np.ndarray, flips: list[int]) -> bool:
    """Verify that applying flips to perm produces the identity."""
    p = perm.copy()
    for k in flips:
        p = apply_flip(p, k)
    return is_sorted(p)


class TestBeamSearch:
    def test_identity(self):
        perm = np.arange(5, dtype=np.int8)
        result = beam_search(perm)
        assert result == []

    def test_simple_perm(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        result = beam_search(perm, beam_width=64)
        assert result is not None
        assert _verify_solution(perm, result)

    def test_solves_all_s5(self):
        distances = get_bfs_distances(5)
        for state_tuple in permutations(range(5)):
            perm = np.array(state_tuple, dtype=np.int8)
            result = beam_search(perm, beam_width=256, timeout=5.0)
            assert result is not None, f"Failed to solve {state_tuple}"
            assert _verify_solution(perm, result)
            # Beam search is not optimal but should find something reasonable
            optimal = distances[state_tuple]
            assert len(result) <= optimal * 3 + 5, f"{state_tuple}: got {len(result)}, optimal {optimal}"

    def test_solves_s7_samples(self):
        distances = get_bfs_distances(7)
        # Test hardest S_7 permutations (distance = 8)
        hard_perms = [s for s, d in distances.items() if d == 8]
        for state_tuple in hard_perms[:5]:
            perm = np.array(state_tuple, dtype=np.int8)
            result = beam_search(perm, beam_width=512, timeout=10.0)
            assert result is not None
            assert _verify_solution(perm, result)

    def test_no_filter_mode(self):
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        result = beam_search(perm, beam_width=64, use_filter=False)
        assert result is not None
        assert _verify_solution(perm, result)


class TestIteratedBeamSearch:
    def test_basic(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        result = iterated_beam_search(perm, beam_widths=[32, 128], timeout=5.0)
        assert result is not None
        assert _verify_solution(perm, result)
