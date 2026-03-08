"""Tests for bidirectional beam search."""

from itertools import permutations

import numpy as np

from flipstack.core.permutation import apply_flip, is_sorted
from flipstack.search.beam_bidir import bidir_beam_search


def _verify_solution(perm: np.ndarray, flips: list[int]) -> bool:
    """Verify a flip sequence sorts the permutation."""
    p = perm.copy()
    for k in flips:
        p = apply_flip(p, k)
    return bool(is_sorted(p))


class TestBidirBeamSearch:
    def test_identity(self) -> None:
        perm = np.arange(5, dtype=np.int8)
        result = bidir_beam_search(perm)
        assert result == []

    def test_solves_simple(self) -> None:
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        result = bidir_beam_search(perm, beam_width=64)
        assert result is not None
        assert _verify_solution(perm, result)

    def test_solves_all_s5(self) -> None:
        """Bidir beam should solve all S_5 permutations."""
        for p in permutations(range(5)):
            perm = np.array(p, dtype=np.int8)
            result = bidir_beam_search(perm, beam_width=64, timeout=5.0)
            assert result is not None, f"Failed for {p}"
            assert _verify_solution(perm, result), f"Invalid solution for {p}"

    def test_early_termination_on_stall(self) -> None:
        """With tiny beam width, search should terminate early rather than loop."""
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        # Very restrictive beam should either solve or terminate quickly
        bidir_beam_search(perm, beam_width=2, max_steps=1000, timeout=2.0)
        # Just checking it doesn't hang — result may or may not be found


class TestGapReduce:
    def test_identity(self) -> None:
        from flipstack.search.gap_reduce import gap_reduce_solve

        perm = np.arange(5, dtype=np.int8)
        result = gap_reduce_solve(perm)
        assert result == []

    def test_solves_reversed(self) -> None:
        from flipstack.search.gap_reduce import gap_reduce_solve

        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        result = gap_reduce_solve(perm, timeout=5.0)
        assert result is not None
        assert _verify_solution(perm, result)

    def test_solves_most_s5(self) -> None:
        """Gap reduce should solve most S_5 permutations."""
        from flipstack.search.gap_reduce import gap_reduce_solve

        solved = 0
        total = 0
        for p in permutations(range(5)):
            perm = np.array(p, dtype=np.int8)
            total += 1
            result = gap_reduce_solve(perm, timeout=1.0)
            if result is not None:
                assert _verify_solution(perm, result)
                solved += 1
        # Should solve at least 90%
        assert solved / total > 0.9, f"Only solved {solved}/{total}"
