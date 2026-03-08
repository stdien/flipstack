"""Tests for GPU beam search (using CPU device)."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch

from flipstack.core.permutation import apply_flip, is_sorted
from flipstack.search.beam_gpu import apply_flip_gpu, gap_h_gpu, gpu_beam_search


def _verify_solution(perm: np.ndarray, flips: list[int]) -> bool:
    p = perm.copy()
    for k in flips:
        p = apply_flip(p, k)
    return is_sorted(p)


class TestApplyFlipGpu:
    def test_matches_cpu(self):
        perm = np.array([3, 1, 4, 0, 2], dtype=np.int8)
        states = torch.from_numpy(perm).unsqueeze(0)
        for k in range(2, 6):
            gpu_result = apply_flip_gpu(states, k)[0].numpy()
            cpu_result = apply_flip(perm, k)
            np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_batch(self):
        perms = np.array([[3, 1, 4, 0, 2], [0, 1, 2, 3, 4]], dtype=np.int8)
        states = torch.from_numpy(perms)
        result = apply_flip_gpu(states, 3)
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[0].numpy(), apply_flip(perms[0], 3))
        np.testing.assert_array_equal(result[1].numpy(), apply_flip(perms[1], 3))


class TestGapHGpu:
    def test_identity_zero(self):
        perm = torch.arange(5).unsqueeze(0).to(torch.int8)
        assert gap_h_gpu(perm).item() == 0.0

    def test_reversed_perm(self):
        perm = torch.tensor([[4, 3, 2, 1, 0]], dtype=torch.int8)
        gap = gap_h_gpu(perm).item()
        assert gap == 1.0  # Only gap is between 0 and sentinel 5

    def test_batch_matches_single(self):
        from flipstack.heuristics.gap import gap_h

        perms_np = np.array(
            [
                [3, 1, 4, 0, 2],
                [4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4],
            ],
            dtype=np.int8,
        )
        states = torch.from_numpy(perms_np)
        gpu_gaps = gap_h_gpu(states).tolist()
        for i in range(3):
            cpu_gap = gap_h(perms_np[i])
            assert gpu_gaps[i] == cpu_gap, f"Mismatch at {i}: {gpu_gaps[i]} != {cpu_gap}"


class TestGpuBeamSearch:
    def test_identity(self):
        perm = np.arange(5, dtype=np.int8)
        result = gpu_beam_search(perm, device="cpu")
        assert result == []

    def test_simple(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        result = gpu_beam_search(perm, beam_width=64, device="cpu", timeout=5.0)
        assert result is not None
        assert _verify_solution(perm, result)

    def test_reversed(self):
        perm = np.array([4, 3, 2, 1, 0], dtype=np.int8)
        result = gpu_beam_search(perm, beam_width=64, device="cpu", timeout=5.0)
        assert result is not None
        assert _verify_solution(perm, result)

    def test_solves_all_s5(self):
        for state_tuple in permutations(range(5)):
            perm = np.array(state_tuple, dtype=np.int8)
            result = gpu_beam_search(perm, beam_width=128, device="cpu", timeout=5.0)
            assert result is not None, f"Failed: {state_tuple}"
            assert _verify_solution(perm, result)

    def test_with_move_filter(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        result = gpu_beam_search(
            perm,
            beam_width=64,
            max_moves=3,
            device="cpu",
            timeout=5.0,
        )
        assert result is not None
        assert _verify_solution(perm, result)
