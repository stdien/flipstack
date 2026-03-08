"""Tests for core types, permutation ops, and I/O."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flipstack.core.io import load_competition_data
from flipstack.core.permutation import (
    apply_flip,
    apply_flip_batch,
    apply_flip_inplace,
    as_compute_dtype,
    inverse_perm,
    is_sorted,
    validate_perm,
)
from flipstack.core.types import SolverResult


@st.composite
def _perm_strategy(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(min_value=2, max_value=20))
    values = list(range(n))
    perm = draw(st.permutations(values))
    return np.array(perm, dtype=np.int8)


perms = _perm_strategy()  # type: ignore[missing-argument]


class TestApplyFlip:
    def test_flip_basic(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        result = apply_flip(perm, 4)
        np.testing.assert_array_equal(result, [1, 0, 2, 3, 4])

    def test_flip_full(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        result = apply_flip(perm, 5)
        np.testing.assert_array_equal(result, [4, 1, 0, 2, 3])

    def test_flip_2(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        result = apply_flip(perm, 2)
        np.testing.assert_array_equal(result, [2, 3, 0, 1, 4])

    @given(perm=perms, data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_flip_is_involution(self, perm, data):
        k = data.draw(st.integers(min_value=2, max_value=len(perm)))
        result = apply_flip(apply_flip(perm, k), k)
        np.testing.assert_array_equal(result, perm)

    def test_flip_does_not_mutate(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        original = perm.copy()
        apply_flip(perm, 3)
        np.testing.assert_array_equal(perm, original)


class TestApplyFlipInplace:
    def test_inplace_matches_copy(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        expected = apply_flip(perm, 3)
        apply_flip_inplace(perm, 3)
        np.testing.assert_array_equal(perm, expected)


class TestApplyFlipBatch:
    @given(perm=perms, data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_batch_matches_individual(self, perm, data):
        k = data.draw(st.integers(min_value=2, max_value=len(perm)))
        batch = np.stack([perm, perm, perm])
        result = apply_flip_batch(batch, k)
        expected = apply_flip(perm, k)
        for i in range(3):
            np.testing.assert_array_equal(result[i], expected)


class TestIsSorted:
    def test_identity(self):
        assert is_sorted(np.arange(5, dtype=np.int8))

    def test_not_sorted(self):
        assert not is_sorted(np.array([1, 0, 2, 3, 4], dtype=np.int8))

    def test_single(self):
        assert is_sorted(np.array([0], dtype=np.int8))


class TestValidatePerm:
    def test_valid(self):
        validate_perm(np.array([2, 0, 1], dtype=np.int8))

    def test_duplicates(self):
        with pytest.raises(ValueError, match="Duplicate"):
            validate_perm(np.array([0, 0, 1], dtype=np.int8))

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="Values must be"):
            validate_perm(np.array([0, 1, 5], dtype=np.int8))

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="Expected length"):
            validate_perm(np.array([0, 1, 2], dtype=np.int8), expected_n=5)


class TestInversePerm:
    @given(perm=perms)
    @settings(max_examples=200, deadline=None)
    def test_inverse_property(self, perm):
        inv = inverse_perm(perm)
        identity = np.arange(len(perm), dtype=perm.dtype)
        np.testing.assert_array_equal(perm[inv], identity)
        np.testing.assert_array_equal(inv[perm], identity)

    def test_inverse_of_inverse(self):
        perm = np.array([2, 0, 3, 1], dtype=np.int8)
        np.testing.assert_array_equal(inverse_perm(inverse_perm(perm)), perm)


class TestAsComputeDtype:
    def test_casts_to_int16(self):
        perm = np.array([0, 1, 2], dtype=np.int8)
        result = as_compute_dtype(perm)
        assert result.dtype == np.int16


class TestIO:
    def test_load_competition_data(self):
        data = load_competition_data("data/test.csv")
        assert len(data) == 2405
        row_id, n, perm = data[0]
        assert row_id == 0
        assert n == 5
        assert len(perm) == 5
        assert perm.dtype == np.int8

    def test_all_perms_valid(self):
        data = load_competition_data("data/test.csv")
        for _row_id, n, perm in data:
            validate_perm(perm, expected_n=n)

    def test_n_range(self):
        data = load_competition_data("data/test.csv")
        ns = {n for _, n, _ in data}
        assert min(ns) == 5
        assert max(ns) == 100


class TestSolverResult:
    def test_solution_string(self):
        perm = np.array([3, 2, 0, 1, 4], dtype=np.int8)
        r = SolverResult(perm_id=0, flips=[4, 2], original_perm=perm)
        assert r.solution_string() == "R4.R2"
        assert r.length == 2

    def test_empty_solution(self):
        perm = np.arange(5, dtype=np.int8)
        r = SolverResult(perm_id=0, flips=[], original_perm=perm)
        assert r.solution_string() == ""
        assert r.length == 0
