"""Trie builder and serializer for optimal pancake sorting solutions.

Builds a reversed trie (identity -> permutations) from DFS solver output.
Serializes to compact binary format v2 for mmap loading by the DataLoader.

v2 format: raw permutations (n bytes each), compact forward edges
(uint8 moves + int32 counts), no backward edges. Works for any n.
"""
from __future__ import annotations

import logging
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

TRIE_MAGIC = b"TRIE"


@dataclass
class TrieArrays:
    """Flat-array representation of a solution trie (v2 format)."""

    n: int
    max_depth: int
    nodes: np.ndarray         # (num_nodes, n) int8 — raw permutations
    depths: np.ndarray        # (num_nodes,) int16
    fwd_offsets: np.ndarray   # (num_nodes,) int32
    fwd_num: np.ndarray       # (num_nodes,) uint8
    fwd_moves: np.ndarray     # (num_fwd_edges,) uint8
    fwd_counts: np.ndarray    # (num_fwd_edges,) int32

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the trie."""
        return len(self.depths)

    @property
    def num_fwd_edges(self) -> int:
        """Number of forward edge entries."""
        return len(self.fwd_moves)


def _ingest_solutions(
    solutions_iter: Iterable[tuple[list[int], list[list[int]]]],
    n: int,
    identity: np.ndarray,
    get_or_create_node: Callable[[np.ndarray, int], int],
    fwd_acc: dict[int, dict[int, int]],
) -> tuple[int, int]:
    """Walk all solutions and populate trie nodes/edges.

    Returns:
        (num_solutions, max_depth).
    """
    num_solutions = 0
    max_depth = 0

    for _perm, flip_sequences in solutions_iter:
        for flips in flip_sequences:
            if not flips:
                continue
            num_solutions += 1
            sol_len = len(flips)
            if sol_len > max_depth:
                max_depth = sol_len

            reversed_flips = list(reversed(flips))
            state = identity.copy()
            parent_id = 0

            for step, move in builtins_enumerate(reversed_flips):
                if move < 2 or move > n:
                    msg = f"Invalid move {move}: must be in [2, {n}]"
                    raise ValueError(msg)

                child_depth = step + 1
                state[:move] = state[move - 1 :: -1] if move == n else state[:move][::-1]
                child_id = get_or_create_node(state, child_depth)

                fwd_acc[child_id][move] += 1
                parent_id = child_id  # noqa: F841

    return num_solutions, max_depth


def build_trie(
    solutions_iter: Iterable[tuple[list[int], list[list[int]]]],
    n: int,
) -> TrieArrays:
    """Build a reversed trie from solver solutions.

    Args:
        solutions_iter: Iterable of (perm, list_of_flip_sequences).
            Each flip sequence is a list of ints [f1, f2, ..., fL].
        n: Permutation size.

    Returns:
        TrieArrays with flat-array trie representation (v2 format).

    Raises:
        ValueError: If depth inconsistency or invalid moves detected.
    """
    # Node storage: key = state bytes, value = node_id
    state_to_id: dict[bytes, int] = {}
    node_states: list[np.ndarray] = []
    node_depths: list[int] = []

    # Edge accumulator: node_id -> {move: count}
    fwd_acc: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # Identity is always node 0
    identity = np.arange(n, dtype=np.int8)
    identity_key = identity.tobytes()
    state_to_id[identity_key] = 0
    node_states.append(identity.copy())
    node_depths.append(0)

    def _get_or_create_node(state: np.ndarray, depth: int) -> int:
        key = state.tobytes()
        if key in state_to_id:
            nid = state_to_id[key]
            if node_depths[nid] != depth:
                msg = (
                    f"Depth inconsistency for state: "
                    f"existing={node_depths[nid]}, new={depth}"
                )
                raise ValueError(msg)
            return nid
        nid = len(node_states)
        state_to_id[key] = nid
        node_states.append(state.copy())
        node_depths.append(depth)
        return nid

    num_solutions, max_depth = _ingest_solutions(
        solutions_iter, n, identity, _get_or_create_node, fwd_acc,
    )

    logger.info(
        "Built trie: n=%d, %d nodes, %d solutions, max_depth=%d",
        n, len(node_states), num_solutions, max_depth,
    )

    num_nodes = len(node_states)
    nodes = np.array(node_states, dtype=np.int8)
    depths = np.array(node_depths, dtype=np.int16)

    # Build forward edges (compact: separate moves and counts arrays)
    fwd_move_list: list[int] = []
    fwd_count_list: list[int] = []
    fwd_offsets = np.zeros(num_nodes, dtype=np.int32)
    fwd_num = np.zeros(num_nodes, dtype=np.uint8)

    for nid in range(num_nodes):
        fwd_offsets[nid] = len(fwd_move_list)
        edges = fwd_acc.get(nid, {})
        fwd_num[nid] = len(edges)
        for move, count in sorted(edges.items()):
            fwd_move_list.append(move)
            fwd_count_list.append(count)

    fwd_moves = np.array(fwd_move_list, dtype=np.uint8) if fwd_move_list else np.empty(0, dtype=np.uint8)
    fwd_counts = np.array(fwd_count_list, dtype=np.int32) if fwd_count_list else np.empty(0, dtype=np.int32)

    return TrieArrays(
        n=n,
        max_depth=max_depth,
        nodes=nodes,
        depths=depths,
        fwd_offsets=fwd_offsets,
        fwd_num=fwd_num,
        fwd_moves=fwd_moves,
        fwd_counts=fwd_counts,
    )


# Alias to avoid shadowing
builtins_enumerate = enumerate

HEADER_SIZE = 24


def save_trie(trie: TrieArrays, path: Path | str) -> None:
    """Save trie to binary file.

    Args:
        trie: TrieArrays to serialize.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        # Header (24 bytes): magic(4) + n(1) + max_depth(1) + pad(2) + num_nodes(8) + num_fwd(8)
        header = struct.pack(
            "<4sBBHqq",
            TRIE_MAGIC,
            trie.n,
            trie.max_depth,
            0,  # pad
            trie.num_nodes,
            trie.num_fwd_edges,
        )
        f.write(header)

        # Data sections (all little-endian, contiguous)
        f.write(trie.nodes.astype("<i1").tobytes())
        f.write(trie.depths.astype("<i2").tobytes())
        f.write(trie.fwd_offsets.astype("<i4").tobytes())
        f.write(trie.fwd_num.astype("u1").tobytes())
        f.write(trie.fwd_moves.astype("u1").tobytes())
        f.write(trie.fwd_counts.astype("<i4").tobytes())

    logger.info("Saved trie to %s (%d bytes)", path, path.stat().st_size)


def load_trie(path: Path | str, mmap: bool = True) -> TrieArrays:
    """Load trie from binary file.

    Args:
        path: Path to .trie file.
        mmap: If True, use memory-mapped I/O (recommended for large files).

    Returns:
        TrieArrays with loaded data.

    Raises:
        ValueError: If file is corrupt or has invalid data.
    """
    path = Path(path)
    file_size = path.stat().st_size

    with path.open("rb") as f:
        raw_header = f.read(HEADER_SIZE)

    if len(raw_header) < HEADER_SIZE:
        msg = f"File too small for header: {len(raw_header)} < {HEADER_SIZE}"
        raise ValueError(msg)

    magic, n, max_depth, _pad, num_nodes, num_fwd = (
        struct.unpack("<4sBBHqq", raw_header)
    )

    if magic != TRIE_MAGIC:
        msg = f"Invalid magic: {magic!r}, expected {TRIE_MAGIC!r}"
        raise ValueError(msg)
    if n < 2:
        msg = f"Invalid n: {n}"
        raise ValueError(msg)

    # Compute expected sizes
    nodes_size = num_nodes * n
    depths_size = num_nodes * 2
    fwd_off_size = num_nodes * 4
    fwd_num_size = num_nodes
    fwd_moves_size = num_fwd
    fwd_counts_size = num_fwd * 4

    expected_size = (
        HEADER_SIZE + nodes_size + depths_size
        + fwd_off_size + fwd_num_size
        + fwd_moves_size + fwd_counts_size
    )

    if file_size != expected_size:
        msg = f"File size mismatch: {file_size} != {expected_size}"
        raise ValueError(msg)

    # Read arrays
    offset = HEADER_SIZE

    def _read_array(
        dtype: str, count: int, shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        nonlocal offset
        nbytes = count * np.dtype(dtype).itemsize
        if mmap and count > 0:
            arr = np.memmap(
                path, dtype=dtype, mode="r", offset=offset,
                shape=shape or (count,),
            )
        elif count > 0:
            arr = np.fromfile(path, dtype=dtype, count=count, offset=offset)
            if shape is not None:
                arr = arr.reshape(shape)
        else:
            arr = np.empty(shape or (0,), dtype=dtype)
        offset += nbytes
        return arr

    nodes = _read_array("<i1", num_nodes * n, shape=(num_nodes, n))
    depths = _read_array("<i2", num_nodes)
    fwd_offsets = _read_array("<i4", num_nodes)
    fwd_num_arr = _read_array("u1", num_nodes)
    fwd_moves = _read_array("u1", num_fwd)
    fwd_counts = _read_array("<i4", num_fwd)

    # Validate
    _validate_loaded_trie(n, num_nodes, num_fwd, fwd_offsets, fwd_num_arr,
                          fwd_moves, fwd_counts, depths)

    return TrieArrays(
        n=n,
        max_depth=max_depth,
        nodes=nodes,
        depths=depths,
        fwd_offsets=fwd_offsets,
        fwd_num=fwd_num_arr,
        fwd_moves=fwd_moves,
        fwd_counts=fwd_counts,
    )


def _validate_loaded_trie(
    n: int,
    num_nodes: int,
    num_fwd: int,
    fwd_offsets: np.ndarray,
    fwd_num: np.ndarray,
    fwd_moves: np.ndarray,
    fwd_counts: np.ndarray,
    depths: np.ndarray,
) -> None:
    """Validate trie data after loading."""
    # Check identity is node 0
    if num_nodes > 0 and depths[0] != 0:
        msg = f"Node 0 depth must be 0 (identity), got {depths[0]}"
        raise ValueError(msg)

    # Check offset+num doesn't exceed total edges
    for nid in range(num_nodes):
        fwd_end = int(fwd_offsets[nid]) + int(fwd_num[nid])
        if fwd_end > num_fwd:
            msg = f"Node {nid}: fwd offset+num ({fwd_end}) > total ({num_fwd})"
            raise ValueError(msg)

    # Check move bounds and counts
    if num_fwd > 0:
        if np.any(fwd_moves < 2) or np.any(fwd_moves > n):
            msg = f"Forward edge moves out of range [2, {n}]"
            raise ValueError(msg)
        if np.any(fwd_counts < 0):
            msg = "Negative forward edge counts"
            raise ValueError(msg)
