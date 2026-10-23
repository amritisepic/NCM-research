"""Sudoku training batches in NCM's format, with validity-preserving augmentation.

Augmentations (each preserves Sudoku validity and solution uniqueness): digit relabeling, permutation of
the three bands and of rows within each band, permutation of the three stacks and of columns within each
stack, and transposition. Puzzle prompt = 91 ids (BOS, 9 rows joined by '|', BOS); target = 89 ids.
"""
from __future__ import annotations

import os

import numpy as np

from ncmr.sudoku import BOS, SEP

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rows_to_ids(board81: np.ndarray) -> np.ndarray:
    """(N, 81) -> (N, 89) with separators after each of the first 8 rows."""
    N = board81.shape[0]
    out = np.full((N, 89), SEP, dtype=np.int64)
    for r in range(9):
        out[:, r * 10:r * 10 + 9] = board81[:, r * 9:(r + 1) * 9]
    return out


def to_ncm(puz81: np.ndarray, sol81: np.ndarray):
    N = puz81.shape[0]
    prompt = np.concatenate([np.full((N, 1), BOS), _rows_to_ids(puz81), np.full((N, 1), BOS)], 1)
    return prompt.astype(np.int64), _rows_to_ids(sol81)


def augment(puz: np.ndarray, sol: np.ndarray, rng: np.random.Generator):
    """puz/sol: (N, 81) uint8. Same random transform applied to each pair (different across pairs)."""
    N = puz.shape[0]
    P = puz.reshape(N, 9, 9).astype(np.int64)
    S = sol.reshape(N, 9, 9).astype(np.int64)
    out_p, out_s = np.empty_like(P), np.empty_like(S)
    for i in range(N):
        relabel = np.concatenate([[0], rng.permutation(9) + 1])
        rows = np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])
        cols = np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])
        p, s = relabel[P[i]][rows][:, cols], relabel[S[i]][rows][:, cols]
        if rng.random() < 0.5:
            p, s = p.T, s.T
        out_p[i], out_s[i] = p, s
    return out_p.reshape(N, 81), out_s.reshape(N, 81)


class SudokuTrain:
    def __init__(self, level: str, path: str | None = None, n_val: int = 2000):
        f = path or os.path.join(ROOT, "data", "sudoku_train", f"{level}.npz")
        d = np.load(f)
        self.P, self.S = d["puzzles"], d["solutions"]
        # hold out the last n_val generated puzzles as our own validation set (never the released test set)
        self.val_P, self.val_S = self.P[-n_val:], self.S[-n_val:]
        self.P, self.S = self.P[:-n_val], self.S[:-n_val]

    def batch(self, rng: np.random.Generator, bs: int):
        idx = rng.integers(0, len(self.P), size=bs)
        p, s = augment(self.P[idx], self.S[idx], rng)
        return to_ncm(p, s)

    def val(self, n: int | None = None):
        n = n or len(self.val_P)
        return to_ncm(self.val_P[:n].astype(np.int64), self.val_S[:n].astype(np.int64))
