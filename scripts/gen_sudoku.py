"""Generate Sudoku training puzzles with unique solutions (our own open-source generator).

1. complete grid by randomized backtracking (row-major fill, shuffled digit order);
2. dig holes in a random cell order, keeping a cell blank only if the puzzle still has a unique solution,
   until the target blank count (41 easy / 46 medium / 51 hard, matching the released test sets);
   restart with a new grid if digging gets stuck;
3. reject any puzzle whose solution grid equals a test-set grid.
Training-time augmentation (digit relabeling, band/stack/row/column permutations, transpose) is applied by
the data loader, so a few tens of thousands of base puzzles suffice.

Output: data/sudoku_train/<level>.npz with puzzles (N, 81) uint8 (0 = blank) and solutions (N, 81) uint8.
Usage: python scripts/gen_sudoku.py --level hard --n 50000 --seed 0 --workers 4
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.sudoku import cells, count_solutions, is_valid_grid  # noqa: E402

BLANKS = {"easy": 41, "medium": 46, "hard": 51}


def random_grid(rng: random.Random) -> list[int]:
    board = [0] * 81
    row = [0] * 9; col = [0] * 9; box = [0] * 9

    def fill(i):
        if i == 81:
            return True
        r, c = divmod(i, 9); b = (r // 3) * 3 + c // 3
        digits = list(range(1, 10)); rng.shuffle(digits)
        for v in digits:
            m = 1 << v
            if not (row[r] & m or col[c] & m or box[b] & m):
                board[i] = v; row[r] |= m; col[c] |= m; box[b] |= m
                if fill(i + 1):
                    return True
                board[i] = 0; row[r] &= ~m; col[c] &= ~m; box[b] &= ~m
        return False

    assert fill(0)
    return board


def make_puzzle(args):
    seed, blanks = args
    rng = random.Random(seed)
    while True:
        sol = random_grid(rng)
        puz = list(sol)
        order = list(range(81)); rng.shuffle(order)
        removed = 0
        for i in order:
            v = puz[i]; puz[i] = 0
            if count_solutions(puz, limit=2) == 1:
                removed += 1
                if removed == blanks:
                    return puz, sol
            else:
                puz[i] = v
        # stuck before reaching the target: new grid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=list(BLANKS))
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from ncmr.ncm_ref_loader import safe_pickle_load
    test_grids = set()
    for lvl in BLANKS:
        p = os.path.join(ROOT, "data", "sudoku", lvl, "solutions.pickle")
        if os.path.exists(p):
            test_grids |= {tuple(cells(s)) for s in safe_pickle_load(p)}
    t0 = time.time()
    base = 1_000_003 * (1 + list(BLANKS).index(a.level)) + 7919 * a.seed
    with Pool(a.workers) as pool:
        out = pool.map(make_puzzle, [(base + i, BLANKS[a.level]) for i in range(a.n)], chunksize=256)
    keep = [(p, s) for p, s in out if tuple(s) not in test_grids]
    assert all(is_valid_grid(s) for _, s in keep[:1000])
    P = np.array([p for p, _ in keep], dtype=np.uint8); S = np.array([s for _, s in keep], dtype=np.uint8)
    os.makedirs(os.path.join(ROOT, "data", "sudoku_train"), exist_ok=True)
    f = os.path.join(ROOT, "data", "sudoku_train", f"{a.level}.npz")
    np.savez_compressed(f, puzzles=P, solutions=S, seed=a.seed)
    print(f"{a.level}: {len(keep)} puzzles ({a.n - len(keep)} rejected as test grids) in {time.time() - t0:.0f}s -> {f}")
