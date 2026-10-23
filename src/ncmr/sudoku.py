"""Sudoku utilities: NCM token format, rule checks, and an exact solution counter.

NCM format (from HazyResearch/ncm tokenization/sudoku_tokenizer.py):
  digits 1-9 -> ids 1-9, blank -> 0, row separator '|' -> 10, BOS -> 11.
  puzzle prompt: 91 ids = BOS + 9 rows of 9 cells joined by 8 separators + BOS.
  solution: 89 ids = 9 rows joined by 8 separators (no BOS).
"""
from __future__ import annotations

SEP, BOS = 10, 11


def cells(ids) -> list[int] | None:
    """Return the 81 cell values (0 = blank) from NCM puzzle/solution ids, or None if malformed."""
    ids = [int(t) for t in ids]
    if ids and ids[0] == BOS:
        ids = ids[1:]
    if ids and ids[-1] == BOS:
        ids = ids[:-1]
    if len(ids) == 89:
        if any(ids[i] != SEP for i in range(9, 89, 10)):
            return None
        ids = [t for i, t in enumerate(ids) if (i + 1) % 10 != 0]
    if len(ids) != 81 or any(not 0 <= t <= 9 for t in ids):
        return None
    return ids


def to_solution_ids(board: list[int]) -> list[int]:
    out = []
    for r in range(9):
        out += board[r * 9:(r + 1) * 9]
        if r < 8:
            out.append(SEP)
    return out


ROWS = [[r * 9 + c for c in range(9)] for r in range(9)]
COLS = [[r * 9 + c for r in range(9)] for c in range(9)]
BOXES = [[(br + dr) * 9 + bc + dc for dr in range(3) for dc in range(3)]
         for br in (0, 3, 6) for bc in (0, 3, 6)]
UNITS = ROWS + COLS + BOXES
PEERS = [sorted({j for u in UNITS if i in u for j in u if j != i}) for i in range(81)]


def is_valid_grid(board) -> bool:
    """All 81 cells filled and every row/column/box is a permutation of 1..9."""
    if board is None or len(board) != 81 or 0 in board:
        return False
    return all(sorted(board[i] for i in u) == list(range(1, 10)) for u in UNITS)


def consistent_with_clues(board, puzzle) -> bool:
    return all(p == 0 or p == b for p, b in zip(puzzle, board))


def is_solution(board, puzzle) -> bool:
    return is_valid_grid(board) and consistent_with_clues(board, puzzle)


def count_solutions(puzzle: list[int], limit: int = 2) -> int:
    """Count completions of `puzzle` (81 ints, 0 = blank), stopping at `limit`. Bitmask DFS, MRV order."""
    board = [int(v) for v in puzzle]
    full = 0x3FE  # bits 1..9
    row = [0] * 9; col = [0] * 9; box = [0] * 9
    for i, v in enumerate(board):
        if v:
            r, c = divmod(i, 9); b = (r // 3) * 3 + c // 3; m = 1 << v
            if row[r] & m or col[c] & m or box[b] & m:
                return 0
            row[r] |= m; col[c] |= m; box[b] |= m
    empties = [i for i, v in enumerate(board) if v == 0]
    count = 0

    def dfs() -> bool:
        nonlocal count
        best, best_mask, best_n = -1, 0, 10
        for i in empties:
            if board[i]:
                continue
            r, c = divmod(i, 9); b = (r // 3) * 3 + c // 3
            mask = full & ~(row[r] | col[c] | box[b])
            n = bin(mask).count("1")
            if n < best_n:
                best, best_mask, best_n = i, mask, n
                if n <= 1:
                    break
        if best < 0:
            count += 1
            return count >= limit
        if best_n == 0:
            return False
        r, c = divmod(best, 9); b = (r // 3) * 3 + c // 3
        m = best_mask
        while m:
            bit = m & -m; m ^= bit
            v = bit.bit_length() - 1
            board[best] = v; row[r] |= bit; col[c] |= bit; box[b] |= bit
            if dfs():
                return True
            board[best] = 0; row[r] &= ~bit; col[c] &= ~bit; box[b] &= ~bit
        return False

    dfs()
    return count


def propagation_rounds(puzzle: list[int], max_rounds: int = 81) -> tuple[int, bool]:
    """Solve by parallel rounds of naked + hidden singles (no search).

    Each round fills, simultaneously, every cell that is forced given the board at the start of the
    round. Returns (rounds used, solved?). `rounds` is a crude serial-depth measure of a puzzle."""
    board = [int(v) for v in puzzle]
    for rnd in range(1, max_rounds + 1):
        cand = [0] * 81
        for i in range(81):
            if board[i] == 0:
                used = {board[j] for j in PEERS[i]}
                cand[i] = sum(1 << v for v in range(1, 10) if v not in used)
        forced = {}
        for i in range(81):
            if board[i] == 0:
                if cand[i] == 0:
                    return rnd, False
                if cand[i] & (cand[i] - 1) == 0:
                    forced[i] = cand[i].bit_length() - 1
        for u in UNITS:
            for v in range(1, 10):
                bit = 1 << v
                if any(board[i] == v for i in u):
                    continue
                spots = [i for i in u if board[i] == 0 and cand[i] & bit]
                if len(spots) == 1:
                    forced.setdefault(spots[0], v)
        if not forced:
            return rnd - 1, False
        for i, v in forced.items():
            board[i] = v
        if 0 not in board:
            return rnd, is_valid_grid(board)
    return max_rounds, False
