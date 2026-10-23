"""TinyStories data as released by HazyResearch/ncm (custom 8192-entry BPE), and our fixed splits.

Source files (hf: hazyresearch/ncm-datasets, folder tinystories/):
  train_bpe_8192.bin + train_bpe_8192_offsets.npy : uint16 token stream of TinyStoriesV2-GPT4 train,
      stories separated by EOS (id 0); offsets[i] = start of story i.
  valid_bpe_8192.pickle : list of 27,544 validation stories (lists of ids, each ending with EOS).

Example construction: a 128-token window that starts at a story start and, if the story is shorter,
continues through its EOS into the following story(ies) ("packed" story openings). Evidence that NCM
does this: the released tokenizer reconstructs EOS 100% but PAD 0%, emitting "Once upon" after an EOS
(results/phase0_tokenizer_analysis.json, R002). Samples and references are cut at the first EOS.
`to_example(..., packed=False)` gives the older PAD-padded form (used only in R002).

FIXED SPLITS (Phase 0; never changed afterwards):
  train = the full NCM train stream (we never evaluate on it).
  val   = validation stories with even index (13,772 stories)  -> all tuning, MAUVE refs during dev.
  test  = validation stories with odd index  (13,772 stories)  -> final evaluation only.
"""
from __future__ import annotations

import os

import numpy as np

EOS, PAD = 0, 1
L = 128
VOCAB = 8192

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data", "tinystories")


def _safe_load_pickle(path):
    from ncmr.ncm_ref_loader import safe_pickle_load
    return safe_pickle_load(path)


def valid_stories() -> list[list[int]]:
    return _safe_load_pickle(os.path.join(DATA_DIR, "valid_bpe_8192.pickle"))


def split_stories(split: str) -> list[list[int]]:
    rows = valid_stories()
    if split == "val":
        return rows[0::2]
    if split == "test":
        return rows[1::2]
    raise ValueError(split)


def to_example(story: list[int], length: int = L) -> np.ndarray:
    ids = list(story[:length])
    ids += [PAD] * (length - len(ids))
    return np.asarray(ids, dtype=np.int64)


def examples(split: str, n: int | None = None, length: int = L, packed: bool = True) -> np.ndarray:
    rows = split_stories(split)
    n = len(rows) if n is None else n
    if not packed:
        return np.stack([to_example(r, length) for r in rows[:n]])
    out = []
    for i in range(n):
        ids, j = [], i
        while len(ids) < length and j < len(rows):
            ids += list(rows[j])  # every story already ends with EOS
            j += 1
        out.append(to_example(ids, length))
    return np.stack(out)


def cut_at_eos(ids) -> list[int]:
    ids = [int(t) for t in ids]
    return ids[:ids.index(EOS)] if EOS in ids else ids


class TrainStream:
    """Random story openings from the training stream (memory-mapped)."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.tokens = np.memmap(os.path.join(data_dir, "train_bpe_8192.bin"), dtype=np.uint16, mode="r")
        self.offsets = np.load(os.path.join(data_dir, "train_bpe_8192_offsets.npy"), allow_pickle=False)
        self.n_stories = len(self.offsets)

    def story(self, i: int, max_len: int = L) -> np.ndarray:
        start = int(self.offsets[i])
        end = int(self.offsets[i + 1]) if i + 1 < self.n_stories else len(self.tokens)
        return np.asarray(self.tokens[start:min(end, start + max_len)], dtype=np.int64)

    def window(self, i: int, length: int = L) -> np.ndarray:
        """Packed story opening: `length` tokens of the stream starting at story i's first token."""
        start = int(self.offsets[i])
        return to_example(np.asarray(self.tokens[start:start + length], dtype=np.int64).tolist(), length)

    def batch(self, rng: np.random.Generator, bs: int, length: int = L) -> np.ndarray:
        idx = rng.integers(0, self.n_stories, size=bs)
        return np.stack([self.window(int(i), length) for i in idx])
