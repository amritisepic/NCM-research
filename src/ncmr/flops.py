"""Analytic FLOP counts (multiply-add = 2 FLOPs). Training step = 3x forward (forward + 2x backward).

Transformer block with width d, MLP 4d: 24 d^2 FLOPs per position (QKV 6d^2, out-proj 2d^2, MLP 16d^2),
plus attention 4 d * (number of attended keys) per position.
"""
from __future__ import annotations


def block_fwd(d: int, n_layer: int, positions: int, keys_per_pos: float) -> float:
    return n_layer * positions * (24 * d * d + 4 * d * keys_per_pos)


def ar_fwd_per_seq(d, n_layer, T=128, vocab=8192) -> float:
    return block_fwd(d, n_layer, T, (T + 1) / 2) + 2 * T * d * vocab


def ncm_gen_fwd_per_seq(d, n_layer, levels, prefix=1, codebook=1024, vq_dim=256, query_level0=False) -> float:
    lv = levels if query_level0 else levels[1:]
    f = n_layer * prefix * (24 * d * d + 4 * d * prefix)
    for lk in lv:
        f += block_fwd(d, n_layer, lk, prefix + lk)
        f += lk * (2 * d * vq_dim + 2 * vq_dim * codebook)  # tied head: projection + codebook distances
    f += sum(lv) * 2 * vq_dim * d  # input projection
    return f


def ncm_gen_sample_fwd(d, n_layer, levels, prefix=1, codebook=1024, vq_dim=256, query_level0=False) -> float:
    """Eager sampling: each level re-processes the prefix (1 token for unconditional text)."""
    lv = levels if query_level0 else levels[1:]
    f = 0.0
    for lk in lv:
        f += block_fwd(d, n_layer, prefix + lk, prefix + lk)
        f += lk * (2 * d * vq_dim + 2 * vq_dim * codebook) + lk * 2 * vq_dim * d
    return f


def tokenizer_fwd_per_seq(d, levels, codebook=1024, T=128, vocab=8192, enc_layers=0, dec_layers=1) -> float:
    f = 2 * sum(levels) * d * codebook                         # VQ distances
    f += len(levels) * T * 2 * 3 * d * d                        # Phi convs (k=3) at full resolution
    f += block_fwd(d, enc_layers + dec_layers, T, T) + 2 * T * d * vocab
    return f
