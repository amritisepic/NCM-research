"""Parallel (Jacobi) passes over all NCM levels, for decoding and for training on the model's own trajectory.

One parallel pass = the training-style block-diagonal forward: every level k is predicted from the cumulative
reconstruction of the *current* codes of levels < k. Iterating it from all-zero codes reaches the sequential
greedy output after at most K passes (level k is exact after k + 1 passes); it stops earlier when no code
changes (exact stop, output identical to sequential decoding) or when a task check passes (Sudoku verifier).

Used by scripts/train_sudoku_L.py (board-feedback repair passes, B6/C5) and scripts/train_wmt14_L.py
(self-consistency fine-tune, B2-trained). Works for models with a level-0 query (Sudoku) and with level 0
predicted from the prefix (WMT14). NFE counts the pass that detects the stop, as scripts/jacobi_levels.py does.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def parallel_logits(net, z_prefix: torch.Tensor, codes: torch.Tensor, pad_mask: torch.Tensor | None = None):
    """Logits (B, S, V) for every code position given the current codes (B, S) of all levels."""
    vq = net.vqvae_local[0].vq
    with torch.no_grad(), torch.autocast(device_type=codes.device.type, enabled=False):
        z_in = vq.idx_to_ncm_input(codes)
    q0 = [net.build_l0_query_input(codes.shape[0], z_in.dtype)] if net.query_level0 else []
    return net(torch.cat([z_prefix, *q0, z_in], 1), prefix_pad_mask=pad_mask)


@torch.no_grad()
def jacobi_decode(net, prefix_fn, n: int, S: int, device, max_passes: int, pad_mask=None, check_fn=None,
                  keep_history: bool = False, full_budget: bool = False, initial_fixed_ok: bool = False):
    """Greedy Jacobi decoding with per-sample stopping.

    prefix_fn(codes, has_prev) -> z_prefix: lets the prefix depend on the previous pass (board feedback);
    has_prev is False on pass 1. check_fn(codes) -> bool tensor (n,): optional task check (stop when True).
    Returns dict(codes=(n, S) final codes, nfe=(n,) passes used, stop=(n,) 0 = cap, 1 = fixed point, 2 = check,
    history=list of (n, S) codes per pass if keep_history). full_budget: always run max_passes passes (stopped
    samples keep their codes), so the executed compute is fixed (training teacher, codex audit2 A2-7).
    initial_fixed_ok: a static prefix_fn may detect convergence on pass 1 (WMT14; codex audit 4, A4-4); the
    default keeps pass-1 detection off for prefixes that change after the first pass (board feedback)."""
    codes = torch.zeros(n, S, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    nfe = torch.full((n,), max_passes, dtype=torch.long, device=device)
    stop = torch.zeros(n, dtype=torch.long, device=device)
    hist = []
    for t in range(1, max_passes + 1):
        logits = parallel_logits(net, prefix_fn(codes, t > 1), codes, pad_mask).float()
        new = logits.argmax(-1)
        new = torch.where(done[:, None], codes, new)
        fixed = (new == codes).all(1) & ~done & (initial_fixed_ok or t > 1)
        chk = check_fn(new) & ~done & ~fixed if check_fn is not None else torch.zeros_like(done)
        for m, code in ((fixed, 1), (chk, 2)):
            nfe[m], stop[m] = t, code
            done |= m
        codes = new
        if keep_history:
            hist.append(codes.clone())
        if bool(done.all()) and not full_budget:
            break
    return {"codes": codes, "nfe": nfe, "stop": stop, "history": hist}


class PrefixFeedback(nn.Module):
    """Adds an embedding of the current output tokens to prefix positions (board feedback for repair passes).
    Zero-initialized: with feedback off or at step 0 the network computes exactly the released model."""

    def __init__(self, vocab: int, dim: int, offset: int, length: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        nn.init.zeros_(self.emb.weight)
        self.offset, self.length = offset, length

    def forward(self, z_prefix: torch.Tensor, guess: torch.Tensor | None, active: torch.Tensor | None = None):
        """guess: (B, length) token ids decoded from the previous pass; active: (B,) bool, False = no guess."""
        if guess is None:
            return z_prefix
        add = self.emb(guess).to(z_prefix.dtype)
        if active is not None:
            add = add * active[:, None, None].to(add.dtype)
        out = z_prefix.clone()
        out[:, self.offset:self.offset + self.length] = out[:, self.offset:self.offset + self.length] + add
        return out
