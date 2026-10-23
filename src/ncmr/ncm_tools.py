"""Analysis helpers for the released NCM multiscale residual VQ (upstream models/quant.py)."""
from __future__ import annotations

import torch


@torch.no_grad()
def zhat_snapshots(vq, idx: torch.Tensor) -> list[torch.Tensor]:
    """Full-resolution cumulative latent z_hat after each level, as independent tensors.

    This is the input the decoder sees under nested dropout ("fine_drop": z_hat truncated after a
    random level, at full resolution), so decoding snapshot k is what stopping after level k yields.
    Same arithmetic as upstream `idx_to_z_q`; upstream `idx_to_zhats` cannot be used because it
    appends aliases of one in-place-updated tensor (HazyResearch/ncm issue #11).
    """
    L = vq.levels[-1]
    n = len(vq.levels)
    z_hat = torch.zeros(idx.shape[0], L, vq.embed_dim, device=idx.device)
    out, acc = [], 0
    for k, lk in enumerate(vq.levels):
        e = vq.embeddings[k](idx[:, acc:acc + lk])
        q = vq._up_to_full(e, k, L) if k != n - 1 else e
        q = vq._apply_quant_resi(q, k)
        z_hat = z_hat + q
        out.append(z_hat)
        acc += lk
    return out


def level_slices(levels: list[int]) -> list[slice]:
    out, acc = [], 0
    for lk in levels:
        out.append(slice(acc, acc + lk))
        acc += lk
    return out
