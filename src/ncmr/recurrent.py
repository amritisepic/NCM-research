"""C4 — Recurrent NCM: carry the generator's state across levels and train with backprop through levels.

Upstream NCM runs each level as an independent forward pass whose only input from earlier levels is the
committed latent z_hat; the transformer's hidden state (its "belief" before sampling) is discarded. Here
level k additionally receives W_carry * upsample(h_{k-1}), the final (ln_f-normalized) hidden states of
level k-1 linearly interpolated to l_k positions, with W_carry zero-initialized (so training starts exactly
at NCM). Training unrolls the levels sequentially with full backprop through the carries (TRM's key
ingredient); inference is identical in NFE (one pass per level). The prompt/prefix attends only to itself,
so its per-layer keys/values are computed once and reused by every level (as upstream's KV cache).

Same weights/heads as the upstream NextConceptTransformer it wraps (+ one d x d projection).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncmr import ncm_ref_loader  # noqa: F401
from models.common import apply_rotary_emb  # noqa: E402  upstream
from models.next_concept import NextConceptTransformer  # noqa: E402


def _qkv(attn, x, cos, sin):
    B, T, C = x.shape
    q, k, v = attn.c_attn(x).split([attn.n_embd, attn._kv_dim, attn._kv_dim], dim=2)
    hd = C // attn.n_head
    q = q.view(B, T, attn.n_head, hd).transpose(1, 2)
    k = k.view(B, T, attn.n_kv_head, hd).transpose(1, 2)
    v = v.view(B, T, attn.n_kv_head, hd).transpose(1, 2)
    if attn.n_kv_head != attn.n_head:
        rep = attn.n_head // attn.n_kv_head
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    if attn.use_qk_norm:
        q, k = attn.q_norm(q), attn.k_norm(k)
    return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin), v


def _block(block, x, cos, sin, pk=None, pv=None, attn_mask=None):
    """Upstream Block forward (pre-LN, attention + MLP) with optional extra prefix keys/values.
    Returns (output, k, v) where k, v are this input's own (rotated) keys/values."""
    a = block.attn
    h = block.ln_1(x)
    q, k, v = _qkv(a, h, cos, sin)
    K = k if pk is None else torch.cat([pk, k], 2)
    V = v if pv is None else torch.cat([pv, v], 2)
    y = F.scaled_dot_product_attention(q, K, V, attn_mask=attn_mask,
                                       dropout_p=a.dropout if a.training else 0.0)
    B, T, C = x.shape
    y = a.resid_dropout(a.c_proj(y.transpose(1, 2).reshape(B, T, C)))
    x = x + y
    x = x + block.mlp(block.ln_2(x))
    return x, k, v


class RecurrentNCM(nn.Module):
    def __init__(self, net: NextConceptTransformer, carry: bool = True):
        super().__init__()
        self.net, self.carry = net, carry
        d = net.config.n_embd
        self.carry_proj = nn.Linear(d, d)
        nn.init.zeros_(self.carry_proj.weight)
        nn.init.zeros_(self.carry_proj.bias)
        vq = net.vqvae_local[0].vq
        self.levels = [int(l) for l in vq.levels]
        self.first = net._lvl_first
        self.P = net.prefix_len

    def _rope(self, k_target):
        """(cos, sin) for prefix positions and for target level k_target (index into heads)."""
        cos = getattr(self.net, f"rope_cos_gen_{k_target}")
        sin = getattr(self.net, f"rope_sin_gen_{k_target}")
        return (cos[:self.P], sin[:self.P]), (cos[self.P:], sin[self.P:])

    def prefix_kv(self, prefix_emb):
        (pc, ps), _ = self._rope(0)
        h, kv = prefix_emb, []
        for blk in self.net.transformer.blocks:
            h, k, v = _block(blk, h, pc, ps)
            kv.append((k, v))
        return kv

    def level_forward(self, x_in, kv, j, carry_h):
        """x_in: (B, l_j, D_vq) coarse latent input for target level j (0-based over predicted levels)."""
        net = self.net
        acc = sum(self.levels[self.first:self.first + j])
        lj = self.levels[self.first + j]
        x = net.transformer.wte(x_in) + net.transformer.wle(net.lvl_1L[:, acc:acc + lj])
        if self.carry and carry_h is not None:
            up = F.interpolate(carry_h.transpose(1, 2), size=lj, mode="linear", align_corners=False).transpose(1, 2)
            x = x + self.carry_proj(up)
        _, (c, s) = self._rope(j)
        for blk, (pk, pv) in zip(net.transformer.blocks, kv):
            x, _, _ = _block(blk, x, c, s, pk, pv)
        h = net.transformer.ln_f(x)
        return net._pad_vocab(net.lm_heads[j](h)), h

    def forward(self, z_prefix, z_in):
        """Teacher-forced training forward.
        z_prefix: (B, P, D_vq) raw-token prefix embeddings; z_in: (B, sum(levels[1:]), D_vq) = upstream
        `idx_to_ncm_input` (inputs of levels 1..K-1). Returns logits for the predicted levels (level 0
        too when query_level0), concatenated along positions."""
        net = self.net
        kv = self.prefix_kv(net.transformer.wte(z_prefix))
        out, carry_h, acc = [], None, 0
        for j, lj in enumerate(self.levels[self.first:]):
            if net.query_level0 and j == 0:
                x_in = net.build_l0_query_input(z_prefix.shape[0], z_prefix.dtype)
            else:
                x_in = z_in[:, acc:acc + lj]
                acc += lj
            logits, carry_h = self.level_forward(x_in, kv, j, carry_h)
            out.append(logits)
        return torch.cat(out, 1)

    @torch.no_grad()
    def generate(self, prefix_tokens, temperature=0.0, l0_idx=None):
        """Mirrors upstream generate (eager) plus the carry. Returns all codes (B, sum(levels))."""
        net = self.net
        vq = net.vqvae_local[0].vq
        B = prefix_tokens.shape[0]
        kv = self.prefix_kv(net.transformer.wte(net.vqvae_local[0].encoder.wte(prefix_tokens)))
        z_hat = torch.zeros(B, self.levels[-1], vq.embed_dim, device=prefix_tokens.device)
        codes, carry_h = [], None

        def sample(logits):
            if temperature == 0:
                return logits.argmax(-1)
            return torch.multinomial(F.softmax(logits.float() / temperature, -1).flatten(0, 1), 1).view(logits.shape[:2])

        if net.query_level0:
            logits, carry_h = self.level_forward(net.build_l0_query_input(B), kv, 0, None)
            idx = sample(logits)
        else:
            idx = l0_idx
        codes.append(idx)
        for li in range(len(self.levels) - 1):
            z_hat, x_in = vq.get_next_autoregressive_input(li, z_hat, vq.embeddings[li](idx))
            j = li + 1 - self.first
            logits, carry_h = self.level_forward(x_in, kv, j, carry_h)
            idx = sample(logits)
            codes.append(idx)
        return torch.cat(codes, 1)
