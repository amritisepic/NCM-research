"""Model builders for our matched baselines. Uses the vendored, unmodified upstream NCM modules.

- NCM tokenizer: upstream `VQVAE` (multiscale residual VQ, EMA codebooks, nested "fine drop" dropout).
  Our sizes are scaled to <= 10M parameters (codebooks included); see `tokenizer_kwargs`.
- NCM generator: upstream `NextConceptTransformer` (block-diagonal level mask, codebook-tied heads).
- AR baseline: decoder-only transformer built from the same upstream `Block` (RoPE, QK-norm, GELU MLP),
  same width/depth/heads as the NCM generator, tied input/output embedding.
- WithinLevelAR: NCM generator variant that is autoregressive *inside* each level (causal mask within the
  level + previous code embedding as input). Used only to estimate within-level total correlation.
"""
from __future__ import annotations

import copy
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ncmr import ncm_ref_loader  # noqa: F401  (sys.path for upstream modules)
from models.common import Block, LayerNorm, precompute_rope_cos_sin  # noqa: E402  upstream
from models.next_concept import NextConceptTransformer  # noqa: E402  upstream
from models.vqvae import VQVAE  # noqa: E402  upstream

NCM_TINYSTORIES_LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]


def n_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


# ----------------------------------------------------------------------------------------------- tokenizer
def tokenizer_kwargs(embed_dim=256, codebook=1024, levels=None, enc_layers=0, n_heads=4, dropout=0.1,
                     learned_l1=False, fine_drop=0.5, decay=0.95, in_vocab=8192, context=128) -> dict:
    levels = list(levels or NCM_TINYSTORIES_LEVELS)
    vq = OmegaConf.create({
        "clss": "multiscale_residual_vector_quantizer", "decay": decay, "epsilon": 1e-5,
        "commitment_cost": 0.25, "learned_l1_sampling": learned_l1, "learned_l1_cascade": learned_l1,
        "learned_l1_cascade_base": 4, "learned_all_sampling": False,
        "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
        "aux": {"fine_drop": {"prob": fine_drop, "min_keep": 1}},
        "levels": {"use_manual_levels": True, "manual_levels": levels},
        "learned_l1_cascade_norm": False, "learned_l1_cascade_norm_final": False,
        "ema_scale_by_world_size": False,
    })
    return dict(in_vocab_size=in_vocab, context_length=context, embed_dim=embed_dim,
                vqvae_vocab_size=[codebook] * len(levels), n_layers=n_heads, vector_quantizer_config=vq,
                compression_factor=1, dropout=dropout, pre_quant_groupnorm=8, use_qk_norm=True,
                pre_quant_dropout=0.0, enc_layers=enc_layers, dec_layers=1)


class TiedOutputVQVAE(VQVAE):
    """Upstream VQVAE with the decoder's output projection tied to the encoder token embedding."""

    def __init__(self, **kw):
        _stdout, sys.stdout = sys.stdout, open("/dev/null", "w")
        try:
            super().__init__(**kw)
        finally:
            sys.stdout.close()
            sys.stdout = _stdout
        # Upstream inits wte and wpe ~ N(0, 1); GroupNorm before quantization removes their common scale, so
        # scaling both to 0.02 keeps the encoder unchanged while giving the tied output layer sane logits.
        nn.init.normal_(self.encoder.wte.weight, std=0.02)
        if self.encoder.wpe is not None:
            nn.init.normal_(self.encoder.wpe.weight, std=0.02)
        self.decoder.output_layer.weight = self.encoder.wte.weight


def build_tokenizer(**kw) -> TiedOutputVQVAE:
    return TiedOutputVQVAE(**tokenizer_kwargs(**kw))


def load_text_tokenizer(path: str | None = None, device: str = "cpu"):
    """(vqvae, bpe) for analyses: our trained tokenizer from `path` (final.pt from train.py), or the released
    NCM TinyStories tokenizer when path is None/'released'."""
    import os
    from ncmr.ncm_ref_loader import load_ncm, safe_pickle_load
    from tokenization.bpe_tokenizer import BPETokenizer
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not path or path == "released":
        b = load_ncm(os.path.join(root, "artifacts", "tinystories"), device=device)
        return b.vqvae, b.pre_tokenizer
    st = torch.load(path, map_location="cpu", weights_only=False)  # our own file
    tok = build_tokenizer(**st["tok_kwargs"])
    tok.load_state_dict(st["model"])
    d = os.path.join(root, "data", "tinystories")
    bpe = BPETokenizer(safe_pickle_load(f"{d}/bpe_vocab_8192.pickle"), safe_pickle_load(f"{d}/bpe_merges_8192.pickle"),
                       {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})
    return tok.to(device).eval(), bpe


def tokenizer_loss(vqvae: VQVAE, x: torch.Tensor, vq_weight: float = 1.0):
    """Reconstruction CE over all 128 positions + upstream vq loss (commitment + cumulative-reconstruction
    MSE). Same computation as upstream VQVAE.forward, except that quantization runs with autocast disabled:
    under bf16/fp16 autocast the nearest-code distances would be computed in low precision (bf16 resolution
    is ~2-4 at these magnitudes), making code assignment noisy. Returns (loss, logs)."""
    z = vqvae._encode_pre_quant(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        z_q, vq_loss, util, gutil, _ = vqvae.vq(z.float())
    logits = vqvae.decoder(z_q)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1))
    with torch.no_grad():
        acc = (logits.argmax(-1) == x).float().mean()
    return ce + vq_weight * vq_loss, {"ce": ce.item(), "vq": vq_loss.detach().item(), "acc": acc.item(), "util": gutil}


# ----------------------------------------------------------------------------------------------- generators
def generator_config(vqvae: VQVAE, n_layer=6, n_head=8, n_embd=512, dropout=0.0, prefix_len=1,
                     query_level0=False, rope_base=100000):
    n_heads_out = len(vqvae.vq.levels) - (0 if query_level0 else 1)
    return OmegaConf.create({
        "n_layer": n_layer, "n_head": n_head, "bias": True, "dropout": dropout, "n_embd": n_embd,
        "context_length": prefix_len + sum(vqvae.vq.levels), "prefix_len": prefix_len,
        "raw_token_prefix": True, "prefix_pad_token_id": 1,
        "vocab_size": list(vqvae.vq.vqvae_vocab_sizes[(0 if query_level0 else 1):]),
        "predict_level0": False, "attn_pattern": "block_diagonal", "use_positional_encoding": True,
        "use_level_encoding": True, "use_rope": True, "rope_base": rope_base, "use_qk_norm": True,
        "ctx_encoder": {"enabled": False}, "use_flex_attention": False, "query_level0": query_level0,
        "tie_heads_to_codebook": True, "tie_heads_learn_scale": True,
    })


def build_ncm_generator(vqvae: VQVAE, **kw) -> NextConceptTransformer:
    _stdout, sys.stdout = sys.stdout, open("/dev/null", "w")
    try:
        net = NextConceptTransformer(vqvae_local=vqvae, config=generator_config(vqvae, **kw))
    finally:
        sys.stdout.close()
        sys.stdout = _stdout
    return net


@torch.no_grad()
def ncm_batch(vqvae: VQVAE, x: torch.Tensor, prefix: torch.Tensor):
    """Teacher-forcing inputs/targets for the NCM generator (as upstream VQVAE.get_ncm_inputs), computed in
    fp32 even inside an autocast region so the codes match the tokenizer's fp32 encoding exactly."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        z_in, targets = vqvae.get_ncm_inputs(x)        # (B, sum(l_1..l_K), D), codes of levels 1..K-1
        z_prefix = vqvae.encoder.wte(prefix)          # raw-token prefix, as at generation time
    return torch.cat([z_prefix, z_in], dim=1), targets


def ncm_loss(net: NextConceptTransformer, z_all: torch.Tensor, targets: torch.Tensor, per_level=False):
    logits = net(z_all)
    nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    nll = nll.view(targets.shape)
    if not per_level:
        return nll.mean()
    return nll.mean(), nll


class ARConfig:
    def __init__(self, n_layer, n_head, n_embd, dropout=0.0, bias=True, use_qk_norm=True):
        self.n_layer, self.n_head, self.n_embd = n_layer, n_head, n_embd
        self.dropout, self.bias, self.use_qk_norm = dropout, bias, use_qk_norm


class ARTransformer(nn.Module):
    """Decoder-only LM with the same upstream Block as the NCM generator (RoPE, QK-norm)."""

    def __init__(self, vocab=8192, n_layer=6, n_head=8, n_embd=512, dropout=0.0, max_len=129, rope_base=100000):
        super().__init__()
        cfg = ARConfig(n_layer, n_head, n_embd, dropout)
        self.wte = nn.Embedding(vocab, n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, bias=True)
        cos, sin = precompute_rope_cos_sin(torch.arange(max_len), n_embd // n_head, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        nn.init.normal_(self.wte.weight, std=0.02)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, idx, past=None):
        T = idx.shape[1]
        x = self.wte(idx)
        rope = (self.rope_cos[:T], self.rope_sin[:T])
        for b in self.blocks:
            x = b(x, attn_mask=None, rope_cos_sin=rope)  # upstream SDPA: causal when no mask is given
        x = self.ln_f(x)
        return x @ self.wte.weight.t()

    def loss(self, x):
        """x: (B, 1+L) with x[:, 0] = EOS prefix. Next-token CE over the L positions (PAD included,
        matching the tokenizer target so both models model the same 128-slot sequence)."""
        logits = self(x[:, :-1])
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, B, L=128, temperature=1.0, eos=0, device="cpu", generator=None):
        """Plain ancestral sampling, no KV cache (NFE = L either way)."""
        x = torch.full((B, 1), eos, dtype=torch.long, device=device)
        for _ in range(L):
            logits = self(x)[:, -1].float() / temperature
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            x = torch.cat([x, nxt], 1)
        return x[:, 1:]


# ------------------------------------------------------------------------------- TC probe (two-stream)
class TwoStreamLevelModel(nn.Module):
    """Controlled pair of models for estimating within-level conditional total correlation (TC).

    For every level k the sequence holds l_k CONTEXT tokens (input = coarse cumulative latent, exactly NCM's
    input) and l_k CODE tokens (input = the same latent + embedding of the previous same-level code, BOS at
    i=0). Context tokens attend to the prefix and to all context tokens of their level (as in NCM).
    Code token i attends to the prefix, all context tokens of its level, and
      mode="ar":   code tokens j <= i of its level   -> models p(c_k,i | c_k,<i, coarser levels)
      mode="fact": only itself (and gets no previous-code input) -> models p(c_k,i | coarser levels)
    Everything else (weights, depth, context) is identical, so
      TC_k (nats/sample) ~= NLL_fact,k - NLL_ar,k   (both summed over the l_k positions).
    A plain causal NCM would also hide the coarse context of later positions and understate TC.
    """

    def __init__(self, ncm_net: NextConceptTransformer, mode: str):
        super().__init__()
        assert mode in ("fact", "ar")
        self.net, self.mode = ncm_net, mode
        vq = ncm_net.vqvae_local[0].vq
        D = vq.embed_dim
        first = ncm_net._lvl_first
        self.lv = [int(l) for l in vq.levels[first:]]
        self.prev_emb = nn.ModuleList([nn.Embedding(vq.vqvae_vocab_sizes[k], D) for k in range(first, len(vq.levels))])
        self.bos = nn.Parameter(torch.zeros(len(self.lv), D))
        self.stream_emb = nn.Parameter(torch.zeros(2, ncm_net.config.n_embd))
        P = ncm_net.prefix_len
        lvl = torch.cat([torch.full((l,), k) for k, l in enumerate(self.lv)])
        pos = torch.cat([torch.arange(l) for l in self.lv])
        T = len(lvl)
        same = lvl[:, None] == lvl[None, :]
        allow = torch.zeros(P + 2 * T, P + 2 * T, dtype=torch.bool)
        allow[:, :P] = True                                   # everyone sees the prefix
        allow[:P, P:] = False
        allow[P:P + T, P:P + T] = same                        # context <-> context within level
        allow[P + T:, P:P + T] = same                         # code -> context within level
        if mode == "ar":
            allow[P + T:, P + T:] = same & (pos[None, :] <= pos[:, None])
        else:
            allow[P + T:, P + T:] = torch.eye(T, dtype=torch.bool)
        self.register_buffer("mask", torch.where(allow, 0.0, float("-inf"))[None, None])
        rp = torch.cat([torch.arange(P), pos, pos])
        cos, sin = precompute_rope_cos_sin(rp, ncm_net.config.n_embd // ncm_net.config.n_head, ncm_net.rope_base)
        self.register_buffer("rcos", cos, persistent=False)
        self.register_buffer("rsin", sin, persistent=False)

    def forward(self, z_all, targets):
        net = self.net
        P = net.prefix_len
        z_prefix, ctx = z_all[:, :P], z_all[:, P:]
        code_in = ctx
        if self.mode == "ar":
            shifted, acc = [], 0
            for j, lk in enumerate(self.lv):
                c = targets[:, acc:acc + lk]
                prev = self.prev_emb[j](c[:, :-1])
                shifted.append(torch.cat([self.bos[j].expand(c.shape[0], 1, -1), prev], 1))
                acc += lk
            code_in = ctx + torch.cat(shifted, 1)
        le = net.transformer.wle(net.lvl_1L)
        x = torch.cat([net.transformer.wte(z_prefix),
                       net.transformer.wte(ctx) + le + self.stream_emb[0],
                       net.transformer.wte(code_in) + le + self.stream_emb[1]], 1)
        for block in net.transformer.blocks:
            x = block(x, attn_mask=self.mask, rope_cos_sin=(self.rcos, self.rsin))
        T = sum(self.lv)
        x = net.transformer.ln_f(x)[:, P + T:]
        out, acc = [], 0
        for i, lk in enumerate(self.lv):
            out.append(net._pad_vocab(net.lm_heads[i](x[:, acc:acc + lk])))
            acc += lk
        return torch.cat(out, 1)
