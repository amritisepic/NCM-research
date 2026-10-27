"""Greedy decoders with the prefix computed once per item, the stop rules D1 times, and level-by-level sampling from a
cut (D1 and E1 in docs/preregistration.md).

Under NCM's masks the prefix (WMT14 source, Sudoku puzzle; one EOS token for TinyStories) attends only to itself, and
its rotary positions are 0..P-1 in every forward call, so its keys and values are the same in every call of a decode.
The upstream decoders recompute them: `NextConceptTransformer.generate`'s eager path runs prefix + level in each of its
calls, and every Jacobi pass (`jacobi_train.parallel_logits`) runs prefix + all levels. The `cached` variants here run
the prefix once (with its pad mask) and every later call on the target rows only, attending to the stored keys and
values. They compute the same function up to float rounding; tests/test_d1_bench.py checks logits and codes against
the upstream decoders. (Upstream's own `use_kv_cache=True` path drops the prefix pad mask, so it is not used.)

  seq_greedy(net, prefix_ids, cached, l0=None)  sequential greedy decode (cached=False: upstream `generate`)
  PrefixCache / cached_parallel_logits          one Jacobi pass over every level from the current codes
  jacobi_stop(...)                              batch-1 greedy Jacobi with the exact / stable-text / verifier stop
  stable_stop(texts, exact_passes)              the same stable-text rule on saved per-pass outputs (R042)
  sample_from_cut(...)                          ancestral sampling of levels >= k given levels < k (E1)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .jacobi_train import parallel_logits


def check_supported(net):
    """The released TinyStories, WMT14 and Sudoku generators: rotary positions, raw-token prefix, no context encoder,
    post-upsample convs, flex attention or autoregressive level 0."""
    c = net.config
    ok = (net.use_rope and not net.use_ctx_encoder and not net.use_post_upsample_conv and not net.use_flex_attention
          and not net.l0_ar and bool(getattr(c, "raw_token_prefix", False))
          and not bool(getattr(c, "contextual_prefix", False)))
    if not ok:
        raise NotImplementedError("cached decoding covers the released generators' configuration only")


def prefix_inputs(net, prefix_ids):
    """(z_prefix, pad_mask) as `generate` builds them: the tokenizer's token embedding of the raw prefix; pad mask
    where the prefix holds the model's pad id (None if the model has none)."""
    P = net.prefix_len
    z = net.vqvae_local[0].encoder.wte(prefix_ids)[:, :P, :]
    pm = (prefix_ids[:, :P] == net.prefix_pad_id) if net.prefix_pad_id is not None else None
    return z, pm


def _pad_bias(pm, dtype, device):
    """(B, 1, 1, P): -inf at padded prefix positions, as `generate` and `_prefix_pad_bias` build it."""
    if pm is None:
        return None
    return torch.where(pm[:, None, None, :], torch.tensor(float("-inf"), dtype=dtype, device=device),
                       torch.tensor(0.0, dtype=dtype, device=device))


class PrefixCache:
    """The prefix's per-block keys and values (rotated), computed once; `call` runs target rows against them."""

    @torch.no_grad()
    def __init__(self, net, z_prefix, pad_mask):
        check_supported(net)
        self.net, self.pm = net, pad_mask
        P = net.prefix_len
        self.P, self.B = P, z_prefix.shape[0]
        dev = z_prefix.device
        self.pbias = _pad_bias(pad_mask, torch.float32, dev)
        h = net.transformer.wte(z_prefix)
        mask = torch.zeros(1, 1, P, P, device=dev)
        if self.pbias is not None:
            mask = mask + self.pbias
        rope = (net.rope_cos[:P], net.rope_sin[:P])        # prefix positions 0..P-1, as in every rotary table
        net._set_kv_caching(True)
        try:
            for blk in net.transformer.blocks:
                h = blk(h, attn_mask=mask, rope_cos_sin=rope)
            self.kv = [(b.attn.cached_k, b.attn.cached_v) for b in net.transformer.blocks]
        finally:
            net._set_kv_caching(False)
        self.h_prefix = net.transformer.ln_f(h)              # for level 0 when it is read off the prefix (WMT14)

    def expand(self, n: int) -> "PrefixCache":
        """The same cache for n rows of one item (no copy)."""
        assert self.B == 1
        c = PrefixCache.__new__(PrefixCache)
        c.net, c.P, c.B = self.net, self.P, n
        c.pm = None if self.pm is None else self.pm.expand(n, -1)
        c.pbias = None if self.pbias is None else self.pbias.expand(n, -1, -1, -1)
        c.kv = [(k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)) for k, v in self.kv]
        c.h_prefix = self.h_prefix.expand(n, -1, -1)
        return c

    @torch.no_grad()
    def call(self, x, base_mask, rope):
        """Run target rows x (B, T, d) (already embedded) through the blocks against the cached prefix. base_mask:
        (1, 1, T, P + T) target-row mask without the pad bias; rope: the rotary table whose entries P..P+T-1 belong to
        these rows. Returns the hidden states after ln_f (B, T, d)."""
        net = self.net
        mask = base_mask if self.pbias is None else base_mask + F.pad(self.pbias, (0, x.shape[1]))
        net._set_kv_caching(True)
        try:
            for blk, (k, v) in zip(net.transformer.blocks, self.kv):
                blk.attn.cached_k, blk.attn.cached_v = k, v
                x = blk(x, attn_mask=mask, rope_cos_sin=rope)
        finally:
            net._set_kv_caching(False)
        return net.transformer.ln_f(x)


def level0_logits_from_prefix(net, cache: PrefixCache):
    """predict_level0 models (WMT14): level 0 is read off the last l_0 non-pad prefix positions."""
    return net._pad_vocab(net.lm_head_l0(net._level0_slice(cache.h_prefix, cache.pm)))


# ------------------------------------------------------------------------------------------------ Jacobi passes
@torch.no_grad()
def cached_parallel_logits(net, cache: PrefixCache, codes):
    """`jacobi_train.parallel_logits` with the prefix taken from `cache`: logits (B, S, V) for every code position
    given the current codes (B, S). Level-0 logits of a predict_level0 model come from the cached prefix."""
    vq = net.vqvae_local[0].vq
    P = net.prefix_len
    with torch.autocast(device_type=codes.device.type, enabled=False):
        z_in = vq.idx_to_ncm_input(codes)
    q0 = [net.build_l0_query_input(codes.shape[0], z_in.dtype)] if net.query_level0 else []
    x = net.transformer.wte(torch.cat([*q0, z_in], 1))
    if net.config.use_level_encoding:
        x = x + net.transformer.wle(net.lvl_1L)
    h = cache.call(x, net.attn_mask_for_masking[:, :, P:, :], (net.rope_cos, net.rope_sin))
    out = [level0_logits_from_prefix(net, cache)] if net.predict_level0 else []
    off = 0
    for i, lk in enumerate(net.levels[net._lvl_first:]):
        out.append(net._pad_vocab(net.lm_heads[i](h[:, off:off + lk])))
        off += lk
    return torch.cat(out, 1)


def make_pass(net, prefix_ids, cached: bool):
    """Returns (pass_fn, setup) for one item: pass_fn(codes) -> logits (B, S, V). With cached=True the prefix is
    computed once when make_pass is called (that cost belongs to the item's decode)."""
    z, pm = prefix_inputs(net, prefix_ids)
    if not cached:
        return lambda codes: parallel_logits(net, z, codes, pm)
    cache = PrefixCache(net, z, pm)
    return lambda codes: cached_parallel_logits(net, cache, codes)


# ------------------------------------------------------------------------------------------------ sequential
def _level_call(net, cache: PrefixCache, x_in, j: int):
    """Logits of level j (>= 1, or 0 for a query_level0 model's queries) from its inputs x_in (B, l_j, e), as
    `generate`'s eager call for that level computes them (`_decode_level_logits`), against the cached prefix."""
    hi = j - net._lvl_first                                # head and rotary-table index of level j
    acc = sum(net.levels[net._lvl_first:j])                # first target row of level j
    lj = net.levels[j]
    x = net.transformer.wte(x_in)
    if net.config.use_level_encoding:
        x = x + net.transformer.wle(net.lvl_1L[:, acc:acc + lj])
    rope = (getattr(net, f"rope_cos_gen_{hi}"), getattr(net, f"rope_sin_gen_{hi}"))
    base = torch.zeros(1, 1, lj, net.prefix_len + lj, device=x.device)
    return net.lm_heads[hi](cache.call(x, base, rope))


@torch.no_grad()
def seq_greedy(net, prefix_ids, cached: bool, l0=None):
    """Greedy sequential decode (B, S). cached=False is upstream `generate(temperature=0)` (eager: the prefix is
    re-run in each of its calls); cached=True runs the prefix once and each level's call on that level's rows. `l0`:
    given level-0 codes (TinyStories); otherwise level 0 comes from the prefix (WMT14) or the level-0 queries (Sudoku)."""
    if not cached:
        eos = getattr(net, "prefix_pad_id", None)
        return net.generate(l0, prefix_ids, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0,
                            cfg_null_eos=eos if eos is not None else 0)
    vq = net.vqvae_local[0].vq
    B, dev = prefix_ids.shape[0], prefix_ids.device
    z, pm = prefix_inputs(net, prefix_ids)
    cache = PrefixCache(net, z, pm)
    if l0 is not None:
        idx = l0
    elif net.predict_level0:
        idx = level0_logits_from_prefix(net, cache).argmax(-1)
    elif net.query_level0:
        idx = _level_call(net, cache, net.build_l0_query_input(B, z.dtype), 0).argmax(-1)
    else:
        raise ValueError("this generator needs the level-0 codes")
    out = [idx]
    z_hat = torch.zeros(B, net.levels[-1], vq.embed_dim, device=dev)
    for j in range(1, len(net.levels)):
        z_hat, x_in = vq.get_next_autoregressive_input(j - 1, z_hat, vq.embeddings[j - 1](out[-1]))
        out.append(_level_call(net, cache, x_in, j).argmax(-1))
    return torch.cat(out, 1)


# ------------------------------------------------------------------------------------------------ stop rules
def stable_stop(texts, exact_passes: int, m: int = 1):
    """R042's "stable x m" on saved per-pass outputs: texts[t - 1] = the decoded text after pass t (t = 1..). Stop at
    the first pass t > m whose text equals that of each of the previous m passes, else at the exact stop. Returns
    (passes, text)."""
    for t in range(m + 1, exact_passes + 1):
        if all(texts[t - 1] == texts[t - 1 - i] for i in range(1, m + 1)):
            return t, texts[t - 1]
    return exact_passes, texts[exact_passes - 1]


@torch.no_grad()
def jacobi_stop(pass_fn, S: int, device, max_passes: int, rule: str = "exact", decode=None, check=None,
                initial_fixed_ok: bool = False):
    """Greedy Jacobi decoding of ONE item from all-zero codes, as `jacobi_train.jacobi_decode` decodes it, stopping by:
      exact     the first pass whose output codes equal its input codes (pass 1 counts only if initial_fixed_ok:
                a static prefix, codex audit 4 A4-4);
      stable1   exact, or the first pass t >= 2 whose decoded text equals pass t-1's (R042's "stable x1");
      verifier  exact, or the first pass whose decoded board passes `check` (R020).
    decode(codes) -> text is the per-pass decode the stable rule reads (run inside any timed region); check(codes) ->
    bool. Returns (codes, passes, stop) with stop in {"fixed", "stable", "check", "cap"}."""
    assert rule in ("exact", "stable1", "verifier")
    codes = torch.zeros(1, S, dtype=torch.long, device=device)
    prev = None
    for t in range(1, max_passes + 1):
        new = pass_fn(codes).float().argmax(-1)
        if (initial_fixed_ok or t > 1) and bool(torch.equal(new, codes)):
            return new, t, "fixed"
        codes = new
        if rule == "stable1":
            text = decode(codes)
            if t >= 2 and text == prev:
                return codes, t, "stable"
            prev = text
        elif rule == "verifier" and check(codes):
            return codes, t, "check"
    return codes, max_passes, "cap"


# ------------------------------------------------------------------------------------------------ sampling (E1)
@torch.no_grad()
def canonical_state(net, canon, k: int):
    """For the canonical codes (1, S) of one item: the cumulative reconstruction z_hat after levels 0..k-1 and the
    input of level k (both (1, ...)), as `generate` builds them level by level."""
    vq = net.vqvae_local[0].vq
    lv = net.levels
    z_hat = torch.zeros(canon.shape[0], lv[-1], vq.embed_dim, device=canon.device)
    x_in, off = None, 0
    for j in range(k):
        z_hat, x_in = vq.get_next_autoregressive_input(j, z_hat, vq.embeddings[j](canon[:, off:off + lv[j]]))
        off += lv[j]
    return z_hat, x_in


@torch.no_grad()
def sample_from_cut(net, cache: PrefixCache, canon, k: int, n: int, gen, temperature: float = 1.0):
    """Ancestral sampling of levels k..K-1 (k >= 1) for n rows of ONE item given its canonical levels < k: level k
    from the model's distribution given c*_{<k}, then each finer level given the coarser sampled ones, at the given
    temperature (Gumbel-max with `gen`). `cache` is the item's PrefixCache (batch 1). Returns codes (n, S)."""
    vq = net.vqvae_local[0].vq
    lv = net.levels
    off_k = sum(lv[:k])
    z_hat, x_in = canonical_state(net, canon, k)
    rows = cache.expand(n)
    first = _level_call(net, cache, x_in, k).float()                      # (1, l_k, V): the same for every row
    parts = [canon[:, :off_k].expand(n, -1)]
    lg = first.expand(n, -1, -1)
    z_hat = z_hat.expand(n, -1, -1).clone()
    for j in range(k, len(lv)):
        if j > k:
            z_hat, x_in = vq.get_next_autoregressive_input(j - 1, z_hat, vq.embeddings[j - 1](parts[-1]))
            lg = _level_call(net, rows, x_in, j).float()
        u = torch.rand(lg.shape, device=lg.device, generator=gen).clamp_min(1e-20)
        parts.append((lg / temperature - torch.log(-torch.log(u))).argmax(-1))
    return torch.cat(parts, 1)
