"""Audit 9: cached/ancestral functions against independent upstream calls.

The full-width WMT test uses random weights with the released generator settings;
it is not a claim to have validated the unavailable released WMT weights.
"""
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ncmr import cached_decode as CD
from ncmr.jacobi_train import parallel_logits
from ncmr.ncm_ref_loader import NextConceptTransformer, VQVAE
from test_d1_bench import tiny_ts_like, tiny_wmt_like, WMT_LEVELS


@pytest.mark.parametrize("domain", ["tinystories", "wmt14"])
def test_ancestral_gumbels_match_full_teacher_calls_on_sampled_histories(domain):
    if domain == "wmt14":
        tok, net, x = tiny_wmt_like(seed=19, P=96)
        pre, vocab = x[1:2], 16
    else:
        tok, net = tiny_ts_like(seed=19)
        pre, vocab = torch.zeros(1, 1, dtype=torch.long), 32
    levels = list(map(int, tok.vq.levels))
    g = torch.Generator().manual_seed(41)
    canon = torch.randint(vocab, (1, sum(levels)), generator=g)
    saved = canon.clone()
    cache = CD.PrefixCache(net, *CD.prefix_inputs(net, pre))
    old_kv = [(k.clone(), v.clone()) for k, v in cache.kv]
    n, cut = 7, 8
    actual_gen = torch.Generator().manual_seed(907)
    got = CD.sample_from_cut(net, cache, canon, cut, n, actual_gen)
    # The oracle never uses canonical_state, _level_call, cache.call, or expand.
    expected = canon.expand(n, -1).clone()
    oracle_gen = torch.Generator().manual_seed(907)
    z, pm = CD.prefix_inputs(net, pre.expand(n, -1))
    for j in range(cut, len(levels)):
        with torch.no_grad():
            full = parallel_logits(net, z, expected, pm).float()
        start = sum(levels[(0 if net.predict_level0 or net.query_level0 else 1):j])
        logits = full[:, start:start + levels[j], :vocab]
        u = torch.rand(logits.shape, generator=oracle_gen).clamp_min(1e-20)
        draw = (logits - torch.log(-torch.log(u))).argmax(-1)
        off = sum(levels[:j])
        expected[:, off:off + levels[j]] = draw
    assert torch.equal(got, expected)
    assert torch.equal(actual_gen.get_state(), oracle_gen.get_state())
    assert torch.equal(canon, saved)
    assert got[:, sum(levels[:cut]):].unique(dim=0).shape[0] > 1
    for (k, v), (ok, ov) in zip(cache.kv, old_kv):
        assert torch.equal(k, ok) and torch.equal(v, ov)
    assert all(not b.attn.caching and b.attn.cached_k is None and b.attn.cached_v is None
               for b in net.transformer.blocks)


@pytest.mark.parametrize("stage", ["prefix", "target"])
def test_attention_cache_state_is_cleared_after_an_exception(monkeypatch, stage):
    _, net, pre = tiny_wmt_like(P=96)
    z, pm = CD.prefix_inputs(net, pre[:1])
    cache = CD.PrefixCache(net, z, pm) if stage == "target" else None

    def fail(*args, **kwargs):
        raise RuntimeError("injected audit failure")

    monkeypatch.setattr(net.transformer.blocks[1], "forward", fail)
    with pytest.raises(RuntimeError, match="injected audit failure"):
        if stage == "prefix":
            CD.PrefixCache(net, z, pm)
        else:
            codes = torch.zeros(1, sum(WMT_LEVELS), dtype=torch.long)
            CD.cached_parallel_logits(net, cache, codes)
    assert all(not b.attn.caching and b.attn.cached_k is None and b.attn.cached_v is None
               for b in net.transformer.blocks)


def test_released_width_depth_wmt_generator_and_padded_96_token_prefix():
    """Full 12 x d768 / 12 heads / 4096 vocab generator; real WMT VQ layout.

    Tokenizer encoder/decoder stacks are irrelevant to raw-prefix generation and
    omitted; the released-shape raw token embedding and multiscale VQ are used.
    Weights are random, with no network or checkpoint download.
    """
    torch.manual_seed(901)
    tok = VQVAE(
        in_vocab_size=40, context_length=64, embed_dim=384, vqvae_vocab_size=[4096] * 11,
        n_layers=6, dec_layers=1, compression_factor=1, dropout=0.0, pre_quant_groupnorm=8,
        use_qk_norm=True, vector_quantizer_config=OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer", "decay": 0.99, "commitment_cost": 0.25,
            "learned_l1_sampling": True, "learned_l1_cascade": True, "learned_l1_cascade_base": 4,
            "learned_l1_cascade_norm": True, "learned_l1_cascade_norm_final": True,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": WMT_LEVELS},
        })).eval()
    tok.encoder.wte = nn.Embedding(151680, 384)
    tok.decoder = nn.Identity()
    cfg = OmegaConf.create({
        "n_layer": 12, "n_head": 12, "bias": True, "dropout": 0.1, "n_embd": 768,
        "context_length": 160, "prefix_len": 96, "raw_token_prefix": True,
        "prefix_pad_token_id": 151643, "vocab_size": [4096] * 10, "predict_level0": True,
        "attn_pattern": "block_diagonal", "use_positional_encoding": True,
        "use_level_encoding": True, "use_rope": True, "rope_base": 100000,
        "use_qk_norm": True, "ctx_encoder": {"enabled": False},
        "use_flex_attention": False, "untied_head_bias": False,
    })
    net = NextConceptTransformer(vqvae_local=tok, config=cfg).eval()
    for p in list(net.parameters()) + list(tok.parameters()):
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(902)
    pre = torch.full((1, 96), 151643, dtype=torch.long)
    pre[:, :13] = torch.randint(1, 1000, (1, 13), generator=g)
    codes = torch.randint(4096, (1, sum(WMT_LEVELS)), generator=g)
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        eager = CD.seq_greedy(net, pre, cached=False)
        cached = CD.seq_greedy(net, pre, cached=True)
        assert torch.equal(cached, eager)
        z, pm = CD.prefix_inputs(net, pre)
        cache = CD.PrefixCache(net, z, pm)
        reference = parallel_logits(net, z, codes, pm)
        actual = CD.cached_parallel_logits(net, cache, codes)
        assert torch.allclose(actual, reference, atol=1e-4, rtol=1e-5), float((actual - reference).abs().max())
        # Inspect all query rows' prefix keys, level-0 read-off and per-level RoPE directly.
        expected_bias = torch.where(pm[:, None, None, :], -torch.inf, 0.0)
        assert torch.equal(cache.pbias, expected_bias)
        assert torch.equal(net._level0_slice(cache.h_prefix, pm), cache.h_prefix[:, 12:13])
        off = 96
        for j, length in enumerate(WMT_LEVELS[1:]):
            assert torch.equal(net.rope_cos[:96], getattr(net, f"rope_cos_gen_{j}")[:96])
            assert torch.equal(net.rope_cos[off:off + length], getattr(net, f"rope_cos_gen_{j}")[96:])
            off += length
        print("WMT released-shape parallel max abs error", float((actual - reference).abs().max()))


def test_sudoku_ancestral_rows_match_upstream_logits_with_hash_pinned_assets(released_sudoku):
    bundle, pre = released_sudoku
    tok, net = bundle.vqvae, bundle.net
    levels = list(map(int, tok.vq.levels))
    g = torch.Generator().manual_seed(17)
    canon = torch.randint(16, (1, sum(levels)), generator=g)
    n, cut = 3, 10
    cache = CD.PrefixCache(net, *CD.prefix_inputs(net, pre[:1]))
    got = CD.sample_from_cut(net, cache, canon, cut, n, torch.Generator().manual_seed(29))
    expected = canon.expand(n, -1).clone()
    gen = torch.Generator().manual_seed(29)
    z, pm = CD.prefix_inputs(net, pre[:1].expand(n, -1))
    for j in range(cut, len(levels)):
        with torch.no_grad():
            full = parallel_logits(net, z, expected, pm).float()
        off = sum(levels[:j])
        lg = full[:, off:off + levels[j]]
        u = torch.rand(lg.shape, generator=gen).clamp_min(1e-20)
        expected[:, off:off + levels[j]] = (lg - torch.log(-torch.log(u))).argmax(-1)
    assert torch.equal(got, expected)
