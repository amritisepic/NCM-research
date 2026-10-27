"""D1 (docs/preregistration.md): the prefix-cached decoders compute what the upstream decoders compute, the batch-1
stop rules decode as `jacobi_decode` and R042's saved-output rule do, and E1's sampler draws from the model's levels."""
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from ncmr import cached_decode as CD
from ncmr.jacobi_train import jacobi_decode, parallel_logits
from ncmr.ncm_ref_loader import VQVAE, NextConceptTransformer

ROOT = Path(__file__).resolve().parents[1]
WMT_LEVELS = [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]
PAD = 39


def tiny_wmt_like(seed=0, P=12):
    """A random generator with WMT14's structure: 11 levels up to 64, level 0 read off a padded source prefix
    (predict_level0), untied heads without bias, rotary positions, QK-norm."""
    torch.manual_seed(seed)
    tok = VQVAE(in_vocab_size=40, context_length=64, embed_dim=8, vqvae_vocab_size=[16] * len(WMT_LEVELS), n_layers=2,
                compression_factor=1, dropout=0, pre_quant_groupnorm=2, use_qk_norm=True,
                vector_quantizer_config=OmegaConf.create({
                    "clss": "multiscale_residual_vector_quantizer", "decay": 0.99, "commitment_cost": 0.25,
                    "learned_l1_sampling": True, "learned_l1_cascade": True, "learned_l1_cascade_base": 4,
                    "learned_l1_cascade_norm": True, "learned_l1_cascade_norm_final": True,
                    "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
                    "levels": {"use_manual_levels": True, "manual_levels": WMT_LEVELS}})).eval()
    cfg = OmegaConf.create({
        "n_layer": 2, "n_head": 2, "bias": True, "dropout": 0.0, "n_embd": 32, "context_length": P + 64,
        "prefix_len": P, "raw_token_prefix": True, "prefix_pad_token_id": PAD, "vocab_size": [16] * 10,
        "predict_level0": True, "attn_pattern": "block_diagonal", "use_positional_encoding": True,
        "use_level_encoding": True, "use_rope": True, "rope_base": 100000, "use_qk_norm": True,
        "ctx_encoder": {"enabled": False}, "use_flex_attention": False, "untied_head_bias": False})
    net = NextConceptTransformer(vqvae_local=tok, config=cfg).eval()
    g = torch.Generator().manual_seed(seed + 1)
    for p in net.parameters():              # spread the random weights so greedy choices are far from ties
        p.data.add_(0.05 * torch.randn(p.shape, generator=g))
    lens = [P, P - 3, 5, 1, P - 1, 7]
    x = torch.randint(2, 30, (len(lens), P), generator=g)
    for i, n in enumerate(lens):
        x[i, n:] = PAD
    return tok, net, x


def tiny_ts_like(seed=0):
    from ncmr.models import build_ncm_generator
    from test_audit_hierarchy import tiny_text_tokenizer
    tok = tiny_text_tokenizer(learned_l0=True, seed=seed)
    torch.manual_seed(seed)
    net = build_ncm_generator(tok, n_layer=2, n_head=2, n_embd=32, dropout=0.0).eval()
    return tok, net


# ------------------------------------------------------------------------------------------------ cached = upstream
def test_cached_sequential_equals_generate_wmt_like():
    tok, net, x = tiny_wmt_like()
    with torch.no_grad():
        eager = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=PAD)
    assert torch.equal(CD.seq_greedy(net, x, cached=True), eager)
    assert torch.equal(CD.seq_greedy(net, x, cached=False), eager)


def test_cached_sequential_honours_the_pad_mask():
    """Garbage under the pad positions must not change the decode (the upstream KV path ignores the mask)."""
    tok, net, x = tiny_wmt_like()
    ref = CD.seq_greedy(net, x, cached=True)
    lg = CD.cached_parallel_logits(net, CD.PrefixCache(net, *CD.prefix_inputs(net, x)), ref)
    with torch.no_grad():
        tok.encoder.wte.weight[PAD].add_(3.0)          # what sits under the pad positions changes
    try:
        out = CD.seq_greedy(net, x, cached=True)
        lg2 = CD.cached_parallel_logits(net, CD.PrefixCache(net, *CD.prefix_inputs(net, x)), ref)
    finally:
        with torch.no_grad():
            tok.encoder.wte.weight[PAD].sub_(3.0)
    assert torch.equal(out, ref)
    assert torch.allclose(lg, lg2, atol=1e-5)


def test_cached_pass_equals_parallel_logits_wmt_like():
    tok, net, x = tiny_wmt_like()
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 16, (x.shape[0], sum(WMT_LEVELS)), generator=g)
    z, pm = CD.prefix_inputs(net, x)
    with torch.no_grad():
        ref = parallel_logits(net, z, codes, pm)
    got = CD.cached_parallel_logits(net, CD.PrefixCache(net, z, pm), codes)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cached", [False, True])
def test_batch1_exact_stop_equals_jacobi_decode_wmt_like(cached):
    tok, net, x = tiny_wmt_like()
    S, K = sum(WMT_LEVELS), len(WMT_LEVELS)
    for i in range(x.shape[0]):
        xi = x[i:i + 1]
        z, pm = CD.prefix_inputs(net, xi)
        ref = jacobi_decode(net, lambda c, h: z, 1, S, "cpu", K + 1, pad_mask=pm, initial_fixed_ok=True)
        codes, passes, stop = CD.jacobi_stop(CD.make_pass(net, xi, cached), S, "cpu", K + 1, "exact",
                                             initial_fixed_ok=True)
        assert torch.equal(codes, ref["codes"]) and passes == int(ref["nfe"][0])
        assert stop == ("fixed" if int(ref["stop"][0]) == 1 else "cap")
        assert torch.equal(codes, CD.seq_greedy(net, xi, cached=True))      # the fixed point is the sequential output


def test_released_sudoku_cached_decoders(released_sudoku):
    from ncmr.sudoku import cells, is_solution
    bundle, x = released_sudoku
    net, tok = bundle.net, bundle.vqvae
    vq = tok.vq
    S, K = sum(vq.levels), len(vq.levels)
    with torch.no_grad():
        eager = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
    assert torch.equal(CD.seq_greedy(net, x, cached=True), eager)
    z, pm = CD.prefix_inputs(net, x)
    assert pm is None
    with torch.no_grad():
        ref = parallel_logits(net, z, eager, pm)
    assert torch.allclose(CD.cached_parallel_logits(net, CD.PrefixCache(net, z, pm), eager), ref, atol=1e-4)

    def decode(c):
        return tok.decoder(vq.idx_to_z_q(c)).argmax(-1)

    for i in range(6):
        xi = x[i:i + 1]
        zi = tok.encoder.wte(xi)
        chk = lambda c: torch.tensor([b is not None and is_solution(b, cells(xi[0].tolist()))
                                      for b in [cells(r) for r in decode(c).tolist()]])
        for rule, check_fn in (("exact", None), ("verifier", chk)):
            ref = jacobi_decode(net, lambda c, h: zi, 1, S, "cpu", K + 1, check_fn=check_fn)
            for cached in (False, True):
                codes, passes, stop = CD.jacobi_stop(CD.make_pass(net, xi, cached), S, "cpu", K + 1, rule,
                                                     check=(lambda c: bool(chk(c)[0])) if check_fn else None)
                assert torch.equal(codes, ref["codes"]) and passes == int(ref["nfe"][0])
                assert stop == {0: "cap", 1: "fixed", 2: "check"}[int(ref["stop"][0])]


def test_cached_sequential_with_given_level0_ts_like():
    tok, net = tiny_ts_like()
    g = torch.Generator().manual_seed(5)
    l0 = torch.randint(0, 32, (5, 1), generator=g)
    pre = torch.zeros(5, 1, dtype=torch.long)
    with torch.no_grad():
        eager = net.generate(l0, pre, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=0)
    assert torch.equal(CD.seq_greedy(net, pre, cached=True, l0=l0), eager)


# ------------------------------------------------------------------------------------------------ stop rules
def _hyp_val(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


ARMS = {"J0": "results/colab/w2c_L_wmt_ctrl/L_wmt_released__hyp_val.json",
        "J1": "results/colab/w2c_L_wmt_ctrl/L_wmt_ctrl_s0__hyp_val.json",
        "J2": "results/colab/w2c_L_wmt_consist/L_wmt_consist_s0__hyp_val.json",
        "J2-F": "results/colab/w2c_L_wmt_consist_frozen/L_wmt_consist_frozen_s0__hyp_val.json"}


def rule_outputs(h, m=1):
    """stable x m on an arm's saved val outputs: (passes, texts) per sentence."""
    out = []
    for i, e in enumerate(h["nfe_val"]):
        texts = [h["hyp_val_capped"][str(t)][i] for t in range(1, e + 1)]
        assert texts[-1] == h["hyp_val"][i]
        out.append(CD.stable_stop(texts, e, m))
    return out


@pytest.mark.parametrize("arm", list(ARMS))
def test_stable_rule_reproduces_r042(arm):
    """The rule D1 times, applied to the saved capped hypotheses, reproduces results/j2f_stop_rule_val_s0.json."""
    rec = json.loads((ROOT / "results/j2f_stop_rule_val_s0.json").read_text(encoding="utf-8"))["arms"][arm]
    h = _hyp_val(ARMS[arm])
    refs_path = ROOT / "data/wmt14/newstest2013_ref.txt"
    for m in (1, 2):
        res = rule_outputs(h, m)
        assert abs(sum(p for p, _ in res) / len(res) - rec[f"stable x{m}"]["passes"]) < 1e-9
        if refs_path.is_file():      # the public newstest2013 references (git-ignored; R042)
            import sacrebleu
            refs = refs_path.read_text(encoding="utf-8").splitlines()
            bleu = sacrebleu.corpus_bleu([t for _, t in res], [refs], lowercase=True).score
            assert abs(bleu - rec[f"stable x{m}"]["bleu"]) < 1e-9


def test_online_stable_rule_equals_saved_output_rule():
    """jacobi_stop's stable1 stops where stable_stop stops on the same per-pass texts, including texts that repeat
    while the codes still change, an exact stop before any repeat, a fixed point at pass 1, and the cap."""
    S, V, cap = 4, 6, 8
    texts_of = {0: "a", 1: "a", 2: "b", 3: "c", 4: "c", 5: "d"}         # code value -> text (several codes, one text)
    scripts = {(1, 2, 3, 4, 4): "stable", (2, 3, 5, 5): "fixed", (0,): "fixed", (1, 2, 3, 5, 2, 3, 5, 2): "cap",
               (2, 0, 1, 1): "stable"}
    for traj, want in scripts.items():
        seq = [traj[min(t, len(traj) - 1)] for t in range(cap)]       # every position takes code seq[t-1] at pass t
        calls = iter(seq)

        def pass_fn(codes):
            lg = torch.full((1, S, V), -1.0)
            lg[0, :, next(calls)] = 1.0
            return lg

        decode = lambda c: texts_of[int(c[0, 0])]
        codes, passes, stop = CD.jacobi_stop(pass_fn, S, "cpu", cap, "stable1", decode=decode, initial_fixed_ok=True)
        exact = next((t + 1 for t in range(cap) if seq[t] == (seq[t - 1] if t else 0)), cap)
        assert (passes, decode(codes)) == CD.stable_stop([texts_of[v] for v in seq], exact, 1), traj
        assert stop == want, traj


# ------------------------------------------------------------------------------------------------ E1 sampler
def test_level_calls_match_teacher_forced_logits_ts_like():
    """The sampler's per-level distribution is the model's: for any codes, the level-j call on the input built from
    levels < j equals the block-diagonal forward's level-j logits."""
    tok, net = tiny_ts_like()
    vq = tok.vq
    g = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 32, (1, sum(vq.levels)), generator=g)
    pre = torch.zeros(1, 1, dtype=torch.long)
    z, pm = CD.prefix_inputs(net, pre)
    with torch.no_grad():
        tf = net(torch.cat([z, vq.idx_to_ncm_input(codes)], 1))          # levels 1..K-1
    cache = CD.PrefixCache(net, z, pm)
    lv = [int(l) for l in vq.levels]
    for j in range(1, len(lv)):
        _, x_in = CD.canonical_state(net, codes, j)
        got = CD._level_call(net, cache, x_in, j)
        off = sum(lv[1:j])
        assert torch.allclose(got, tf[:, off:off + lv[j]], atol=1e-5), j


def test_sample_from_cut_keeps_coarse_levels_and_is_greedy_at_low_temperature():
    tok, net = tiny_ts_like()
    vq = tok.vq
    lv = [int(l) for l in vq.levels]
    g = torch.Generator().manual_seed(11)
    l0 = torch.randint(0, 32, (1, 1), generator=g)
    pre = torch.zeros(1, 1, dtype=torch.long)
    greedy = CD.seq_greedy(net, pre, cached=True, l0=l0)
    cache = CD.PrefixCache(net, *CD.prefix_inputs(net, pre))
    for k in (1, 5, 11):
        out = CD.sample_from_cut(net, cache, greedy, k, 4, torch.Generator().manual_seed(k), temperature=1e-6)
        assert torch.equal(out, greedy.expand(4, -1)), k       # ancestral greedy from the greedy prefix = greedy
        hot = CD.sample_from_cut(net, cache, greedy, k, 64, torch.Generator().manual_seed(k), temperature=1.0)
        assert torch.equal(hot[:, :sum(lv[:k])], greedy[:, :sum(lv[:k])].expand(64, -1))
        assert hot[:, sum(lv[:k]):].unique(dim=0).shape[0] > 1    # T = 1 rows differ


# ------------------------------------------------------------------------------------------------ the bench script
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_batch1", ROOT / "scripts" / "bench_batch1.py")
BB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BB)


def test_block_flops_convention_matches_what_runs():
    """flops.py's 24 d^2 + 4 d * keys per row and layer is what FlopCounterMode counts for the generator blocks with
    a dense mask (every key computed)."""
    tok, net, x = tiny_wmt_like()
    u = BB.generator_block_flops(net)
    d, P, T = 32, net.prefix_len, sum(WMT_LEVELS[1:])
    for q, k, key in ((P + T, P + T, "pass"), (P, P, "prefix")):
        h = torch.randn(1, q, d)
        mask = torch.zeros(1, 1, q, k)
        rope = (net.rope_cos[:q], net.rope_sin[:q])
        def run():
            y = h
            for blk in net.transformer.blocks:
                y = blk(y, attn_mask=mask, rope_cos_sin=rope)
        assert BB.counted_flops(run) == u[key]


def test_bench_wmt_like_end_to_end():
    """Configurations decode as the saved-output rule says, cached = eager, flops per item add up, gate fields."""
    tok, net, x = tiny_wmt_like()
    items = [BB.Item(x[i:i + 1]) for i in range(x.shape[0])]
    detok = lambda ids: " ".join(str(t) for t in ids[:ids.index(0)] if t != PAD) if 0 in ids else " ".join(map(str, ids))
    cfgs = BB.make_configs("wmt14", {"J0": net, "J2-F": net}, tok, detok, len(WMT_LEVELS) + 1)
    r = BB.run_timing(cfgs, items, "cpu", "fp32", repeats=2, warmup=1)
    for side in ("seq", "jacobi exact", "jacobi stable1"):
        a, b = r[f"J0 {side} eager"], r[f"J0 {side} cached"]
        assert a["out"] == b["out"] and a["passes"] == b["passes"]
    assert r["J0 jacobi exact eager"]["out"] == r["J0 seq eager"]["out"]
    assert all(p <= e for p, e in zip(r["J0 jacobi stable1 eager"]["passes"], r["J0 jacobi exact eager"]["passes"]))
    assert not any(r[n]["nondeterministic"] for n in r)
    unit = BB.flop_units(net, tok, items[0])
    e, g, rows = BB.flops_per_item("J0 jacobi stable1 cached", "stable1", 3, unit)
    assert e == unit["count_prefix"] + 3 * unit["count_pass_cached"] + 3 * unit["count_decode"]
    assert rows == net.prefix_len + 3 * sum(WMT_LEVELS[1:])
    assert unit["count_seq_eager"] > unit["count_seq_cached"] and unit["count_pass"] > unit["count_pass_cached"]


def test_bench_sudoku_released(released_sudoku):
    from ncmr.sudoku import cells
    bundle, x = released_sudoku
    net, tok = bundle.net, bundle.vqvae
    items = [BB.Item(x[i:i + 1], {"clues": cells(x[i].tolist()), "solution": None}) for i in range(3)]
    cfgs = BB.make_configs("sudoku", {"S": net}, tok, None, len(tok.vq.levels) + 1)
    r = BB.run_timing(cfgs, items, "cpu", "fp32", repeats=1, warmup=1)
    g = BB.integrity_sudoku(r)
    assert BB.gate_ok(g) and all(v == [] for v in g.values())
    assert all(s in ("check", "fixed") for s in r["jacobi verifier cached"]["stop"])


def test_mismatch_tolerance_reads_the_gate():
    assert BB.gate_ok({"a": [1, 2], "b": []}) and not BB.gate_ok({"a": [1, 2, 3]})
    c = BB.ratio_ci([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert c["ratio"] == 0.5 and c["ci95"] == [0.5, 0.5]
