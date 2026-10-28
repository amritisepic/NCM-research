"""W2M-G (docs/preregistration.md, "W2M-G"): src/ncmr/w2mg.py, scripts/w2mg_eval.py and scripts/w2mg_gate.py on tiny CPU
models (one test on the released generator). Gen-PPL scorers are replaced by a text-dependent stand-in; everything else
(sampler, banks, JS, root reliance, shares, readings, caching, replication) is the real code."""
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncmr import w2m as W
from ncmr import w2mg as G
from ncmr.corruption import CorruptionCfg
from ncmr.models import build_ncm_generator
from test_audit_hierarchy import tiny_text_tokenizer

ROOT = Path(__file__).resolve().parents[1]
LV = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
OFF = lambda c: sum(LV[:c])                                  # first code of level c


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("w2mg_eval")
GATE = _load("w2mg_gate")


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def world():
    tok = tiny_text_tokenizer(learned_l0=True, seed=0)
    for p in tok.parameters():
        p.requires_grad_(False)
    tok_ids = {id(p) for p in tok.parameters()}
    nets = {}
    for i, name in enumerate(["released", "m0", "m2", "m3", "m4"]):
        torch.manual_seed(100 + i)
        net = build_ncm_generator(tok, n_layer=2, n_head=2, n_embd=32, dropout=0.0).eval()
        g = torch.Generator().manual_seed(200 + i)
        with torch.no_grad():
            for p in net.parameters():                       # spread the weights so the models really differ
                if id(p) not in tok_ids:
                    p.add_(0.05 * torch.randn(p.shape, generator=g))
        nets[name] = net
    flag = W.FlagVector(int(tok.embed_dim))
    with torch.no_grad():
        flag.w.copy_(0.5 * torch.randn(flag.w.shape, generator=torch.Generator().manual_seed(7)))
    x = torch.randint(2, 32, (4, 128), generator=torch.Generator().manual_seed(3))
    return SimpleNamespace(tok=tok, nets=nets, flag=flag, L0=torch.ones(32) / 32, x=x,
                           pe=lambda b: tok.encoder.wte(torch.zeros(b, 1, dtype=torch.long)))


def member(w, name, f=None):
    return G.Member(w.nets[name], w.flag, f) if f is not None else G.Member(w.nets[name])


def tf_logits(w, net, codes, flag=None, f=0.0):
    """Teacher-forced logits (n, 640, V) on `codes` with a constant flag f on every input position."""
    inp = w.tok.vq.idx_to_ncm_input(codes)
    fl = None if flag is None else torch.full(inp.shape[:2], float(f))
    return W.forward_logits(net, w.pe, inp, "cpu", flag, fl, bs=4)


# ------------------------------------------------------------------------------------------------ AC1
def _check_trace_against_teacher_forcing(w, fn, members_by_level, n=3, seed=5):
    trace = []
    codes = G.sample_codes(fn, w.tok, w.L0, n, seed, "cpu", trace=trace)
    assert codes.shape == (n, sum(LV)) and len(trace) == 11
    tf = {}
    for j in range(1, 12):
        m = members_by_level(j)
        key = id(m)
        if key not in tf:
            tf[key] = tf_logits(w, m.net, codes, m.flag, m.f)
        lo = sum(LV[1:j])
        assert torch.allclose(trace[j - 1], tf[key][:, lo:lo + LV[j]], atol=1e-5, rtol=1e-5), j
    return codes, trace


def test_sampler_logits_equal_teacher_forced_logits_pure_crossed_and_flagged(world):
    w = world
    m2, m3, m4a, m4b = member(w, "m2"), member(w, "m3"), member(w, "m4", 0.5), member(w, "m4", 1.0)
    _check_trace_against_teacher_forcing(w, G.pure(m2), G.pure(m2))
    _check_trace_against_teacher_forcing(w, G.crossed(m2, m3, 4), G.crossed(m2, m3, 4))
    _check_trace_against_teacher_forcing(w, G.pure(m4a), G.pure(m4a))                    # constant flag 0.5
    fn = G.crossed(m4b, m3, 6)                                                              # flagged prefix, plain suffix
    _check_trace_against_teacher_forcing(w, fn, fn)
    fn = G.crossed(m2, m4a, 2)                                                              # plain prefix, flagged suffix
    _check_trace_against_teacher_forcing(w, fn, fn)


def test_the_flag_changes_the_logits_and_the_models_differ(world):
    """Guards the equivalence test against being vacuous."""
    w = world
    tr0, tr5, tr3 = [], [], []
    G.sample_codes(G.pure(member(w, "m4", 0.0)), w.tok, w.L0, 2, 1, "cpu", trace=tr0)
    G.sample_codes(G.pure(member(w, "m4", 0.5)), w.tok, w.L0, 2, 1, "cpu", trace=tr5)
    G.sample_codes(G.pure(member(w, "m3")), w.tok, w.L0, 2, 1, "cpu", trace=tr3)
    assert not torch.allclose(tr0[0], tr5[0], atol=1e-3) and not torch.allclose(tr0[0], tr3[0], atol=1e-3)


def test_sampler_matches_teacher_forcing_on_the_released_generator():
    from ncmr.ncm_ref_loader import load_ncm
    B = load_ncm(str(ROOT / "artifacts" / "tinystories"), device="cpu")
    net, tok = B.net.eval(), B.vqvae.eval()
    flag = W.FlagVector(int(tok.embed_dim))
    with torch.no_grad():
        flag.w.copy_(0.1 * torch.randn(flag.w.shape, generator=torch.Generator().manual_seed(1)))
    L0 = B.ckpt_extra["inference"]["level0_unigram"].float()
    pe = lambda b: tok.encoder.wte(torch.zeros(b, 1, dtype=torch.long))
    assert [int(l) for l in tok.vq.levels] == LV and sum(LV) == 641
    for f in (0.0, 1.0):
        trace = []
        codes = G.sample_codes(G.pure(G.Member(net, flag, f)), tok, L0, 2, 3, "cpu", trace=trace)
        inp = tok.vq.idx_to_ncm_input(codes)
        tf = W.forward_logits(net, pe, inp, "cpu", flag, torch.full(inp.shape[:2], f), bs=2)
        for j in (1, 5, 11):
            lo = sum(LV[1:j])
            assert torch.allclose(trace[j - 1], tf[:, lo:lo + LV[j]], atol=1e-4, rtol=1e-4), (f, j)


# ------------------------------------------------------------------------------------------------ AC2
def test_pure_cells_are_identical_at_every_cut_and_crossed_same_model_equals_pure(world):
    w = world
    m2 = member(w, "m2")
    ref = G.sample_codes(G.pure(m2), w.tok, w.L0, 4, 11, "cpu")
    for c in (2, 4, 6):
        assert torch.equal(G.sample_codes(G.crossed(m2, member(w, "m2"), c), w.tok, w.L0, 4, 11, "cpu"), ref), c
        assert torch.equal(G.sample_codes(G.crossed(m2, m2, c), w.tok, w.L0, 4, 11, "cpu"), ref), c
    assert torch.equal(G.sample_codes(G.pure(m2), w.tok, w.L0, 4, 11, "cpu"), ref)           # deterministic
    assert not torch.equal(G.sample_codes(G.pure(m2), w.tok, w.L0, 4, 12, "cpu"), ref)       # the seed matters


def test_two_cells_share_level_zero_and_everything_before_the_cut(world):
    w = world
    m2, m3 = member(w, "m2"), member(w, "m3")
    p2 = G.sample_codes(G.pure(m2), w.tok, w.L0, 6, 21, "cpu")
    p3 = G.sample_codes(G.pure(m3), w.tok, w.L0, 6, 21, "cpu")
    assert torch.equal(p2[:, :1], p3[:, :1]) and not torch.equal(p2, p3)                      # common level-0 codes
    for c in (2, 4, 6):
        x23 = G.sample_codes(G.crossed(m2, m3, c), w.tok, w.L0, 6, 21, "cpu")
        x32 = G.sample_codes(G.crossed(m3, m2, c), w.tok, w.L0, 6, 21, "cpu")
        assert torch.equal(x23[:, :OFF(c)], p2[:, :OFF(c)]) and torch.equal(x32[:, :OFF(c)], p3[:, :OFF(c)]), c
        assert not torch.equal(x23, p2) and not torch.equal(x32, p3), c                       # ... and differ after it
        assert not torch.equal(x23[:, OFF(c):], p2[:, OFF(c):])
    for c in (2, 4, 6):                       # the prefix model samples levels 1..c-1, the suffix model levels c..11
        tr = []
        cc = G.sample_codes(G.crossed(m2, m3, c), w.tok, w.L0, 3, 21, "cpu", trace=tr)
        for lvl in (c - 1, c):
            net = w.nets["m2" if lvl < c else "m3"]
            lo = sum(LV[1:lvl])
            assert torch.allclose(tr[lvl - 1], tf_logits(w, net, cc)[:, lo:lo + LV[lvl]], atol=1e-5), (c, lvl)
            other = w.nets["m3" if lvl < c else "m2"]
            assert not torch.allclose(tr[lvl - 1], tf_logits(w, other, cc)[:, lo:lo + LV[lvl]], atol=1e-4), (c, lvl)
    a = G.sample_codes(G.crossed(m2, m3, 2), w.tok, w.L0, 6, 21, "cpu")
    b = G.sample_codes(G.crossed(m2, m3, 4), w.tok, w.L0, 6, 21, "cpu")
    assert torch.equal(a[:, :OFF(2)], b[:, :OFF(2)])


def test_noise_is_one_generator_drawn_in_level_order_whatever_the_model(world):
    """Re-derive the Gumbel noise from a fresh generator: level 0 multinomial, then one uniform tensor of the logits'
    shape per level. Every model's codes are the argmax of its own logits plus that noise."""
    w = world
    for fn in (G.pure(member(w, "m2")), G.pure(member(w, "m3")), G.crossed(member(w, "m2"), member(w, "m3"), 4)):
        trace = []
        codes = G.sample_codes(fn, w.tok, w.L0, 3, 33, "cpu", trace=trace)
        g = torch.Generator().manual_seed(33)
        assert torch.equal(codes[:, :1], torch.multinomial(w.L0, 3, replacement=True, generator=g).view(3, 1))
        for j in range(1, 12):
            u = torch.rand(trace[j - 1].shape, generator=g).clamp_min(1e-20)
            assert torch.equal(codes[:, OFF(j):OFF(j) + LV[j]], (trace[j - 1] - torch.log(-torch.log(u))).argmax(-1)), j


# ------------------------------------------------------------------------------------------------ AC3
def test_flag_zero_is_exactly_no_flag_and_positive_f_adds_f_times_w(world):
    w = world
    x = torch.randn(2, 8, 8)
    assert torch.equal(G.Member(w.nets["m4"], w.flag, 0.0).add_flag(x), x)
    assert G.Member(w.nets["m4"]).add_flag(x) is x
    for f in (0.5, 1.0):
        got = G.Member(w.nets["m4"], w.flag, f).add_flag(x)
        assert torch.allclose(got, x + f * w.flag.w, atol=1e-7) and not torch.equal(got, x)
    with pytest.raises(ValueError):
        G.Member(w.nets["m4"], None, 0.5)
    # teacher-forced logits and sampled codes / logits at f = 0 equal the no-flag ones bit for bit
    codes = G.sample_codes(G.pure(G.Member(w.nets["m4"])), w.tok, w.L0, 3, 8, "cpu")
    assert torch.equal(tf_logits(w, w.nets["m4"], codes, w.flag, 0.0), tf_logits(w, w.nets["m4"], codes))
    t0, t1 = [], []
    c0 = G.sample_codes(G.pure(G.Member(w.nets["m4"], w.flag, 0.0)), w.tok, w.L0, 3, 8, "cpu", trace=t0)
    c1 = G.sample_codes(G.pure(G.Member(w.nets["m4"])), w.tok, w.L0, 3, 8, "cpu", trace=t1)
    assert torch.equal(c0, c1) and all(torch.equal(a, b) for a, b in zip(t0, t1))


def test_positive_flag_enters_every_level_input(world):
    """With f > 0 the level-j logits equal the plain call on (x_in + f w), for every level 1..11."""
    from ncmr import cached_decode as CD
    w = world
    net, vq = w.nets["m4"], w.tok.vq
    codes = G.sample_codes(G.pure(G.Member(net)), w.tok, w.L0, 2, 4, "cpu")
    pre = torch.zeros(1, 1, dtype=torch.long)
    cache = CD.PrefixCache(net, *CD.prefix_inputs(net, pre)).expand(2)
    trace = []
    G.sample_codes(G.pure(G.Member(net, w.flag, 1.0)), w.tok, w.L0, 2, 4, "cpu", trace=trace)
    # the sampled codes under f = 1 differ from `codes`, so rebuild the hierarchy from the codes that run saw
    c1 = G.sample_codes(G.pure(G.Member(net, w.flag, 1.0)), w.tok, w.L0, 2, 4, "cpu")
    z_hat = torch.zeros(2, 128, vq.embed_dim)
    for j in range(1, 12):
        z_hat, x_in = vq.get_next_autoregressive_input(j - 1, z_hat, vq.embeddings[j - 1](c1[:, OFF(j - 1):OFF(j)]))
        with torch.no_grad():
            ref = CD._level_call(net, cache, x_in + 1.0 * w.flag.w, j)
        assert torch.allclose(trace[j - 1], ref, atol=1e-5), j
        with torch.no_grad():
            assert not torch.allclose(ref, CD._level_call(net, cache, x_in, j), atol=1e-3), j


# ------------------------------------------------------------------------------------------------ AC4: JS divergence
def _js_np(la, lb):
    p, q = np.exp(la - la.max(-1, keepdims=True)), np.exp(lb - lb.max(-1, keepdims=True))
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)
    H = lambda r: -(np.where(r > 0, r * np.log(np.where(r > 0, r, 1)), 0)).sum(-1)
    return H(m) - 0.5 * H(p) - 0.5 * H(q)


def test_js_from_logits_matches_the_entropy_formula_and_is_symmetric_nonnegative():
    g = torch.Generator().manual_seed(0)
    la, lb = 3 * torch.randn(5, 7, 50, generator=g), 3 * torch.randn(5, 7, 50, generator=g)
    got = G.js_from_logits(la, lb)
    assert np.allclose(got.numpy(), _js_np(la.double().numpy(), lb.double().numpy()), atol=1e-12)
    assert torch.equal(got, G.js_from_logits(lb, la)) and bool((got >= 0).all()) and float(got.min()) > 0
    assert float(G.js_from_logits(la, la).abs().max()) < 1e-12
    assert float(got.max()) <= math.log(2) + 1e-12
    big = torch.zeros(1, 4)
    big[0, 0] = 1e4
    assert abs(float(G.js_from_logits(big, torch.roll(big, 1, -1))) - math.log(2)) < 1e-9      # disjoint supports: log 2


def test_js_bank_zero_for_identical_models_symmetric_and_pooled_as_registered(world):
    w = world
    codes = G.sample_codes(G.pure(member(w, "m2")), w.tok, w.L0, 5, 2, "cpu")
    same = G.js_bank(w.nets["m2"], w.nets["m2"], w.tok, w.pe, codes, "cpu", bs=2)
    assert same.shape == (5, 4) and float(np.abs(same).max()) < 1e-12
    ab = G.js_bank(w.nets["m2"], w.nets["m3"], w.tok, w.pe, codes, "cpu", bs=2)
    ba = G.js_bank(w.nets["m3"], w.nets["m2"], w.tok, w.pe, codes, "cpu", bs=3)
    assert np.allclose(ab, ba, atol=1e-12) and (ab >= 0).all() and ab.min() > 0
    # independent: full logits, per-position JS by the entropy formula, per-level mean over positions, mean over levels 2..5
    la, lb = tf_logits(w, w.nets["m2"], codes).numpy().astype(np.float64), tf_logits(w, w.nets["m3"], codes).numpy().astype(np.float64)
    per_pos = _js_np(la, lb)                                                              # (5, 640)
    lvl = [per_pos[:, sum(LV[1:j]):sum(LV[1:j]) + LV[j]] for j in (2, 3, 4, 5)]
    assert np.allclose(ab, np.stack([l.mean(1) for l in lvl], 1), atol=1e-9)
    pooled = G.pool_js(ab)
    assert abs(pooled["J"] - np.mean([l.mean() for l in lvl])) < 1e-9
    assert np.allclose(pooled["per_level"], [l.mean() for l in lvl], atol=1e-9)
    assert abs(pooled["J"] - np.concatenate([l.ravel() for l in lvl]).mean()) > 1e-6        # not the all-position mean


def test_pool_js_is_mean_of_level_means():
    a = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 1.0, 0.0]])
    p = G.pool_js(a)
    assert p["per_level"] == [2.0, 2.0, 2.0, 2.0] and p["J"] == 2.0


def test_bootstrap_ratio_point_ci_and_both_banks_resampled():
    rng = np.random.default_rng(0)
    clean = rng.uniform(0.5, 1.5, (64, 4))
    gen = 3.0 * clean
    r = G.bootstrap_ratio(gen, clean, 200, 0)
    assert abs(r["r"] - 3.0) < 1e-12 and r["lo"] <= r["r"] <= r["hi"]
    # a constant bank and a noisy one: resampling only one bank would leave a zero-width interval
    flat, noisy = np.ones((64, 4)), rng.uniform(0.5, 1.5, (64, 4))
    a, b = G.bootstrap_ratio(noisy, flat, 200, 0), G.bootstrap_ratio(flat, noisy, 200, 0)
    assert a["hi"] - a["lo"] > 0.02 and b["hi"] - b["lo"] > 0.02
    assert len(r["per_level"]) == 4 and all(x["lo"] <= x["r"] <= x["hi"] for x in r["per_level"])
    assert G.bootstrap_ratio(gen, clean, 200, 0) == r and G.bootstrap_ratio(gen, clean, 200, 1) != r
    assert G.Protocol().n_boot == 2000 and G.Protocol().boot_seed == 0


# ------------------------------------------------------------------------------------------------ AC4: root reliance
def test_draw_roots_are_seeded_cpu_unigram_draws():
    L0 = torch.softmax(torch.randn(32, generator=torch.Generator().manual_seed(1)), 0)
    r = G.draw_roots(L0, 5, 8, 2027)
    g = torch.Generator().manual_seed(2027)
    assert r.shape == (5, 8) and torch.equal(r, torch.multinomial(L0, 40, replacement=True, generator=g).view(5, 8))
    assert torch.equal(r, G.draw_roots(L0, 5)) and not torch.equal(r, G.draw_roots(L0, 5, 8, 2028))
    assert G.Protocol().root_seed == 2027 and G.Protocol().n_roots == 8


def test_level1_nll_with_the_true_root_is_the_teacher_forced_level1_nll(world):
    w = world
    canon = w.tok.vq.z_to_idx(w.tok._encode_pre_quant(w.x))
    for name in ("m2", "m4"):
        got = G.level1_nll(w.nets[name], w.tok, canon[:, 0], canon[:, 1:9], "cpu", bs=3)
        lg = tf_logits(w, w.nets[name], canon)[:, :8]
        want = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), canon[:, 1:9].reshape(-1), reduction="none").view(-1, 8).sum(1)
        assert torch.allclose(got, want, atol=1e-4)


def test_root_reliance_matches_a_teacher_forced_brute_force(world):
    w = world
    canon = w.tok.vq.z_to_idx(w.tok._encode_pre_quant(w.x))
    r = G.root_reliance({"m2": w.nets["m2"], "m3": w.nets["m3"]}, w.tok, w.L0, canon, "cpu", 8, 2027)
    assert torch.equal(r["roots"], G.draw_roots(w.L0, 4, 8, 2027))
    for name in ("m2", "m3"):
        def nll(root):
            c = canon.clone()
            c[:, 0] = root
            lg = tf_logits(w, w.nets[name], c)[:, :8]                 # level 1 sees only the root (and the prefix)
            return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), canon[:, 1:9].reshape(-1), reduction="none").view(-1, 8).sum(1)
        want = torch.stack([nll(r["roots"][:, d]) - nll(canon[:, 0]) for d in range(8)], 1).mean(1)
        assert np.allclose(r["rho"][name], want.numpy(), atol=1e-4), name
    assert float(np.abs(r["rho"]["m2"]).max()) > 1e-3 and not np.allclose(r["rho"]["m2"], r["rho"]["m3"], atol=1e-4)


def test_root_reliance_is_zero_for_a_model_that_ignores_the_root_and_draws_are_shared(world, monkeypatch):
    w = world
    deaf = build_ncm_generator(w.tok, n_layer=2, n_head=2, n_embd=32, dropout=0.0).eval()
    with torch.no_grad():
        deaf.transformer.wte.weight.zero_()                           # the code inputs are projected to a constant
    canon = w.tok.vq.z_to_idx(w.tok._encode_pre_quant(w.x))
    seen = {}
    real = G.level1_nll

    def spy(net, tok, roots, targets, device, **kw):
        seen.setdefault(id(net), []).append(roots.clone())
        return real(net, tok, roots, targets, device, **kw)

    monkeypatch.setattr(G, "level1_nll", spy)
    r = G.root_reliance({"deaf": deaf, "m2": w.nets["m2"], "m3": w.nets["m3"]}, w.tok, w.L0, canon, "cpu")
    assert float(np.abs(r["rho"]["deaf"]).max()) == 0.0 and float(np.abs(r["rho"]["m2"]).max()) > 1e-3
    drawn = [seen[id(n)][1:] for n in (deaf, w.nets["m2"], w.nets["m3"])]              # [0] is the true-root call
    assert all(len(d) == 8 for d in drawn)
    for d in range(8):
        assert torch.equal(drawn[0][d], drawn[1][d]) and torch.equal(drawn[0][d], drawn[2][d])
        assert torch.equal(drawn[0][d], r["roots"][:, d])


def test_paired_ci():
    a = np.array([2.0, 3.0, 4.0, 5.0] * 10)
    b = a - np.array([0.5, 1.0, 1.5, 2.0] * 10)
    d = G.paired_ci(a, b, 500, 0)
    assert abs(d["mean"] - 1.25) < 1e-12 and 0 < d["lo"] <= d["mean"] <= d["hi"]
    d2 = G.paired_ci(b, a, 500, 0)
    assert d2["hi"] < 0 and abs(d2["mean"] + 1.25) < 1e-12


# ------------------------------------------------------------------------------------------------ (iv) diagnostics
def test_constant_flag_nll_and_kl_diagnostics(world):
    w = world
    canon = w.tok.vq.z_to_idx(w.tok._encode_pre_quant(w.x))
    inp, tg = w.tok.vq.idx_to_ncm_input(canon), canon[:, 1:]
    net = w.nets["m4"]
    n0 = G.nll_const_flag(net, w.flag, w.pe, inp, tg, 0.0, "cpu", 2)
    plain = W.input_nll(W.forward_logits(net, w.pe, inp, "cpu"), tg).double().numpy()
    assert np.array_equal(n0, plain)                                                  # f = 0 is the no-flag NLL
    n1 = G.nll_const_flag(net, w.flag, w.pe, inp, tg, 1.0, "cpu", 2)
    want = W.input_nll(W.forward_logits(net, w.pe, inp + w.flag.w, "cpu"), tg).double().numpy()
    assert np.allclose(n1, want, atol=1e-4) and not np.allclose(n1, n0, atol=1e-3)
    tf = torch.rand(inp.shape[:2], generator=torch.Generator().manual_seed(0))
    nt = G.nll_const_flag(net, w.flag, w.pe, inp, tg, 0.0, "cpu", 2, flags=tf)
    assert np.allclose(nt, W.input_nll(W.forward_logits(net, w.pe, inp + tf[..., None] * w.flag.w, "cpu"), tg).double().numpy(), atol=1e-4)
    kl = G.kl_const_flag(net, w.flag, w.pe, inp, (0.0, 0.5, 1.0), "cpu", bs=2)
    p0 = W.forward_logits(net, w.pe, inp, "cpu").double().log_softmax(-1)
    for f in (0.5, 1.0):
        pf = W.forward_logits(net, w.pe, inp + f * w.flag.w, "cpu").double().log_softmax(-1)
        assert abs(kl[f] - float((p0.exp() * (p0 - pf)).sum(-1).mean())) < 1e-5, f
    assert kl[0.0] == 0.0 and 0 < kl[0.5] < kl[1.0]


# ------------------------------------------------------------------------------------------------ AC5: shares and readings
def test_shares_point_values_and_jackknife():
    l22, l33 = np.array([3.0, 3.0, 3.0]), np.array([1.0, 1.0, 1.0])
    s = G.shares(l22, l33, np.array([1.5, 1.5, 1.5]), np.array([2.5, 2.5, 2.5]))
    assert s["delta"] == 2.0 and s["pi"] == 0.25 and s["sigma"] == 0.75 and s["interaction"] == 0.0
    assert s["pi_se"] == 0.0 and s["sigma_se"] == 0.0
    # varying groups: compare with a hand-written leave-one-out
    rng = np.random.default_rng(1)
    a = [rng.uniform(1, 2, 8) + o for o in (2.0, 0.0, 0.8, 1.7)]
    s = G.shares(*a)
    pi = lambda idx: (a[2][idx].mean() - a[1][idx].mean()) / (a[0][idx].mean() - a[1][idx].mean())
    full = np.arange(8)
    loo = np.array([pi(np.delete(full, i)) for i in range(8)])
    assert abs(s["pi"] - pi(full)) < 1e-12
    assert abs(s["pi_se"] - math.sqrt(7 / 8 * ((loo - loo.mean()) ** 2).sum())) < 1e-12 and s["pi_se"] > 0
    assert abs(s["interaction"] - (1 - s["pi"] - s["sigma"])) < 1e-12
    with pytest.raises(AssertionError):
        G.shares(a[0], a[1], a[2][:5], a[3])                               # the pairing is by group


def test_gate_boundaries():
    assert G.gate_ii(0.25, 1e-9) and G.gate_ii(0.3, 0.01)
    assert not G.gate_ii(0.2499, 0.5) and not G.gate_ii(0.5, 0.0) and not G.gate_ii(0.5, -0.1)
    assert not G.gate_ii(float("nan"), 0.5) and not G.gate_ii(0.5, float("nan"))


def test_reading_ii_boundaries():
    assert G.read_ii(0.5, 0.75, 0.0) == "early-state selection"
    assert G.read_ii(0.4999, 0.75, 0.0) == "distributed" and G.read_ii(0.5, 0.7499, 0.0) == "distributed"
    assert G.read_ii(0.0, 0.0, 0.8) == "conditional rendering" and G.read_ii(0.0, 0.0, 0.7999) == "distributed"
    assert G.read_ii(0.5, 0.75, 0.8) == "both"


def test_sanity_ii_three_se():
    ok = G.sanity_ii(42.62 + 3.0, 1.0, 21.69 - 3.0, 1.0)
    assert ok["within_3se"] and abs(ok["z_m2m2"] - 3.0) < 1e-9 and abs(ok["z_m3m3"] + 3.0) < 1e-9
    assert not G.sanity_ii(42.62 + 3.01, 1.0, 21.69, 1.0)["within_3se"]
    assert not G.sanity_ii(42.62, 1.0, 21.69 - 3.01, 1.0)["within_3se"]


def test_reading_i_thresholds():
    R = lambda r, lo, hi: {"r": r, "lo": lo, "hi": hi}
    assert G.read_i({k: R(3.0, 1.2001, 5) for k in "abc"}) == "holds"
    assert G.read_i({"a": R(2.9999, 2, 5), "b": R(4, 2, 5), "c": R(4, 2, 5)}) == "mixed"        # r_b >= 3 for ALL banks
    assert G.read_i({"a": R(4, 1.2, 5), "b": R(4, 2, 5), "c": R(4, 2, 5)}) == "mixed"           # lower bound must exceed 1.2
    assert G.read_i({k: R(1.0, 0.8, 1.2) for k in "abc"}) == "overturned"
    assert G.read_i({"a": R(1.0, 0.8, 1.2), "b": R(1.0, 0.8, 1.2001), "c": R(1.0, 0.8, 1.0)}) == "mixed"
    assert G.read_i({"a": R(1.0, 0.8, 1.2), "b": R(4, 2, 5), "c": R(1.0, 0.8, 1.0)}) == "mixed"


def test_reading_iii_thresholds():
    assert G.read_iii(0.0001, 0.5) == "holds" and G.read_iii(-0.5, -0.0001) == "overturned"
    assert G.read_iii(0.0, 0.5) == "neither" and G.read_iii(-0.5, 0.0) == "neither" and G.read_iii(-0.1, 0.1) == "neither"


def test_iv_rule_and_reading():
    ok = dict(entropy=3.95, dup=12.0, m3_entropy=4.0, m3_dup=10.0)
    R = lambda a, b: {"gpt2-large": a, "TinyStories-33M": b}
    assert G.iv_rule(R(0.8, 0.8), ok["m3_entropy"] - 0.05, ok["m3_dup"] + 2, ok["m3_entropy"], ok["m3_dup"])
    assert not G.iv_rule(R(0.7999, 0.9), 4.0, 10.0, 4.0, 10.0) and not G.iv_rule(R(0.9, 0.7999), 4.0, 10.0, 4.0, 10.0)
    assert not G.iv_rule(R(0.9, 0.9), 4.0 - 0.05 - 1e-9, 10.0, 4.0, 10.0)                    # entropy collapse
    assert not G.iv_rule(R(0.9, 0.9), 4.0, 12.001, 4.0, 10.0)                                 # duplicates
    assert G.rescue(42.0, 22.0, 22.0) == 1.0 and G.rescue(42.0, 42.0, 22.0) == 0.0 and abs(G.rescue(42.0, 26.0, 22.0) - 0.8) < 1e-12
    assert G.read_iv({0.5: False, 1.0: False}, None) == "holds"
    assert G.read_iv({0.5: True, 1.0: False}, {0.5: True}) == "overturned"
    assert G.read_iv({0.5: True, 1.0: False}, {0.5: False}) == "mixed"
    assert G.read_iv({0.5: True, 1.0: True}, {0.5: False, 1.0: True}) == "overturned"
    assert G.read_iv({0.5: True, 1.0: False}, None) == "pending replication"


# ------------------------------------------------------------------------------------------------ decode / re-encode
def test_decode_codes_cuts_at_eos_and_drops_pad_and_reencode_goes_through_text():
    codes = torch.tensor([[5, 1, 6, 0, 7, 7], [1, 1, 3, 4, 1, 2], [0, 5, 5, 5, 5, 5]])
    assert G.decode_codes(_FakeTok(), codes) == [[5, 6], [3, 4, 2], []]
    bpe = SimpleNamespace(decode=lambda ids: " ".join(str(t) for t in ids if t != 3), encode=lambda t: [int(v) + 100 for v in t.split()])
    assert G.reencode(bpe, [[1, 3, 2], []]) == [[101, 102], []]


class _FakeTok(nn.Module):
    def __init__(self):
        super().__init__()
        self.vq = SimpleNamespace(idx_to_z_q=lambda c: F.one_hot(c, 8).float())
        self.decoder = nn.Identity()


# ------------------------------------------------------------------------------------------------ the script
class FakeBPE:
    def decode(self, ids):
        return " ".join(f"w{t}" for t in ids)

    def encode(self, text):
        return [int(w[1:]) for w in text.split()]


@pytest.fixture
def fake_scorers(monkeypatch):
    from ncmr import eval_text

    def gen_ppl(texts, scorer, device=None, batch_size=16):
        h = sum(int(wd[1:]) * (i % 7 + 1) for t in texts for i, wd in enumerate(t.split()))
        return 5.0 + (h % 1000) / 10 + (3.0 if "Tiny" in scorer else 0.0)

    monkeypatch.setattr(eval_text, "gen_ppl", gen_ppl)


SMALL = G.Protocol(n_clean=4, group_n=2, replay_seeds=(0, 1, 2), replicate_seeds=(8, 9, 10), n_boot=50, eval_bs=2)


def make_env(w, evaluate=None):
    from ncmr.eval_text import evaluate_samples
    bpe = FakeBPE()
    ev = evaluate or (lambda ids, groups: evaluate_samples(ids, bpe.decode, groups, scorers=tuple(S.SCORER_IDS.values()), device="cpu"))
    return S.Env("cpu", w.tok, bpe, w.L0, dict(w.nets), w.flag, w.x, ev, {n: "sha-" + n for n in w.nets}, {n: n for n in w.nets})


def _quiet(*_):
    pass


def test_run_writes_every_section_and_registered_cells(world, fake_scorers, tmp_path, monkeypatch):
    w = world
    calls = []
    real = G.sample_codes
    monkeypatch.setattr(G, "sample_codes", lambda fn, tok, L0, n, seed, device, **kw: (calls.append((seed, n)), real(fn, tok, L0, n, seed, device, **kw))[1])
    res = S.run(make_env(w), str(tmp_path), SMALL, 0, _quiet)
    assert res["complete"] is True
    keys = {"m2_m2", "m3_m3", "m4_f0", "m4_f0.5", "m4_f1"} | {f"{a}_c{c}" for a in ("m2to3", "m3to2") for c in (2, 4, 6)}
    assert set(res["cells"]) == keys                                                      # 8 + 3 cells, no replication
    for k, c in res["cells"].items():
        assert c["spec"]["seeds"] == [0, 1, 2] and c["n_groups"] == 3 and c["n"] == 6
        for sc in G.SCORERS:
            m = c[f"genppl_{sc}"]
            assert len(m["per_group"]) == 3 and np.allclose(m["log_loss_per_group"], np.log(m["per_group"]))
            assert abs(m["se"] - m["sd"] / math.sqrt(3)) < 1e-12
    for sec in ("ii", "i", "iii", "iv", "direction1_gate"):
        assert sec in res, sec
    assert {"2", "4", "6"} == set(res["ii"]["shares"]["gpt2-large"]) and res["ii"]["reading"].keys() == set(G.SCORERS)
    assert res["i"]["reading"] in ("holds", "overturned", "mixed") and set(res["iii"]["rho"]) == set(S.RHO_MODELS)
    assert res["iv"]["reading"] == "holds" and set(res["iv"]["seeds_0_7"]["f"]) == {"0", "0.5", "1"}
    # the shares in result.json are the shares of the cells' stored log losses
    ll = lambda k: res["cells"][k]["genppl_gpt2-large"]["log_loss_per_group"]
    want = G.shares(ll("m2_m2"), ll("m3_m3"), ll("m2to3_c2"), ll("m3to2_c2"))
    assert res["ii"]["shares"]["gpt2-large"]["2"]["pi"] == want["pi"] and res["direction1_gate"]["pi_2_gpt2-large"] == want["pi"]
    # (iii): the paired difference is rho(M3) - rho(M2) over the same windows; (i): ratios are J_b / J_clean
    iii = res["iii"]
    assert abs(iii["m3_minus_m2"]["mean"] - (iii["rho"]["m3"]["mean"] - iii["rho"]["m2"]["mean"])) < 1e-9
    assert iii["reading"] == G.read_iii(iii["m3_minus_m2"]["lo"], iii["m3_minus_m2"]["hi"])
    i = res["i"]
    assert set(i["J"]) == {"clean", "released", "m2", "m3"} and i["levels"] == [2, 3, 4, 5]
    for b in ("released", "m2", "m3"):
        assert abs(i["ratios"][b]["r"] - i["J"][b]["J"] / i["J"]["clean"]["J"]) < 1e-9
    canon = w.tok.vq.z_to_idx(w.tok._encode_pre_quant(w.x))
    want = G.pool_js(G.js_bank(w.nets["m2"], w.nets["m3"], w.tok, w.pe, canon, "cpu", bs=2))
    assert abs(i["J"]["clean"]["J"] - want["J"]) < 1e-12
    # the sample banks: 2 groups (seeds 900, 901) of each model, M4 at f = 0; (i) reads the released / M2 / M3 banks
    members = {"released": member(w, "released"), "m2": member(w, "m2"), "m3": member(w, "m3"), "m4f0": member(w, "m4", 0.0)}
    banks = {}
    for b, m in members.items():
        banks[b] = torch.cat([real(G.pure(m), w.tok, w.L0, 2, sd, "cpu") for sd in (900, 901)])
        assert np.array_equal(np.load(tmp_path / "cells" / f"bank_{b}.npy"), banks[b].numpy())
    for b in ("released", "m2", "m3"):
        j = G.pool_js(G.js_bank(w.nets["m2"], w.nets["m3"], w.tok, w.pe, banks[b], "cpu", bs=2))
        assert abs(i["J"][b]["J"] - j["J"]) < 1e-12, b
    m4_in = w.tok.vq.idx_to_ncm_input(banks["m4f0"])
    kl = G.kl_const_flag(w.nets["m4"], w.flag, w.pe, m4_in, (0.5,), "cpu", bs=2)[0.5]
    assert abs(res["iv"]["diagnostics"]["kl_f0_to_f"]["m4_samples"]["0.5"] - kl) < 1e-6
    assert i["reading"] == G.read_i({b: {k: i["ratios"][b][k] for k in ("r", "lo", "hi")} for b in ("released", "m2", "m3")})
    # (iv) diagnostics: f = 0 is the no-flag NLL of M4, KL(f=0 || f=0) is 0, a positive f moves the NLL and the distribution
    dg = res["iv"]["diagnostics"]
    inp = w.tok.vq.idx_to_ncm_input(canon)
    nll0 = W.input_nll(W.forward_logits(w.nets["m4"], w.pe, inp, "cpu"), canon[:, 1:]).double().numpy()
    assert abs(dg["nll_per_window"]["clean"]["0"]["mean"] - nll0.mean()) < 1e-9 and dg["nll_per_window"]["clean"]["0"]["delta_vs_f0_mean"] == 0.0
    assert abs(dg["nll_per_window"]["clean"]["1"]["delta_vs_f0_mean"]) > 1e-6 and "true_flags" in dg["nll_per_window"]["corrupted"]
    assert set(dg["kl_f0_to_f"]) == {"clean", "corrupted", "m4_samples"}
    assert all(v["0"] == 0.0 and 0 < v["0.5"] < v["1"] for v in dg["kl_f0_to_f"].values())
    # seeds and group sizes: cells use seeds 0-2 (x 13 cells... 11 cells x 3 groups), banks 900 and 901
    assert all(n == 2 for _, n in calls)
    seeds = [s for s, _ in calls]
    assert sorted(set(seeds) - {900, 901}) == [0, 1, 2] and seeds.count(900) == 4 and seeds.count(901) == 4
    assert len(seeds) == 11 * 3 + 8
    assert not any(s >= 8 and s < 900 for s in seeds)
    txt = (tmp_path / "result.json").read_text()
    assert str(tmp_path) not in txt and os.path.expanduser("~") not in txt


def test_registered_banks_seeds_and_counts(world, monkeypatch):
    P = G.Protocol()
    assert (P.n_clean, P.group_n, P.bank_seeds, P.replay_seeds, P.replicate_seeds) == (512, 256, (900, 901), tuple(range(8)), tuple(range(8, 16)))
    assert (P.cuts, P.js_levels, P.corrupt_seed, P.f_sweep) == ((2, 4, 6), (2, 3, 4, 5), 2026, (0.0, 0.5, 1.0))
    cells = S.cell_specs(P)
    assert len(cells) == 11 and all(len(c["seeds"]) == 8 and c["group_n"] == 256 for c in cells)             # 2048 samples each
    assert [c["cut"] for c in cells if c["kind"] == "crossed"] == [2, 2, 4, 4, 6, 6]
    assert {c["key"] for c in S.cell_specs(P, replicate=True, fs=[0.5])} == {"m2_m2_rep", "m3_m3_rep", "m4_f0.5_rep"}
    assert all(c["seeds"] == list(range(8, 16)) for c in S.cell_specs(P, replicate=True, fs=[0.5]))
    # clean bank = examples("val", 512); corrupted bank = plausible_wrong, released net, seed 2026, W1's rates
    seen = {}
    monkeypatch.setattr(S, "examples", lambda split, n: (seen.update(ex=(split, n)), np.zeros((n, 128), dtype=np.int64))[1])
    assert S.clean_windows(P, "cpu").shape == (512, 128) and seen["ex"] == ("val", 512)
    w = world
    env = make_env(w)
    monkeypatch.setattr(S.W, "corrupted_contexts", lambda *a, **k: seen.update(cc=(a, k)) or "ctx")
    assert S.build_corrupted_bank(env, P) == "ctx"
    (kind, tok, net, L0, pe, x, dev, seed, rates), _ = seen["cc"]
    assert kind == "plausible_wrong" and net is w.nets["released"] and x is w.x and seed == 2026
    assert list(rates) == list(CorruptionCfg(mode="mech", mean_rate=0.5, spread=0.5).rates(12))
    src = (ROOT / "scripts" / "mech_eval_w2m.py").read_text()
    assert 'CorruptionCfg(mode="mech", mean_rate=0.5, spread=0.5)' in src and "a.eval_seed" in src          # the mech eval's call


def test_real_val_windows_are_the_first_512_even_index_items():
    from ncmr.data import examples
    x = S.clean_windows(G.Protocol(), "cpu")
    assert x.shape == (512, 128) and np.array_equal(x.numpy(), examples("val", 512))


def test_resume_skips_cached_cells_and_reproduces_result(world, fake_scorers, tmp_path, monkeypatch):
    w = world
    out = str(tmp_path / "a")
    first = S.run(make_env(w), out, SMALL, 0, _quiet)
    n_files = sorted(os.listdir(os.path.join(out, "cells")))
    assert "m2to3_c2.json" in n_files and "bank_released.npy" in n_files and "rho_m3.npy" in n_files and "js_clean.npy" in n_files
    sampled, evaluated, other = [], [], []
    real = G.sample_codes
    monkeypatch.setattr(G, "sample_codes", lambda *a, **k: (sampled.append(1), real(*a, **k))[1])
    for fn in ("js_bank", "root_reliance", "kl_const_flag", "nll_const_flag"):          # the analyses come from their caches too
        monkeypatch.setattr(G, fn, (lambda f: lambda *a, **k: (other.append(f), None)[1])(fn))
    monkeypatch.setattr(S, "build_corrupted_bank", lambda *a, **k: other.append("corrupted_bank"))
    ev = make_env(w).evaluate
    second = S.run(make_env(w, lambda ids, groups: (evaluated.append(1), ev(ids, groups))[1]), out, SMALL, 0, _quiet)
    assert sampled == [] and evaluated == [] and other == []                              # every cell, bank and analysis came from the cache
    assert second == first
    assert json.loads((tmp_path / "a" / "result.json").read_text()) == json.loads(json.dumps(first))
    cell = json.loads((tmp_path / "a" / "cells" / "m2_m2.json").read_text())
    assert len(cell["sample_ids"]) == 6 and cell["groups"] == [0, 0, 1, 1, 2, 2] and cell["spec"]["seeds"] == [0, 1, 2]


def test_interrupted_run_leaves_a_partial_result_and_resumes_to_the_same_result(world, fake_scorers, tmp_path):
    w = world
    ev = make_env(w).evaluate
    count = {"n": 0}

    def flaky(ids, groups):
        count["n"] += 1
        if count["n"] == 4:
            raise RuntimeError("preempted")
        return ev(ids, groups)

    out = str(tmp_path / "r")
    with pytest.raises(RuntimeError):
        S.run(make_env(w, flaky), out, SMALL, 0, _quiet)
    part = json.loads((tmp_path / "r" / "result.json").read_text())
    assert part["complete"] is False and len(part["cells"]) == 3 and "ii" not in part
    redo = {"n": 0}
    final = S.run(make_env(w, lambda ids, groups: (redo.__setitem__("n", redo["n"] + 1), ev(ids, groups))[1]), out, SMALL, 0, _quiet)
    assert redo["n"] == 11 - 3                                                            # only the unfinished cells are run
    clean = S.run(make_env(w), str(tmp_path / "fresh"), SMALL, 0, _quiet)
    assert final == clean and final["complete"] is True


def test_a_rerun_with_another_protocol_or_checkpoint_is_refused(world, fake_scorers, tmp_path):
    w = world
    out = str(tmp_path / "m")
    S.run(make_env(w), out, SMALL, 0, _quiet)
    other = G.Protocol(n_clean=4, group_n=3, replay_seeds=(0, 1, 2), replicate_seeds=(8, 9, 10), n_boot=50, eval_bs=2)
    with pytest.raises(SystemExit):
        S.run(make_env(w), out, other, 0, _quiet)
    env = make_env(w)
    env.ckpt_sha256["m2"] = "different"
    with pytest.raises(SystemExit):
        S.run(env, out, SMALL, 0, _quiet)
    with pytest.raises(SystemExit):
        S.run(make_env(w), out, SMALL, 1, _quiet)
    with pytest.raises(SystemExit):                                                        # a protocol field that is in no cell's spec
        S.run(make_env(w), out, G.Protocol(n_clean=4, group_n=2, replay_seeds=(0, 1, 2), replicate_seeds=(8, 9, 10), n_boot=51, eval_bs=2), 0, _quiet)


def test_a_cached_cell_with_another_spec_is_refused(world, fake_scorers, tmp_path):
    w = world
    out = tmp_path / "s"
    S.run(make_env(w), str(out), SMALL, 0, _quiet)
    path = out / "cells" / "m2to3_c4.json"
    cell = json.loads(path.read_text())
    cell["spec"]["cut"] = 6
    path.write_text(json.dumps(cell))
    with pytest.raises(SystemExit):
        S.run(make_env(w), str(out), SMALL, 0, _quiet)


def _seq(vals):
    it = iter(vals)
    return lambda *a, **k: next(it)


@pytest.mark.parametrize("name,rule,expect,reading", [
    ("never", lambda *a, **k: False, [], "holds"),
    ("both", lambda *a, **k: True, [0.5, 1.0], "overturned"),
    # iv_rows asks the rule for f = 0.5 then f = 1 on seeds 0-7 (twice: before and after the replication), then for the replicated f's
    ("half_replicates", _seq([True, False, True, False, True]), [0.5], "overturned"),
    ("half_fails_to_replicate", _seq([True, False, True, False, False]), [0.5], "mixed"),
])
def test_replication_runs_if_and_only_if_the_overturn_rule_is_met(world, fake_scorers, tmp_path, monkeypatch, name, rule, expect, reading):
    w = world
    sampled = []
    real = G.sample_codes
    monkeypatch.setattr(G, "sample_codes", lambda fn, tok, L0, n, seed, device, **kw: (sampled.append(seed), real(fn, tok, L0, n, seed, device, **kw))[1])
    monkeypatch.setattr(G, "iv_rule", rule)
    res = S.run(make_env(w), str(tmp_path / name), SMALL, 0, _quiet)
    reps = sorted(k for k in res["cells"] if k.endswith("_rep"))
    want = (["m2_m2_rep", "m3_m3_rep"] + [f"m4_f{S.fkey(f)}_rep" for f in expect]) if expect else []
    assert reps == sorted(want)
    assert set(s for s in sampled if s < 900) == ({0, 1, 2, 8, 9, 10} if expect else {0, 1, 2})
    assert res["iv"]["reading"] == reading
    if expect:
        assert all(res["cells"][k]["spec"]["seeds"] == [8, 9, 10] for k in reps)
        assert set(res["iv"]["seeds_8_15"]["f"]) == {S.fkey(f) for f in expect}
    else:
        assert res["iv"]["seeds_8_15"] is None


# ------------------------------------------------------------------------------------------------ analysis_ii wiring, gate
def fake_cell(ll_large, ll_ts, entropy=4.0, dup=10.0):
    m = {"n": 8, "entropy": entropy, "len_tokens": 100.0, "dup_adj1_per_1k": dup, "dup_adj2_per_1k": 1.0}
    for sc, ll in (("gpt2-large", ll_large), ("TinyStories-33M", ll_ts)):
        m[f"genppl_{sc}"] = {"mean": float(np.mean(np.exp(ll))), "sd": 1.0, "se": 0.4, "per_group": list(np.exp(ll)),
                             "log_loss_per_group": list(ll)}
    return {"spec": {}, "metrics": m, "sample_ids": [], "groups": []}


def cut_cells(pi2_large, pi2_ts, G_=4):
    """Cells whose pi_2 is the given value under each scorer (L(M2,M2) = 3, L(M3,M3) = 1, Delta = 2)."""
    one = lambda v: np.full(G_, v)
    cells = {"m2_m2": fake_cell(one(3.0), one(3.0)), "m3_m3": fake_cell(one(1.0), one(1.0))}
    for c in (2, 4, 6):
        a, b = (1.0 + 2 * pi2_large, 1.0 + 2 * pi2_ts) if c == 2 else (2.0, 2.0)
        cells[f"m2to3_c{c}"] = fake_cell(one(a), one(b))
        cells[f"m3to2_c{c}"] = fake_cell(one(2.0), one(2.0))
    return cells


@pytest.mark.parametrize("pi_l,pi_t,passes", [(0.25, 0.1, True), (0.2499, 0.1, False), (0.6, 0.0, False), (0.6, 0.0001, True)])
def test_analysis_ii_gate_boundaries(pi_l, pi_t, passes):
    out = S.analysis_ii(G.Protocol(), cut_cells(pi_l, pi_t))
    g = out["direction1_gate"]
    assert abs(g["pi_2_gpt2-large"] - pi_l) < 1e-9 and abs(g["pi_2_TinyStories-33M"] - pi_t) < 1e-9
    assert g["passes"] is passes


def test_analysis_ii_needs_all_cells():
    cells = cut_cells(0.5, 0.5)
    del cells["m3to2_c6"]
    assert S.analysis_ii(G.Protocol(), cells) is None


def test_iv_rows_use_genppl_means_not_log_losses():
    cells = {"m2_m2": fake_cell(np.log([40.0, 44.0]), np.log([10.0, 14.0])), "m3_m3": fake_cell(np.log([20.0, 24.0]), np.log([5.0, 7.0]), 4.0, 10.0)}
    cells["m4_f0.5"] = fake_cell(np.log([24.0, 28.0]), np.log([6.0, 8.0]), 3.96, 11.0)
    row = S.iv_rows(cells, "", [0.5])["f"]["0.5"]
    assert abs(row["R"]["gpt2-large"] - (42.0 - 26.0) / (42.0 - 22.0)) < 1e-9 and abs(row["R"]["TinyStories-33M"] - (12.0 - 7.0) / (12.0 - 6.0)) < 1e-9
    assert row["rule_met"] is True                                                         # 0.8 / 0.833, entropy 3.96, dup 11
    cells["m4_f0.5"] = fake_cell(np.log([24.0, 28.0]), np.log([6.0, 8.0]), 3.94, 11.0)
    assert S.iv_rows(cells, "", [0.5])["f"]["0.5"]["rule_met"] is False


def _write_result(path, cells, complete=True):
    res = {"complete": complete, "cells": {k: {"spec": {}, **c["metrics"], "n_groups": 4} for k, c in cells.items()}}
    Path(path).write_text(json.dumps(res))


@pytest.mark.parametrize("pi_l,pi_t,code", [(0.25, 0.1, 0), (0.2499, 0.1, 2), (0.9, 0.0, 2), (0.9, 0.01, 0)])
def test_gate_script_exit_codes(tmp_path, pi_l, pi_t, code, capsys):
    p = tmp_path / "result.json"
    _write_result(p, cut_cells(pi_l, pi_t))
    assert GATE.main(["--result", str(p)]) == code
    assert ("PASS" if code == 0 else "FAIL") in capsys.readouterr().out


def test_gate_script_exit_3_when_missing_or_incomplete(tmp_path):
    assert GATE.main(["--result", str(tmp_path / "nope.json")]) == 3
    p = tmp_path / "r.json"
    p.write_text("{not json")
    assert GATE.main(["--result", str(p)]) == 3
    _write_result(p, cut_cells(0.9, 0.9), complete=False)
    assert GATE.main(["--result", str(p)]) == 3                                            # unfinished run
    cells = cut_cells(0.9, 0.9)
    del cells["m3to2_c2"]
    _write_result(p, cells)
    assert GATE.main(["--result", str(p)]) == 3
    cells = cut_cells(0.9, 0.9)
    cells["m2_m2"] = fake_cell(np.full(4, 1.0), np.full(4, 1.0))
    cells["m3_m3"] = fake_cell(np.full(4, 1.0), np.full(4, 1.0))                            # Delta = 0: no shares
    _write_result(p, cells)
    assert GATE.main(["--result", str(p)]) == 3
    p.write_text("[]")
    assert GATE.main(["--result", str(p)]) == 3


def test_gate_script_command_line_exit_codes(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    run = lambda path: subprocess.run([sys.executable, str(ROOT / "scripts" / "w2mg_gate.py"), "--result", str(path)], env=env,
                                      capture_output=True, text=True, timeout=300).returncode
    p = tmp_path / "result.json"
    _write_result(p, cut_cells(0.5, 0.5))
    assert run(p) == 0
    _write_result(p, cut_cells(0.1, 0.5))
    assert run(p) == 2
    assert run(tmp_path / "missing.json") == 3


# ------------------------------------------------------------------------------------------------ load_env wiring
def test_load_env_reads_the_four_checkpoints_and_m4_flag(tmp_path, monkeypatch):
    from ncmr.ncm_ref_loader import load_ncm
    B = load_ncm(str(ROOT / "artifacts" / "tinystories"), device="cpu")
    sd = {k: v.clone() for k, v in B.net.state_dict().items()}
    head = next(k for k in sd if "lm_heads.0" in k and sd[k].dtype.is_floating_point)
    w = torch.arange(384, dtype=torch.float32) * 0.001
    del B                                                  # the test holds one copy of the weights, load_env five more
    cks = {}
    for i, (name, d) in enumerate(S.ARM_DIRS.items()):
        s = {k: v for k, v in sd.items()}
        s[head] = sd[head] + 0.01 * (i + 1)                                                 # each arm's weights are distinguishable
        (tmp_path / f"{d}_s0").mkdir()
        (tmp_path / f"{d}_s0" / "final.pt").write_bytes(f"placeholder {name}".encode())    # hashed; the weights come from `cks`
        cks[f"{d}_s0"] = {"model": s, **({"flag": {"w": w}} if name == "m4" else {})}
    real_load = torch.load
    monkeypatch.setattr(S.torch, "load", lambda p, **kw: cks[Path(p).parent.name] if Path(p).name == "final.pt" else real_load(p, **kw))
    monkeypatch.setattr(S, "examples", lambda split, n: np.zeros((n, 128), dtype=np.int64))
    env = S.load_env(str(tmp_path), 0, "cpu", G.Protocol(n_clean=2))
    assert set(env.nets) == {"released", "m0", "m2", "m3", "m4"} and env.clean_x.shape == (2, 128)
    for i, name in enumerate(S.ARM_DIRS):
        assert torch.allclose(env.nets[name].state_dict()[head], sd[head] + 0.01 * (i + 1))
    assert torch.equal(env.nets["released"].state_dict()[head], sd[head])
    assert torch.allclose(env.flag.w, w) and set(env.ckpt_sha256) == set(S.ARM_DIRS)
    assert env.runs["m3"] == "L_ts_m3_uniform_wrong_s0/final.pt" and len(set(env.ckpt_sha256.values())) == 4
    assert not env.nets["m2"].training and env.nets["m2"] is not env.nets["released"]
    assert env.member("m4", 0.5).flag is env.flag and env.member("m2").flag is None
    (tmp_path / "L_ts_m0_uniform_s0" / "final.pt").unlink()
    with pytest.raises(SystemExit):
        S.load_env(str(tmp_path), 0, "cpu", G.Protocol(n_clean=2))
