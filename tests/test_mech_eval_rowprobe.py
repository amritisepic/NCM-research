"""W2Ma row-level probe of scripts/mech_eval_w2m.py (docs/preregistration.md, "W2Ma"): contexts with 50% clean windows,
paired plausible-wrong / uniform-wrong masks, level-k pooling, held-out AUC per kind and level, too-weak rule, and the
pooled probe stored as "not read". CPU, tiny models."""
import importlib.util
import os

import pytest
import torch

from ncmr import w2m as W
from ncmr.corruption import CorruptionCfg
from w2m_support import prefix_fn, tiny_case

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("mech_eval_w2m", os.path.join(ROOT, "scripts", "mech_eval_w2m.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]


def _setup(n, seed=3):
    tok, net = tiny_case()
    net.eval()
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(2, 32, (n, 128), generator=g)
    V = int(tok.vq.vqvae_vocab_sizes[0])
    rates = CorruptionCfg(mode="mech", mean_rate=0.5, spread=0.5).rates(len(LEVELS))
    return tok, net, x, torch.ones(V) / V, prefix_fn(tok), rates


@pytest.fixture(scope="module")
def case():
    tok, net, x, L0, pe, rates = _setup(24)
    return tok, net, x, L0, pe, rates, M.row_contexts_pair(tok, net, L0, pe, x, "cpu", 7, rates)


@pytest.mark.parametrize("n", [16, 21])
def test_exactly_round_half_of_the_rows_are_clean(n):
    tok, net, x, L0, pe, rates = _setup(n)
    c = M.row_contexts("uniform_wrong", tok, net, L0, pe, x, "cpu", 5, rates)
    assert int(c["clean"].sum()) == round(0.5 * n)
    assert torch.equal(c["corrupted"], ~c["clean"])


def test_clean_rows_are_canonical_with_zero_flags_and_corrupted_rows_have_masked_slots(case):
    tok, _, x, _, _, _, pair = case
    for kind, c in pair.items():
        canon_in = tok.vq.idx_to_ncm_input(c["canonical"])
        cl, co = c["clean"], c["corrupted"]
        assert torch.allclose(c["inputs"][cl], canon_in[cl], atol=1e-5), kind
        assert float(c["flags"][cl].abs().sum()) == 0.0
        n_masked = sum(m.sum(1) for m in c["masks"])
        assert bool((n_masked[co] >= 1).all()) and bool((n_masked[cl] == 0).all())
        assert bool((c["flags"][co] > 0).any(1).all())                          # every corrupted row carries flags
        # the corrupted rows' inputs really differ from the canonical ones
        assert bool((~torch.isclose(c["inputs"][co], canon_in[co], atol=1e-5)).flatten(1).any(1).all()), kind


def test_plausible_and_uniform_contexts_share_clean_rows_and_masks_but_not_replacements(case):
    _, _, _, _, _, _, pair = case
    pw, uw = pair["plausible_wrong"], pair["uniform_wrong"]
    assert torch.equal(pw["clean"], uw["clean"]) and 0 < int(pw["clean"].sum()) < len(pw["clean"])
    assert all(torch.equal(a, b) for a, b in zip(pw["masks"], uw["masks"]))
    assert torch.equal(pw["canonical"], uw["canonical"]) and torch.equal(pw["flags"], uw["flags"])
    assert not torch.equal(pw["inputs"], uw["inputs"])                          # different replacement codes


def test_a_corrupted_row_without_a_masked_slot_is_refused():
    tok, net, x, L0, pe, _ = _setup(16)
    with pytest.raises(AssertionError, match="without a masked slot"):
        M.row_contexts("uniform_wrong", tok, net, L0, pe, x, "cpu", 5, [0.0] * len(LEVELS))


def test_level_span_is_the_target_layout_slice():
    lo = 0
    for k in range(1, len(LEVELS)):                                             # independent running offset
        assert M.level_span(LEVELS, k) == (lo, lo + LEVELS[k])
        lo += LEVELS[k]
    assert lo == sum(LEVELS) - LEVELS[0]


def test_pool_level_takes_exactly_that_levels_positions():
    n, d = 5, 3
    hid = torch.zeros(n, sum(LEVELS[1:]), d)
    lo = 0
    for k in range(1, len(LEVELS)):                                             # level k's positions hold the value k
        hid[:, lo:lo + LEVELS[k]] = float(k)
        lo += LEVELS[k]
    for k in range(1, len(LEVELS)):
        p = M.pool_level(hid, LEVELS, k)
        assert p.shape == (n, d) and bool((p == float(k)).all()), (k, p[0])
    g = torch.Generator().manual_seed(0)
    h = torch.randn(n, 640, d, generator=g)
    assert torch.allclose(M.pool_level(h, LEVELS, 3), h[:, 24:48].mean(1))     # level 3 = positions 8 + 16 .. 8 + 16 + 24


def test_row_probe_reads_the_level_it_is_asked_about():
    n, d, ntr = 96, 4, 64
    g = torch.Generator().manual_seed(1)
    y = torch.zeros(n, dtype=torch.bool)
    y[::2] = True
    hid = torch.randn(n, 640, d, generator=g)
    lo, hi = M.level_span(LEVELS, 3)
    hid[:, lo:hi, 0] += 3.0 * y[:, None].float()                               # only level 3 carries the label
    aucs = M.row_probe_aucs(hid, y, LEVELS, M.JS, ntr)
    assert sorted(aucs) == [str(k) for k in M.JS]
    assert aucs["3"] > 0.99
    assert max(aucs[str(k)] for k in M.JS if k != 3) < 0.9


@pytest.mark.parametrize("aucs,want", [({"1": 0.7, "2": 0.6, "3": 0.74999}, True), ({"1": 0.7, "2": 0.75}, False),
                                       ({"1": 0.5, "2": 0.99}, False), ({"1": None, "2": 0.4}, True), ({"1": None, "2": 0.9}, False)])
def test_too_weak_to_read_only_when_every_uniform_wrong_auc_is_below_the_floor(aucs, want):
    assert M.too_weak_to_read(aucs) is want


def test_evaluate_model_stores_pooled_probe_as_not_read_and_the_row_level_probe(case):
    tok, net, x, L0, pe, rates, pair = case
    n, ntr = x.shape[0], 16
    canon = pair["uniform_wrong"]["canonical"]
    perturb = W.make_perturbations(canon, LEVELS, [int(v) for v in tok.vq.vqvae_vocab_sizes], M.JS, 1)
    ctxs = {"plausible_wrong": W.corrupted_contexts("plausible_wrong", tok, net, L0, pe, x, "cpu", 9, rates),
            "uniform_wrong": W.corrupted_contexts("uniform_wrong", tok, net, L0, pe, x, "cpu", 10, rates)}
    out = M.evaluate_model(net, None, tok, pe, canon, perturb, ctxs, ntr, "cpu", 0, rctxs=pair)
    assert "probe_auc" not in out
    assert out["probe_pooled"]["status"] == "not read (W2Ma: confounded by level identity, R043)"
    assert {"plausible_wrong", "uniform_wrong"} <= set(out["probe_pooled"])      # the existing entries are kept
    rl = out["probe_row_level"]
    assert isinstance(rl["too_weak_to_read"], bool)
    for kind in ("plausible_wrong", "uniform_wrong"):
        assert sorted(rl[kind]) == [str(k) for k in M.JS]
        # recompute independently: zero-flag hidden states, literal level-1 / level-2 slices, held-out windows after ntr
        _, hid = W.forward_logits(net, pe, pair[kind]["inputs"], "cpu", hidden=True, keep_logits=False)
        y = pair[kind]["corrupted"]
        for k, (a, b) in {1: (0, 8), 2: (8, 24)}.items():
            f = hid[:, a:b].mean(1)
            want = W.fit_probe_auc(f[:ntr, None], y[:ntr, None], f[ntr:, None], y[ntr:, None])
            assert rl[kind][str(k)] == pytest.approx(want, abs=1e-9)
    assert rl["too_weak_to_read"] == all(v is None or v < 0.75 for v in rl["uniform_wrong"].values())


@pytest.mark.parametrize("kind", ["plausible_wrong", "uniform_wrong"])
def test_row_contexts_at_clean_frac_zero_replicate_w2m_corrupted_contexts(kind):
    """The script's replica of `W.corrupted_contexts` is faithful: same seed, no clean rows -> identical contexts."""
    tok, net, x, L0, pe, rates = _setup(12)
    a = M.row_contexts(kind, tok, net, L0, pe, x, "cpu", 4, rates, clean_frac=0.0)
    b = W.corrupted_contexts(kind, tok, net, L0, pe, x, "cpu", 4, rates)
    assert not bool(a["clean"].any())
    assert torch.equal(a["inputs"], b["inputs"]) and torch.equal(a["flags"], b["flags"]) and torch.equal(a["targets"], b["targets"])
    assert all(torch.equal(p, q) for p, q in zip(a["masks"], b["masks"]))
