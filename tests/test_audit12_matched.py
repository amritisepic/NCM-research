"""Audit 12: execute the real mech training loop and inspect matched slots.

These are passing audit checks, not defect regressions. Shared CPU setup is in
conftest.py; the tiny objects use the upstream tokenizer and generator classes.
The raw-result and sample integrity checks are hash-pinned separately.
"""
import pytest
import torch

from ncmr.train import fp32_eval
from ncmr.corruption import CorruptionCfg, build_hierarchy
from ncmr.models import build_ncm_generator
from ncmr.w2m import MaskPlan, MechArm
from w2m_support import Harness, make_arm, tiny_case


def traced_run(tmp_path, arm, seed):
    """Instrument the production AST loop without replacing its training work."""
    h = Harness(tmp_path, arm=arm, seed=seed, B=16, accum=4, at=(2, 4))
    batches, plans, micros, proposal_rng = [], [], [], []
    original_batch = h.ns["stream"].batch
    original_plan = h.mech.draw_plan
    original_record = h.mech.record
    original_proposals = h.mech.proposals

    def batch(rng, n):
        x = original_batch(rng, n)
        batches.append(x.copy())
        return x

    def draw_plan(n):
        plan = original_plan(n)
        plans.append((plan.clean.clone(), [m.clone() for m in plan.masks]))
        return plan

    def record(used, targets, plan):
        micros.append((targets.clone(), used.clone(), plan.clean.clone(),
                       [m.clone() for m in plan.masks], plan.row0))
        return original_record(used, targets, plan)

    def proposals(canon, plan):
        before = torch.get_rng_state().clone()
        out = original_proposals(canon, plan)
        proposal_rng.append((before, torch.get_rng_state().clone()))
        return out

    h.ns["stream"].batch = batch
    h.mech.draw_plan, h.mech.record, h.mech.proposals = draw_plan, record, proposals
    h.run(4)
    return h, batches, plans, micros, proposal_rng, h.state()["torch_rng"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_m2_m3_actual_loop_batches_clean_rows_and_slots_match(tmp_path, seed):
    a, b = [traced_run(tmp_path / arm, arm, seed) for arm in ("m2", "m3")]
    assert len(a[1]) == len(b[1]) == 4
    for x, y in zip(a[1], b[1]):
        assert (x == y).all()
    for (clean_a, masks_a), (clean_b, masks_b) in zip(a[2], b[2]):
        assert torch.equal(clean_a, clean_b)
        assert int(clean_a.sum()) == 4
        for ma, mb in zip(masks_a, masks_b):
            assert torch.equal(ma, mb)
            assert not ma[clean_a].any()
    assert len(a[3]) == len(b[3]) == 16  # four micro-batches per step
    for ma, mb in zip(a[3], b[3]):
        ca, ua, cla, mka, row_a = ma
        cb, ub, clb, mkb, row_b = mb
        assert row_a == row_b
        assert torch.equal(ca, cb)
        assert torch.equal(cla, clb)
        assert all(torch.equal(x, y) for x, y in zip(mka, mkb))
        mask = torch.cat(mka, 1)
        for canonical, used in ((ca, ua), (cb, ub)):
            assert torch.equal(used != canonical, mask)
            assert torch.equal(used[~mask], canonical[~mask])
    for run in (a, b):
        assert all(torch.equal(before, after) for before, after in run[4])
        assert run[0].sample_calls == [2, 4]
    assert torch.equal(a[5], b[5])  # training dropout stream has the same draws
    assert a[0].data_rng.bit_generator.state == b[0].data_rng.bit_generator.state
    assert torch.equal(a[0].mech.mask_gen.get_state(), b[0].mech.mask_gen.get_state())
    assert a[0].mech.realized() == b[0].mech.realized()
    assert a[0].losses()[-1]["loss"] != b[0].losses()[-1]["loss"]


def test_frozen_wrong_is_conditional_on_canonical_context_and_excludes_canonical():
    tok, net = tiny_case()
    m = make_arm("m2", tok, net, seed=1)
    gen = torch.Generator().manual_seed(123)
    x = torch.randint(0, 32, (8, 128), generator=gen)
    canon, plan = m.canonical(x), m.draw_plan(8)
    seen = {}
    original_logits, original_fine = m._logits, m.draw_fine

    def logits(model, fwd, context):
        seen["context"] = context.clone()
        assert model is m.frozen and fwd is m.frozen
        assert not any(mod.training for mod in model.modules())
        return original_logits(model, fwd, context)

    def fine(probs):
        seen["probabilities"] = probs.clone()
        return original_fine(probs)

    m._logits, m.draw_fine = logits, fine
    with torch.no_grad(), fp32_eval(m.frozen, "cpu"):
        expected = original_logits(m.frozen, m.frozen, canon)
    expected.scatter_(-1, canon[:, m.l0:, None], float("-inf"))
    expected = expected.softmax(-1)  # registered policy_temp = 1
    before = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        repl = m.proposals(canon, plan)
        assert torch.is_autocast_enabled("cpu")
    assert not torch.is_autocast_enabled("cpu")
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(seen["context"], canon)
    assert seen["probabilities"].dtype == torch.float32
    torch.testing.assert_close(seen["probabilities"], expected, rtol=0, atol=0)
    assert not bool((repl[:, :m.l0] == canon[:, :m.l0]).any())
    assert not bool((repl[:, m.l0:] == canon[:, m.l0:]).any())


def test_equal_realized_aggregates_do_not_prove_identical_slots():
    tok, net = tiny_case()
    a, b = [make_arm(arm, tok, net, seed=1) for arm in ("m2", "m3")]
    plan = a.draw_plan(16)
    # Permute only the corrupted rows: every per-level count stays the same,
    # while which example receives which masked positions changes.
    order = torch.arange(16)
    unclean = (~plan.clean).nonzero().flatten()
    order[unclean] = unclean.roll(1)
    other = MaskPlan(plan.clean.clone(), [mask[order] for mask in plan.masks])
    assert any(not torch.equal(x, y) for x, y in zip(plan.masks, other.masks))
    canon = torch.zeros(16, a.S, dtype=torch.long)
    for arm, p in ((a, plan), (b, other)):
        used = torch.where(torch.cat(p.masks, 1), torch.ones_like(canon), canon)
        arm.record(used, canon, p)
    assert a.realized() == b.realized()


def test_uniform_wrong_maps_uniform_offsets_to_exactly_the_noncanonical_support(monkeypatch):
    tok, net = tiny_case()
    m = make_arm("m3", tok, net)
    # Enumerate each possible offset once instead of relying on a frequency test.
    B, V, canonical_code = 31, 32, 7
    canon = torch.full((B, m.S), canonical_code, dtype=torch.long)
    plan = m.draw_plan(B)

    def enumerate_offsets(low, high, size, **kwargs):
        assert low == 1 and high == V and size[0] == B
        return torch.arange(1, V).view(B, 1).expand(size)

    monkeypatch.setattr(torch, "randint", enumerate_offsets)
    result = m.proposals(canon, plan)
    support = torch.tensor([i for i in range(V) if i != canonical_code])
    for block in result.split(m.levels, 1)[:-1]:
        assert torch.equal(block.sort(0).values, support[:, None].expand_as(block))
    assert torch.equal(result[:, -m.levels[-1]:], canon[:, -m.levels[-1]:])


def test_wrong_replacement_and_shared_slots_on_hash_pinned_released_assets(released_sudoku):
    bundle, puzzles = released_sudoku
    tok = bundle.vqvae
    # The mech path is a text-only path predicting levels >= 1. Use that head
    # convention with the hash-checked released tokenizer and puzzle inputs.
    net = build_ncm_generator(tok, n_layer=1, n_head=2, n_embd=32, dropout=0.1)
    x = puzzles[:4, 1:-1].contiguous()
    prefix = lambda b: tok.encoder.wte(puzzles[:b, :1])
    V = int(tok.vq.vqvae_vocab_sizes[0])
    rates = CorruptionCfg(mode="mech").rates(len(tok.vq.levels))
    arms = [MechArm(arm, tok, net, torch.ones(V) / V, prefix, "cpu", 1, rates,
                    frozen=MechArm.frozen_copy(net) if arm == "m2" else None)
            for arm in ("m2", "m3")]
    plans = [m.draw_plan(len(x)) for m in arms]
    assert torch.equal(plans[0].clean, plans[1].clean)
    assert all(torch.equal(a, b) for a, b in zip(plans[0].masks, plans[1].masks))
    canonical = [m.canonical(x) for m in arms]
    assert torch.equal(*canonical)
    for m, plan, canon in zip(arms, plans, canonical):
        before = torch.get_rng_state().clone()
        repl = m.proposals(canon, plan)
        assert torch.equal(before, torch.get_rng_state())
        hierarchy = build_hierarchy(tok, x, CorruptionCfg(mode="mech"),
                                    policy_codes=repl, masks=plan.masks, canonical=canon)
        assert torch.equal(hierarchy["used"] != canon, torch.cat(plan.masks, 1))
        assert torch.equal(hierarchy["targets"], canon)
