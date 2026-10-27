"""W2M code gate (docs/preregistration.md, "W2M"): mask stream, clean rows, wrong-by-construction replacements, flags,
zero-initialised flag vector, shuffled flags, used-parent policy, M7's rates. CPU, tiny models."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ncmr.corruption import CorruptionCfg, build_hierarchy
from ncmr import w2m
from ncmr.w2m import ARMS, A3_DISAGREEMENT, FlagVector, MaskPlan, MechArm, m7_level_rates, mask_flags, shuffle_within_level
from w2m_support import Harness, make_arm, prefix_fn, tiny_case, script_nodes, run_nodes

ROOT = Path(__file__).resolve().parents[1]
W1_ARMS = ["m0", "m1", "m2", "m3", "m4", "m5", "m6"]    # arms that corrupt with W1's rates


@pytest.fixture(scope="module")
def case():
    return tiny_case()


def steps_of(arm, case, n_steps=3, B=16, seed=0, **kw):
    """Plans of n_steps optimizer steps, with each arm's own proposal draws in between."""
    tok, net = case
    m = make_arm(arm, tok, net, seed=seed, **kw)
    g = torch.Generator().manual_seed(5)
    plans = []
    for _ in range(n_steps):
        x = torch.randint(0, 32, (B, 128), generator=g)
        plan = m.draw_plan(B)
        m.proposals(m.canonical(x), plan)
        plans.append(plan)
    return plans, m


# ---------------------------------------------------------------------------------------------- mask stream
def test_mask_stream_is_identical_across_arms_with_w1_rates(case):
    ref, _ = steps_of("m0", case)
    assert any(int(m.sum()) > 0 for p in ref for m in p.masks), "degenerate fixture"
    for arm in W1_ARMS[1:]:
        plans, _ = steps_of(arm, case)
        for step, (p, q) in enumerate(zip(ref, plans)):
            assert torch.equal(p.clean, q.clean), (arm, step)
            for k, (a, b) in enumerate(zip(p.masks, q.masks)):
                assert torch.equal(a, b), (arm, step, k)


def test_mask_stream_is_independent_of_proposal_draws_and_differs_by_step_and_seed(case):
    tok, net = case
    plain = make_arm("m0", tok, net)
    a = [plain.draw_plan(16) for _ in range(3)]            # no proposal draws at all
    plans, _ = steps_of("m1", case)                        # policy proposals in between
    assert all(torch.equal(x.masks[0], y.masks[0]) and torch.equal(x.masks[5], y.masks[5]) for x, y in zip(a, plans))
    assert not torch.equal(a[0].masks[5], a[1].masks[5])   # a stream, not a constant
    other, _ = steps_of("m0", case, seed=1)
    assert not all(torch.equal(x.masks[5], y.masks[5]) for x, y in zip(a, other))


def test_mask_plan_slices_per_micro_batch(case):
    tok, net = case
    plan = make_arm("m0", tok, net).draw_plan(16)
    parts = [plan.rows(0, 4), plan.rows(4, 10), plan.rows(10, 16)]
    assert torch.equal(torch.cat([p.clean for p in parts]), plan.clean)
    for k in range(len(plan.masks)):
        assert torch.equal(torch.cat([p.masks[k] for p in parts]), plan.masks[k])


def test_m7_shares_the_clean_rows_but_not_the_slots(case):
    tok, net = case
    K = len(tok.vq.levels)
    r7 = m7_level_rates(A3_DISAGREEMENT, [32] * K)
    base, _ = steps_of("m0", case)
    m7, _ = steps_of("m7", case, rates=r7)
    assert all(torch.equal(a.clean, b.clean) for a, b in zip(base, m7))
    assert any(not torch.equal(a.masks[k], b.masks[k]) for a, b in zip(base, m7) for k in range(1, K - 1))


# ---------------------------------------------------------------------------------------------- clean rows
@pytest.mark.parametrize("B", [8, 16, 256])
def test_exactly_25_percent_of_rows_are_clean_and_have_no_masked_slot(case, B):
    tok, net = case
    m = make_arm("m0", tok, net)
    chosen = []
    for _ in range(3):
        plan = m.draw_plan(B)
        assert int(plan.clean.sum()) == B // 4
        for mk in plan.masks:
            assert not bool(mk[plan.clean].any())
        assert all(bool(mk[~plan.clean].any()) for mk in plan.masks[:-1])
        assert not bool(plan.masks[-1].any())              # the finest level is never corrupted
        chosen.append(plan.clean.clone())
    assert not all(torch.equal(chosen[0], c) for c in chosen[1:])   # re-chosen every batch


def test_corruption_rates_on_unclean_rows_are_w1s(case):
    tok, net = case
    K = len(tok.vq.levels)
    m = make_arm("m0", tok, net)
    plan = m.draw_plan(6000)
    want = CorruptionCfg(mode="mech").rates(K)
    assert want[0] == pytest.approx(0.75) and want[10] == pytest.approx(0.25) and want[11] == 0
    for k in range(K - 1):
        got = float(plan.masks[k][~plan.clean].float().mean())
        assert got == pytest.approx(want[k], abs=0.03), k


# ---------------------------------------------------------------------------------------------- wrong replacements
def peaked_logits(m):
    """Replace the policy forward by one that puts (almost) all mass on the canonical code: a proposer without the
    exclusion would return the canonical code."""
    def _logits(model, fwd, ctx):
        return 60.0 * F.one_hot(ctx[:, m.l0:], 32).float()
    m._logits = _logits
    return m


def apply_arm(m, tok, x, B):
    plan = m.draw_plan(B)
    canon = m.canonical(x)
    repl = m.proposals(canon, plan)
    h = build_hierarchy(tok, x, CorruptionCfg(mode="mech"), policy_codes=repl, masks=plan.masks, canonical=canon)
    return plan, canon, repl, h


@pytest.mark.parametrize("arm", ["m2", "m3", "m4", "m5"])
def test_replacements_are_always_wrong(case, arm):
    tok, net = case
    m = peaked_logits(make_arm(arm, tok, net))
    # level-0 unigram concentrated on the (shared) canonical level-0 code: only the exclusion can make it wrong
    x1 = torch.randint(0, 32, (1, 128))
    x = x1.repeat(32, 1)
    c0 = int(m.canonical(x)[0, 0])
    m.L0 = torch.full((32,), 0.001)
    m.L0[c0] = 1.0
    n_masked = 0
    for _ in range(4):
        plan, canon, repl, h = apply_arm(m, tok, x, 32)
        masked = torch.cat(plan.masks, 1)
        n_masked += int(masked.sum())
        assert torch.equal(h["canonical"], canon)
        assert bool((h["used"][masked] != canon[masked]).all()), arm
        assert bool((repl[masked] != canon[masked]).all())
        assert torch.equal(h["used"][~masked], canon[~masked])
        assert torch.equal(h["targets"], canon)             # canonical labels
        assert float(h["corrupted_frac"]) > 0
    assert n_masked > 500


def test_control_policy_proposals_are_not_excluded(case):
    """Sanity of the fixture above: the same peaked policy WITHOUT the exclusion returns the canonical code (M1)."""
    tok, net = case
    m = peaked_logits(make_arm("m1", tok, net))
    x = torch.randint(0, 32, (32, 128))
    plan, canon, repl, h = apply_arm(m, tok, x, 32)
    masked = torch.cat(plan.masks, 1)
    assert float((h["used"] == canon)[:, 1:][masked[:, 1:]].float().mean()) > 0.95   # levels >= 1 follow the peaked policy


def test_realized_disagreement_of_wrong_arms_equals_the_mask_rate(case):
    tok, net = case
    m = make_arm("m3", tok, net)
    x = torch.randint(0, 32, (400, 128))
    plan, canon, repl, h = apply_arm(m, tok, x, 400)
    for k, lk in enumerate(m.levels[:-1]):
        masked = float(plan.masks[k].float().mean())
        assert h["corrupted_frac_per_level"][k] == pytest.approx(masked, abs=1e-9)


# ---------------------------------------------------------------------------------------------- M0 / M7 uniform
def test_m7_rates_follow_d_over_one_minus_one_over_v():
    r = m7_level_rates(A3_DISAGREEMENT, [4096] * 12)
    assert r[0] == pytest.approx(0.735 / (1 - 1 / 4096)) and r[1] == pytest.approx(0.691 / (1 - 1 / 4096))
    assert r[11] == 0.0 and all(0 <= v <= 1 for v in r)
    assert r[0] > 0.735                                     # strictly above d: a uniform draw can hit the canonical code
    assert m7_level_rates([0.5], [2]) == (1.0,)


def test_m7_realizes_a3s_disagreement_on_the_corrupted_rows(case):
    tok, net = case
    K = len(tok.vq.levels)
    m = make_arm("m7", tok, net, rates=m7_level_rates(A3_DISAGREEMENT, [32] * K))
    x = torch.randint(0, 32, (6000, 128))
    plan, canon, repl, h = apply_arm(m, tok, x, 6000)
    dis = [d.float() for d in torch.split(h["used"] != h["targets"], m.levels, dim=1)]
    for k in range(K):
        got = float(dis[k][~plan.clean].mean())
        assert got == pytest.approx(A3_DISAGREEMENT[k], abs=0.03), k


def test_a3_disagreement_constants_match_the_logged_trajectory():
    rows = [json.loads(l) for l in open(ROOT / "results/colab/w1_L_ts_onpol/L_ts_onpol_s0__metrics.jsonl")]
    logged = [r["corrupted_frac_per_level"] for r in rows if "corrupted_frac_per_level" in r and r["step"] <= 600]
    mean = np.mean(logged, 0)
    assert np.abs(mean - np.array(A3_DISAGREEMENT)).max() < 1.5e-3


# ---------------------------------------------------------------------------------------------- M6 / M1
def test_m6_conditions_on_the_used_level0_and_m1_on_the_canonical(case):
    tok, net = case
    for arm in ("m1", "m6"):
        m = make_arm(arm, tok, net)
        seen = {}
        orig = m._logits

        def spy(model, fwd, ctx, orig=orig):
            seen["ctx"] = ctx.clone()
            return orig(model, fwd, ctx)

        m._logits = spy
        x = torch.randint(0, 32, (64, 128))
        plan = m.draw_plan(64)
        canon = m.canonical(x)
        g2 = torch.Generator().manual_seed(0)
        g2.set_state(m.repl_gen.get_state())
        lv0 = torch.multinomial(m.L0, 64, replacement=True, generator=g2).view(64, 1)   # the proposal the arm will draw
        repl = m.proposals(canon, plan)
        assert torch.equal(repl[:, :1], lv0)
        used0 = torch.where(plan.masks[0], lv0, canon[:, :1])
        assert bool((used0 != canon[:, :1]).any()), "fixture: some level-0 code must change"
        assert torch.equal(seen["ctx"][:, 1:], canon[:, 1:])               # clean levels 1..
        if arm == "m6":
            assert torch.equal(seen["ctx"][:, :1], used0)
            assert not torch.equal(seen["ctx"], canon)
        else:
            assert torch.equal(seen["ctx"], canon)


def test_m1_proposals_equal_the_wave1_policy(case):
    """M1 is A3's policy: same samples as scripts/train_gen.py::policy_codes from the same generator state."""
    tok, net = case
    m = make_arm("m1", tok, net)
    funcs, _, _ = script_nodes("cur")
    ns = dict(torch=torch, fnet=net, net=net, tok=tok, device="cpu", L0=m.L0, l0=1, policy_passes=1, policy_temp=1.0,
              prefix_emb=prefix_fn(tok), fp32_eval=w2m.fp32_eval)
    run_nodes([funcs["policy_codes"]], ns)
    x = torch.randint(0, 32, (12, 128))
    g = torch.Generator().manual_seed(0)
    g.set_state(m.repl_gen.get_state())
    expected = ns["policy_codes"](x, g)
    got = m.proposals(m.canonical(x), m.draw_plan(12))
    assert torch.equal(got, expected)


def test_policy_and_frozen_forwards_are_fp32_eval_and_leave_modes_and_global_rng_alone(case):
    tok, net = case
    net.train()
    before = torch.get_rng_state().clone()
    modes = {id(mod): mod.training for mod in net.modules()}
    for arm in ("m1", "m2", "m6"):
        m = make_arm(arm, tok, net)
        x = torch.randint(0, 32, (8, 128), generator=torch.Generator().manual_seed(1))
        with torch.autocast("cpu", dtype=torch.bfloat16):              # as the trainer's region would be
            m.proposals(m.canonical(x), m.draw_plan(8))
            assert torch.is_autocast_enabled("cpu")                     # not leaked off either
    assert {id(mod): mod.training for mod in net.modules()} == modes
    assert torch.equal(before, torch.get_rng_state())                   # dropout's stream is untouched by proposals
    assert all(not p.requires_grad for p in MechArm.frozen_copy(net).parameters())
    assert not MechArm.frozen_copy(net).training


def test_frozen_copy_is_a_snapshot_and_proposals_use_it(case):
    tok, net = case
    frozen = MechArm.frozen_copy(net)
    m = make_arm("m2", tok, net, frozen=frozen)
    snap = {k: v.clone() for k, v in frozen.state_dict().items()}
    live = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)                                                 # "training" moves the live net only
    assert all(torch.equal(snap[k], v) for k, v in frozen.state_dict().items())
    seen = []
    orig = m._logits
    m._logits = lambda model, fwd, ctx: (seen.append(model), orig(model, fwd, ctx))[1]
    m.proposals(m.canonical(torch.randint(0, 32, (4, 128))), m.draw_plan(4))
    assert seen and seen[0] is frozen
    net.load_state_dict(live)


# ---------------------------------------------------------------------------------------------- flags
def linear_support(lj, L):
    """For each source code i of a level of length lj, the set of full-length positions t it contributes to under
    F.interpolate(mode='linear', align_corners=False) -- computed by hand."""
    sup = [set() for _ in range(lj)]
    for t in range(L):
        src = max((t + 0.5) * lj / L - 0.5, 0.0)
        i0 = int(np.floor(src))
        i1 = min(i0 + 1, lj - 1)
        lam = src - i0
        if 1 - lam > 0:
            sup[i0].add(t)
        if lam > 0 and i1 != i0:
            sup[i1].add(t)
    return sup


def oracle_flags(levels, replaced, L=128):
    """replaced: list of (level, position). Returns (S - l0,) flags by hand: contamination of the cumulative
    reconstruction, area-averaged over the span of every input position."""
    out = []
    for k in range(1, len(levels)):
        cont = np.zeros(L, bool)
        for (j, p) in replaced:
            if j >= k:
                continue
            if j == 0:
                cont[:] = True                                    # one code spans the whole window
            else:
                cont[list(linear_support(levels[j], L)[p])] = True
        lk = levels[k]
        for q in range(lk):
            lo, hi = (q * L) // lk, -(-((q + 1) * L) // lk)       # adaptive average pooling bin
            out.append(cont[lo:hi].mean())
    return np.array(out)


@pytest.mark.parametrize("learned", [True, False])
def test_flags_equal_the_hand_propagated_mask(learned):
    tok, _ = tiny_case(learned_l0=learned)
    levels = [int(l) for l in tok.vq.levels]
    cases = [[(3, 5)], [(1, 0)], [(1, 7)], [(2, 15), (6, 30)], [(9, 91)], [(0, 0)], [(10, 3), (4, 31)], []]
    masks = [torch.zeros(len(cases), l, dtype=torch.bool) for l in levels]
    for r, rep in enumerate(cases):
        for (j, p) in rep:
            masks[j][r, p] = True
    got = mask_flags(tok.vq, masks)
    assert got.shape == (len(cases), sum(levels[1:]))
    for r, rep in enumerate(cases):
        want = oracle_flags(levels, rep)
        assert np.allclose(got[r].numpy(), want, atol=1e-6), (learned, rep)
    assert float(got[-1].abs().sum()) == 0.0                          # no replaced code -> all-zero flags
    assert bool((got[5] == 1.0).all())                                # a replaced level-0 code contaminates every position
    assert float(got.min()) >= 0 and float(got.max()) <= 1


def test_flags_depend_on_the_sampled_mask_only_not_on_the_codes(case):
    tok, net = case
    assert list(inspect.signature(MechArm.flags).parameters) == ["self", "plan"]
    assert list(inspect.signature(mask_flags).parameters)[:2] == ["vq", "masks"]
    m = make_arm("m4", tok, net)
    x = torch.randint(0, 32, (16, 128))
    plan, canon, repl, h = apply_arm(m, tok, x, 16)
    f = m.flags(plan)
    # a masked slot whose replacement happened to equal the canonical code is still flagged
    h2 = build_hierarchy(tok, x, CorruptionCfg(mode="mech"), policy_codes=canon, masks=plan.masks, canonical=canon)
    assert int((h2["used"] != canon).sum()) == 0 and int(torch.cat(plan.masks, 1).sum()) > 0
    assert torch.equal(f, mask_flags(tok.vq, plan.masks))
    assert float(f.sum()) > 0


def test_non_flag_arms_have_no_flags(case):
    tok, net = case
    for arm in ("m0", "m1", "m2", "m3", "m6", "m7"):
        m = make_arm(arm, tok, net)
        assert m.flag_mode is None and m.flags(m.draw_plan(4)) is None


def test_flag_use_is_wired_into_training_only_for_m4_and_m5(case):
    """A nonzero w changes the training loss of M4 / M5 and cannot exist for the others."""
    base = Harness_loss("m4", w=None)
    moved = Harness_loss("m4", w=0.5)
    assert base != moved
    assert Harness_loss("m2", w=None) == base                           # w = 0: M4 is M2 exactly
    assert Harness_loss("m5", w=None) == base                           # shuffled flags with w = 0 as well


def Harness_loss(arm, w):
    import tempfile
    d = tempfile.mkdtemp()
    flags_w = None if w is None else torch.full((8,), float(w))
    h = Harness(d, arm=arm, flags_w=flags_w)
    h.run(1)
    return h.losses()[0]["loss"]


def test_zero_initialised_flag_vector_gives_the_released_models_logits(case):
    tok, net = case
    net.eval()
    x = torch.randint(0, 32, (4, 128))
    canon = tok.vq.z_to_idx(tok._encode_pre_quant(x))
    inputs = tok.vq.idx_to_ncm_input(canon)
    flags = torch.rand(4, inputs.shape[1])
    fv = FlagVector(tok.embed_dim)
    assert float(fv.w.abs().sum()) == 0.0
    with torch.no_grad():
        ref = net(torch.cat([prefix_fn(tok)(4), inputs], 1))
        got = net(torch.cat([prefix_fn(tok)(4), fv(inputs, flags)], 1))
        fv.w.fill_(0.3)
        moved = net(torch.cat([prefix_fn(tok)(4), fv(inputs, flags)], 1))
        none = net(torch.cat([prefix_fn(tok)(4), fv(inputs, torch.zeros_like(flags))], 1))
    assert torch.equal(ref, got)
    assert not torch.equal(ref, moved)
    assert torch.equal(ref, none)                                       # all-zero flags: w has no effect (inference)


def test_flag_vector_is_a_trainable_parameter_without_weight_decay_in_the_run():
    import tempfile
    h = Harness(tempfile.mkdtemp(), arm="m4")
    assert h.flagmod.w.requires_grad and h.flagmod.w.shape == (h.tok.embed_dim,)
    ps = [p for g in h.trainer.opt.param_groups for p in g["params"]]
    assert any(p is h.flagmod.w for p in ps)
    nd = next(g for g in h.trainer.opt.param_groups if any(p is h.flagmod.w for p in g["params"]))
    assert nd["weight_decay"] == 0.0
    h.run(1)
    assert float(h.flagmod.w.abs().sum()) > 0                           # it receives gradient and moves
    assert h.trainer.opt.state_dict()["state"]


def test_inference_and_evaluation_paths_never_use_the_flag_vector():
    funcs, _, _ = script_nodes("cur")
    import ast
    for name in ("val_nll", "draw_samples", "sample_metrics", "_sample_metrics", "policy_codes", "compute_loss"):
        names = {n.id for n in ast.walk(funcs[name]) if isinstance(n, ast.Name)}
        assert not names & {"flagmod", "mech", "FlagVector"}, name


def test_val_nll_is_unchanged_by_the_flag_vector(tmp_path):
    h = Harness(tmp_path, arm="m4", flags_w=torch.full((8,), 0.7))
    funcs, _, _ = script_nodes("cur")
    ns = dict(h.ns)
    ns.update(val_x=torch.randint(0, 32, (4, 128)), PAD=1)
    run_nodes([funcs["val_nll"]], ns)
    a = ns["val_nll"]()
    h.flagmod.w.data.zero_()
    assert ns["val_nll"]() == a


# ---------------------------------------------------------------------------------------------- shuffled flags
def test_shuffled_flags_preserve_per_row_per_level_values_and_use_independent_permutations():
    sizes = [8, 16, 24]
    B = 6
    f = torch.arange(B * sum(sizes), dtype=torch.float).view(B, -1)       # distinct values: a permutation is detectable
    g = torch.Generator().manual_seed(3)
    s = shuffle_within_level(f, sizes, g)
    for fb, sb in zip(torch.split(f, sizes, 1), torch.split(s, sizes, 1)):
        assert torch.equal(fb.sort(1).values, sb.sort(1).values)           # per-row, per-level counts / values preserved
        assert not torch.equal(fb, sb)
        perms = {tuple(torch.argsort(sb[r]).tolist()) for r in range(B)}
        assert len(perms) == B                                             # an independent permutation per row
    assert not torch.equal(s, shuffle_within_level(f, sizes, torch.Generator().manual_seed(4)))


def test_m5_flags_are_m4s_shuffled_and_use_their_own_generator(case):
    tok, net = case
    m4, m5 = make_arm("m4", tok, net), make_arm("m5", tok, net)
    sizes = m5.levels[1:]
    for _ in range(3):
        p4, p5 = m4.draw_plan(16), m5.draw_plan(16)
        f4, f5 = m4.flags(p4), m5.flags(p5)
        for a, b in zip(torch.split(f4, sizes, 1), torch.split(f5, sizes, 1)):
            assert torch.equal(a.sort(1).values, b.sort(1).values)
        assert not torch.equal(f4, f5)
        m4.proposals(m4.canonical(torch.zeros(16, 128, dtype=torch.long)), p4)
        m5.proposals(m5.canonical(torch.zeros(16, 128, dtype=torch.long)), p5)
    assert torch.equal(m4.mask_gen.get_state(), m5.mask_gen.get_state())   # shuffling never touches the mask stream
    assert torch.equal(m4.repl_gen.get_state(), m5.repl_gen.get_state())   # ... nor the replacement draws
    assert not torch.equal(m4.flag_gen.get_state(), m5.flag_gen.get_state())


def test_gradient_flows_only_through_the_flag_term_of_the_inputs(case):
    tok, net = case
    fv = FlagVector(tok.embed_dim)
    inp = torch.randn(2, 5, tok.embed_dim)
    out = fv(inp, torch.ones(2, 5)).sum()
    out.backward()
    assert torch.equal(fv.w.grad, torch.full_like(fv.w, 10.0))


# ---------------------------------------------------------------------------------------------- bookkeeping
def test_realized_disagreement_is_recorded_per_level_for_all_and_unclean_rows(case):
    tok, net = case
    m = make_arm("m3", tok, net)
    x = torch.randint(0, 32, (64, 128))
    plan, canon, repl, h = apply_arm(m, tok, x, 64)
    m.record(h["used"], h["targets"], plan)
    r = m.realized()
    dis = [d.float() for d in torch.split(h["used"] != h["targets"], m.levels, dim=1)]
    assert r["rows"] == 64 and r["clean_rows"] == 16
    for k in range(len(dis)):
        assert r["disagreement_all_rows_per_level"][k] == pytest.approx(float(dis[k].mean()), abs=1e-4)
        assert r["disagreement_corrupted_rows_per_level"][k] == pytest.approx(float(dis[k][~plan.clean].mean()), abs=1e-4)
    assert r["disagreement_all_rows_per_level"][0] == pytest.approx(0.75 * r["disagreement_corrupted_rows_per_level"][0], abs=1e-4)
    assert r["disagreement_all_rows_per_level"][-1] == 0.0
