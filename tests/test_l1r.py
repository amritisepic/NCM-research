"""L1R / arm M2R (src/ncmr/l1r.py, scripts/train_gen.py; docs/preregistration.md "L1R"): the replay equals a real M2 / M2R
run's streams, M2R differs from M2 only in the receivers' L1 targets, the balanced selection is a maximum circulation,
resume is exact, the in-training root check fails closed, M0-M7 are unchanged."""
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ncmr import l1r as L1R
from ncmr import w2m as W2M
from ncmr.corruption import CorruptionCfg
from ncmr.w2m import ARMS, MechArm
from w2m_support import Harness, Stream, prefix_fn, run_nodes, script_nodes, tiny_case

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
V = 32                                    # tiny tokenizer: level-0 vocabulary
W1 = 8                                    # level-1 width


@pytest.fixture(autouse=True)
def varied_codes(monkeypatch):
    """The tiny random tokenizer maps almost every window to one root, which leaves no circulation to test. Both the
    run's and the replay's `canonical` get a root and an L1 block that depend on the window (same function on both)."""
    orig = MechArm.canonical

    def canonical(self, x):
        c = orig(self, x).clone()
        c[:, 0] = x[:, 0] % V
        c[:, 1:1 + W1] = x[:, 1:1 + W1] % V
        return c
    monkeypatch.setattr(MechArm, "canonical", canonical)


def dry_arm(tok, net, seed=0):
    return MechArm("m2r", tok, net, torch.ones(V) / V, prefix_fn(tok), "cpu", seed,
                   CorruptionCfg(mode="mech").rates(len(tok.vq.levels)), dry=True)


def replay_rec(seed=0, B=64, steps=6, accum=4):
    tok, net = tiny_case()
    arm, rng = dry_arm(tok, net, seed), np.random.default_rng(seed)
    rec = L1R.replay(lambda: torch.from_numpy(Stream().batch(rng, B)), arm, steps, accum, out_vocab=int(net.max_vocab))
    return rec, arm


def make_plan(seed=0, B=64, steps=6, accum=4, plan_seed=None):
    rec, _ = replay_rec(seed, B, steps, accum)
    return L1R.build_plan(rec, seed if plan_seed is None else plan_seed)


# ============================================================================================== AC1: replay exactness
def real_run_roots(arm_name, seed, B, steps, accum):
    """What a real M2 / M2R MechArm draws, step by step, with the training loop's partition."""
    tok, net = tiny_case()
    arm = MechArm(arm_name, tok, net, torch.ones(V) / V, prefix_fn(tok), "cpu", seed,
                  CorruptionCfg(mode="mech").rates(len(tok.vq.levels)), frozen=MechArm.frozen_copy(net))
    rng, rows = np.random.default_rng(seed), []
    for s in range(steps):
        x = torch.from_numpy(Stream().batch(rng, B))
        plan, row0 = arm.draw_plan(B), 0
        for xb in torch.chunk(x, accum):
            canon = arm.canonical(xb)
            repl = arm.proposals(canon, plan.rows(row0, row0 + len(xb)))
            for i in plan.masks[0][row0:row0 + len(xb), 0].nonzero().flatten().tolist():
                rows.append((s, row0 + i, int(canon[i, 0]), int(repl[i, 0]), canon[i, 1:1 + W1].tolist()))
            row0 += len(xb)
    return rows, arm


@pytest.mark.parametrize("arm_name", ["m2", "m2r"])
@pytest.mark.parametrize("B,accum", [(32, 4), (30, 4), (24, 3)])
def test_replay_equals_a_real_run_and_leaves_the_generators_in_the_same_state(arm_name, B, accum):
    steps = 5
    real, real_arm = real_run_roots(arm_name, 3, B, steps, accum)
    tok, net = tiny_case()
    arm, rng = dry_arm(tok, net, 3), np.random.default_rng(3)
    drng_before = rng.bit_generator.state
    torch_before = torch.get_rng_state().clone()
    rec = L1R.replay(lambda: torch.from_numpy(Stream().batch(rng, B)), arm, steps, accum, out_vocab=int(net.max_vocab))
    got = list(zip(rec["step"].tolist(), rec["row"].tolist(), rec["a"].tolist(), rec["u"].tolist(), rec["y"].tolist()))
    assert len(real) > 20 and got == real
    assert torch.equal(arm.repl_gen.get_state(), real_arm.repl_gen.get_state())      # consumption by shape matches
    assert torch.equal(arm.mask_gen.get_state(), real_arm.mask_gen.get_state())
    assert torch.equal(torch.get_rng_state(), torch_before)                          # no dropout-stream draw
    assert rng.bit_generator.state != drng_before                                    # (it advanced only through next_batch)
    assert all(a != u for _, _, a, u, _ in got)


def test_replay_without_the_fine_draws_does_not_match_the_real_stream():
    """Negative control for the test above: skipping the level >= 1 consumption gives other supplied roots."""
    real, _ = real_run_roots("m2", 3, 32, 5, 4)
    tok, net = tiny_case()
    arm, rng = dry_arm(tok, net, 3), np.random.default_rng(3)
    rec = L1R.replay(lambda: torch.from_numpy(Stream().batch(rng, 32)), arm, 5, 4, out_vocab=None)
    assert [u for *_, u, _ in real] != rec["u"].tolist()


def test_the_replay_is_deterministic_and_the_plan_sha_follows_its_arrays():
    p1, p2 = make_plan(), make_plan()
    assert p1.sha256 == p2.sha256
    assert make_plan(plan_seed=1).sha256 != p1.sha256                               # another selection seed
    assert make_plan(seed=1).sha256 != p1.sha256                                    # another data / mask stream
    q = make_plan()
    q.donor = q.donor.copy()
    k = int(np.nonzero(q.donor >= 0)[0][0])
    q.donor[k] = -1
    assert q.sha256 != p1.sha256


# ===================================================================================== AC3: the LP, selection, coverage
def fake_rec(arcs, B=1000):
    """records for arcs (a, u) in the given order, one row per step (so they are in (step, row) order)"""
    a, u = np.array([x[0] for x in arcs]), np.array([x[1] for x in arcs])
    return {"step": np.arange(len(a)), "row": np.zeros(len(a), dtype=np.int64), "a": a, "u": u,
            "y": np.arange(len(a))[:, None] * 10 + np.arange(W1)[None], "B": B, "steps": len(a)}


def brute_max(arcs):
    best = 0
    for mask in range(1 << len(arcs)):
        bal = {}
        n = 0
        for i, (a, u) in enumerate(arcs):
            if mask >> i & 1:
                bal[a] = bal.get(a, 0) + 1
                bal[u] = bal.get(u, 0) - 1
                n += 1
        if n > best and not any(bal.values()):
            best = n
    return best


def check_selection(arcs, seed=0):
    a, u = np.array([x[0] for x in arcs]), np.array([x[1] for x in arcs])
    rng = np.random.default_rng(seed)
    sel = L1R.select_balanced(a, u, rng)
    out, inn = np.bincount(a[sel], minlength=8), np.bincount(u[sel], minlength=8)
    assert (out == inn).all()                                                       # balanced: in = out at every root
    return sel, a, u, rng


def test_selection_is_balanced_and_maximal_against_brute_force_on_tiny_multigraphs():
    gen = np.random.default_rng(0)
    nonzero = 0
    for trial in range(60):
        n_arcs = int(gen.integers(1, 13))
        arcs = []
        while len(arcs) < n_arcs:
            a, u = (int(v) for v in gen.integers(0, 4, 2))
            if a != u:
                arcs.append((a, u))
        sel, *_ = check_selection(arcs, seed=trial)
        assert int(sel.sum()) == brute_max(arcs), arcs
        nonzero += int(sel.sum()) > 0
    assert nonzero > 15                                                             # the comparison is not vacuous


def test_hand_checked_circulations():
    assert check_selection([(0, 1), (1, 2)])[0].sum() == 0                          # a path has no circulation
    assert check_selection([(0, 1), (1, 2), (2, 0)])[0].all()                       # a 3-cycle is kept whole
    sel = check_selection([(0, 1)] * 3 + [(1, 0)] + [(2, 3)] * 2)[0]
    assert sel.sum() == 2 and sel[[3]].all() and sel[:3].sum() == 1 and not sel[4:].any()   # 1 of the 3 parallel arcs
    sel = check_selection([(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])[0]
    assert sel.all()


def test_the_rows_of_a_pair_are_drawn_at_random_and_seeded():
    arcs = [(0, 1)] * 6 + [(1, 0)] * 2
    picks = set()
    for seed in range(30):
        sel, *_ = check_selection(arcs, seed)
        assert sel[:6].sum() == 2 and sel[6:].all()
        picks.add(tuple(np.nonzero(sel[:6])[0]))
    assert len(picks) > 5                                                           # not the first two rows each time
    assert np.array_equal(check_selection(arcs, 7)[0], check_selection(arcs, 7)[0])


def test_the_lp_solution_is_verified_and_fails_closed(monkeypatch):
    import scipy.optimize as so
    real = so.linprog
    def shifted(*a, **k):
        r = real(*a, **k)
        return SimpleNamespace(status=0, x=r.x + 0.4, message="")
    monkeypatch.setattr(so, "linprog", shifted)
    with pytest.raises(RuntimeError, match="not integral"):
        L1R.max_circulation(np.array([0, 1]), np.array([1, 0]))

    def unbalanced(*a, **k):
        r = real(*a, **k)
        x = r.x.copy()
        x[0] = max(0.0, x[0] - 1)
        return SimpleNamespace(status=0, x=x, message="")
    monkeypatch.setattr(so, "linprog", unbalanced)
    with pytest.raises(RuntimeError, match="conservation"):
        L1R.max_circulation(np.array([0, 1, 0]), np.array([1, 0, 1]))


def test_donor_assignment_is_a_balanced_permutation_within_each_root():
    rec = replay_rec()[0]
    plan = L1R.build_plan(rec, 0)
    sel = plan.donor >= 0
    assert plan.n_selected == int(sel.sum()) > 30
    S = np.nonzero(sel)[0]
    assert sorted(plan.donor[S].tolist()) == S.tolist()                             # each selected row is a donor exactly once
    assert (plan.a[plan.donor[S]] == plan.u[S]).all()                               # donor's canonical root = receiver's supplied root
    assert (plan.donor[S] != S).all()                                               # nobody gets its own block
    assert (plan.donor[~sel] == -1).all()
    out, inn = np.bincount(plan.a[S], minlength=V), np.bincount(plan.u[S], minlength=V)
    assert (out == inn).all()
    # random, not a fixed order: another seed re-pairs (same selection size)
    other = L1R.build_plan(rec, 5)
    assert other.n_selected == plan.n_selected and not np.array_equal(other.donor, plan.donor)
    with pytest.raises(RuntimeError, match="unbalanced"):
        L1R.assign_donors(np.array([0, 1]), np.array([1, 2]), np.array([True, True]), np.random.default_rng(0))


def test_coverage_is_selected_over_root_corrupted_rows(tmp_path):
    arcs = [(0, 1)] * 3 + [(1, 0)] + [(2, 3)] * 2
    plan = L1R.build_plan(fake_rec(arcs), 0)
    assert plan.n_records == 6 and plan.n_selected == 2 and plan.coverage == pytest.approx(1 / 3)
    s = plan.summary()
    assert s["coverage"] == pytest.approx(1 / 3) and s["n_root_corrupted_rows"] == 6 and s["n_selected"] == 2
    assert len(s["plan_sha256"]) == 64
    assert L1R.build_plan(fake_rec([(0, 1), (1, 0)]), 0).coverage == 1.0
    assert L1R.build_plan(fake_rec([(0, 1), (1, 2)]), 0).coverage == 0.0


def test_the_gate_raises_below_coverage_min_and_writes_the_summary(tmp_path):
    plan = L1R.build_plan(fake_rec([(0, 1)] * 3 + [(1, 0)] + [(2, 3)] * 2), 0)      # coverage 1/3
    with pytest.raises(L1R.L1RCoverageError, match="coverage gate"):
        L1R.gate(plan, 0.80, str(tmp_path))
    j = json.load(open(tmp_path / "l1r_plan.json"))
    assert j["gate"] == "FAILED" and j["n_selected"] == 2 and j["n_root_corrupted_rows"] == 6
    assert not (tmp_path / "l1r_plan.npz").exists()
    os.makedirs(tmp_path / "ok")
    L1R.gate(plan, 0.30, str(tmp_path / "ok"))
    j = json.load(open(tmp_path / "ok" / "l1r_plan.json"))
    assert j["plan_sha256"] == plan.sha256 and "gate" not in j
    z = np.load(tmp_path / "ok" / "l1r_plan.npz")
    assert np.array_equal(z["donor"], plan.donor) and np.array_equal(z["y"], plan.y) and int(z["B"]) == plan.B


def small_cfg(tmp_path, coverage_min, steps=2):
    c = OmegaConf.load(ROOT / "configs/scale/L_ts_m2r_frozen_pw_l1reassign.yaml")
    c.train.update({"steps": steps, "batch_size": 16, "grad_accum": 2, "eval_n": 8, "eval_every": 10 ** 6, "ckpt_every": 10 ** 6})
    c.l1r.coverage_min = coverage_min
    c.sample_eval = {}
    p = tmp_path / "cfg.yaml"
    OmegaConf.save(c, p)
    return p


def run_train_gen(args):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": os.pathsep.join([str(ROOT / "src")])}
    return subprocess.run([PY, str(ROOT / "scripts/train_gen.py"), *args], capture_output=True, text=True, env=env, timeout=600,
                          cwd=ROOT)


def test_train_gen_exits_nonzero_before_training_when_coverage_is_below_the_minimum(tmp_path):
    run = tmp_path / "run"
    r = run_train_gen(["--config", str(small_cfg(tmp_path, 0.9999)), "--run_dir", str(run), "--skip_final_eval"])
    assert r.returncode != 0 and "coverage gate failed" in r.stderr, r.stderr[-800:]
    j = json.load(open(run / "l1r_plan.json"))
    assert j["gate"] == "FAILED" and j["coverage"] < 0.9999 and j["n_root_corrupted_rows"] > 0
    assert not (run / "ckpt.pt").exists() and not (run / "final.pt").exists() and not (run / "metrics.jsonl").exists()


# ===================================================== the training loop (the real while node) with tiny CPU objects
def attach_plan(monkeypatch, plan):
    """Every MechArm built by the Harness is an M2R arm carrying `plan` (before any resume node runs)."""
    init = MechArm.__init__

    def init2(self, *a, **k):
        init(self, *a, **k)
        self.arm, self.l1r = "m2r", plan
    monkeypatch.setattr(MechArm, "__init__", init2)


class Capture:
    """Records, per micro-batch, build_hierarchy's inputs / outputs and the targets the loss is computed on."""

    def __init__(self, h):
        self.calls, self.ce = [], []
        bh = h.ns["build_hierarchy"]

        def bh2(*a, **k):
            o = bh(*a, **k)
            self.calls.append({"h": o, "masks": [m.clone() for m in k["masks"]], "repl": k["policy_codes"].clone()})
            return o

        def ce(logits, t, **k):
            self.ce.append(t.clone())
            return F.cross_entropy(logits, t, **k)
        h.ns["build_hierarchy"], h.ns["F"] = bh2, SimpleNamespace(cross_entropy=ce)


def run_captured(tmp, plan, steps, B=64, accum=4, monkeypatch=None):
    with monkeypatch.context() as mp:
        if plan is not None:
            attach_plan(mp, plan)
        h = Harness(tmp, arm="m2", B=B, accum=accum)
        cap = Capture(h)
        h.run(steps)
    return h, cap


def test_m2r_equals_m2_everywhere_except_the_receivers_l1_targets(tmp_path, monkeypatch):
    B, accum, steps = 64, 4, 6
    plan = make_plan(B=B, steps=steps, accum=accum)
    assert plan.n_selected > 30 and plan.coverage > 0.3
    _, m2 = run_captured(tmp_path / "m2", None, steps, B, accum, monkeypatch)
    _, m2r = run_captured(tmp_path / "m2r", plan, steps, B, accum, monkeypatch)
    nb = B // accum
    assert len(m2.calls) == len(m2r.calls) == len(m2.ce) == len(m2r.ce) == steps * accum
    changed, all_l1_m2, all_l1_m2r = [], [], []
    canon_root, canon_l1, supplied = {}, {}, {}
    for i, (c0, c1, t0, t1) in enumerate(zip(m2.calls, m2r.calls, m2.ce, m2r.ce)):
        step, row0 = divmod(i, accum)[0], (i % accum) * nb
        for k in ("inputs", "targets", "canonical", "used", "z_used"):               # inputs, labels, codes fed forward
            assert torch.equal(c0["h"][k], c1["h"][k]), (i, k)
        assert torch.equal(c0["repl"], c1["repl"]) and all(torch.equal(x, y) for x, y in zip(c0["masks"], c1["masks"]))
        t0, t1 = t0.view(nb, -1), t1.view(nb, -1)
        assert torch.equal(t0, c0["h"]["targets"][:, 1:])                            # M2's loss targets are the canonical labels
        assert torch.equal(t0[:, W1:], t1[:, W1:])                                   # level >= 2 targets identical
        for r in range(nb):
            key = step * B + row0 + r
            canon_root[key], canon_l1[key] = int(c0["h"]["canonical"][r, 0]), t0[r, :W1].tolist()
            supplied[key] = int(c0["repl"][r, 0]) if bool(c0["masks"][0][r, 0]) else None
            all_l1_m2.append(tuple(t0[r, :W1].tolist()))
            all_l1_m2r.append(tuple(t1[r, :W1].tolist()))
            if not torch.equal(t0[r, :W1], t1[r, :W1]):
                changed.append(key)
    recv = {int(plan.key[j]): j for j in np.nonzero(plan.donor >= 0)[0]}
    assert set(changed) <= set(recv)                                                 # L1 differs only on receivers
    expect = {k for k, j in recv.items() if not np.array_equal(plan.y[plan.donor[j]], plan.y[j])}
    assert set(changed) == expect and len(changed) > 25
    assert sorted(all_l1_m2) == sorted(all_l1_m2r)                                   # the multiset of L1 blocks is unchanged
    for key, j in recv.items():
        d = int(plan.donor[j])
        dkey = int(plan.key[d])
        assert supplied[key] == plan.u[j] and canon_root[key] == plan.a[j]
        assert canon_root[dkey] == supplied[key]                                     # the block's canonical root = supplied root
        assert dkey != key and canon_l1[dkey] == plan.y[d].tolist()
    # the receiver's block in the M2R run is exactly that donor's canonical block
    by_key = {}
    for i, t1 in enumerate(m2r.ce):
        for r, row in enumerate(t1.view(nb, -1)[:, :W1].tolist()):
            by_key[(i // accum) * B + (i % accum) * nb + r] = row
    assert all(by_key[k] == canon_l1[int(plan.key[plan.donor[j]])] for k, j in recv.items())


def test_the_loss_is_computed_on_the_reassigned_targets(tmp_path, monkeypatch):
    plan = make_plan(steps=2)
    base = Harness(tmp_path / "a", arm="m2", B=64, accum=4).run(1)
    with monkeypatch.context() as mp:
        attach_plan(mp, plan)
        r = Harness(tmp_path / "b", arm="m2", B=64, accum=4).run(1)
    assert base.losses()[0]["loss"] != r.losses()[0]["loss"]                         # same inputs, other L1 targets
    assert r.losses()[0]["l1r_reassigned_frac"] > 0 and "l1r_reassigned_frac" not in base.losses()[0]
    assert plan.n_selected > 0


def test_the_stats_record_the_canonical_labels_not_the_reassigned_ones(tmp_path, monkeypatch):
    plan = make_plan(steps=2)
    base = Harness(tmp_path / "a", arm="m2", B=64, accum=4).run(2)
    with monkeypatch.context() as mp:
        attach_plan(mp, plan)
        r = Harness(tmp_path / "b", arm="m2", B=64, accum=4).run(2)
    assert base.mech.stats == r.mech.stats


# ==================================================================================================== AC4: resume
def test_resume_equals_an_uninterrupted_m2r_run_and_a_changed_plan_fails_closed(tmp_path, monkeypatch):
    plan = make_plan(steps=8)
    with monkeypatch.context() as mp:
        attach_plan(mp, plan)
        whole = Harness(tmp_path / "whole", arm="m2", B=64, accum=4).run(4)
        first = Harness(tmp_path / "split", arm="m2", B=64, accum=4).run(2)
        st = torch.load(tmp_path / "split" / "ckpt.pt", map_location="cpu", weights_only=False)
        assert st["extra"]["l1r_plan_sha256"] == plan.sha256 and first.trainer.step == 2
        rest = Harness(tmp_path / "split", arm="m2", B=64, accum=4, resume=True)
        assert rest.trainer.step == 2
        rest.run(4)
    a, b = whole.state(), rest.state()
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    for k in ("mask_rng", "repl_rng", "flag_rng"):
        assert torch.equal(a["mech"][k], b["mech"][k]), k
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert [r["loss"] for r in whole.losses()[2:]] == [r["loss"] for r in rest.losses()[2:]]
    assert [r.get("l1r_reassigned_frac") for r in whole.losses()[2:]] == [r.get("l1r_reassigned_frac") for r in rest.losses()[2:]]
    assert any(r["l1r_reassigned_frac"] > 0 for r in rest.losses()[2:])
    # a changed plan (here: another selection seed) fails closed, as does a checkpoint without a plan hash
    other = make_plan(steps=8, plan_seed=77)
    assert other.sha256 != plan.sha256
    with monkeypatch.context() as mp:
        attach_plan(mp, other)
        with pytest.raises(RuntimeError, match="L1R plan differs"):
            Harness(tmp_path / "split", arm="m2", B=64, accum=4, resume=True)
    Harness(tmp_path / "plain", arm="m2", B=64, accum=4).run(2)                     # a checkpoint of a plan-less M2 run
    with monkeypatch.context() as mp:
        attach_plan(mp, plan)
        with pytest.raises(RuntimeError, match="L1R plan differs"):
            Harness(tmp_path / "plain", arm="m2", B=64, accum=4, resume=True)


# ====================================================================================== AC5: the in-training check
@pytest.mark.parametrize("what", ["a", "u", "y", "row", "drop", "extra"])
def test_the_root_check_stops_the_run_on_any_difference_from_the_replay(tmp_path, monkeypatch, what):
    plan = make_plan(steps=4)
    arrays = ("step", "row", "a", "u", "y")
    ok = L1R.L1RPlan({**{k: getattr(plan, k) for k in arrays}, "B": plan.B, "steps": plan.steps}, plan.donor)
    with monkeypatch.context() as mp:
        attach_plan(mp, ok)
        Harness(tmp_path / "ok", arm="m2", B=64, accum=4).run(3)                     # the unmodified plan runs
    idx = int(np.nonzero(plan.step == 2)[0][1])                                      # a record of the third step
    if what == "row":                                                                # the same values, recorded one row later
        rows = set(plan.row[plan.step == 2].tolist())
        idx = next(int(i) for i in np.nonzero((plan.step == 2) & (plan.donor < 0))[0] if plan.row[i] + 1 not in rows)
    if what == "a":                                                                  # (an unselected row: no earlier donor depends on it)
        idx = int(np.nonzero((plan.step == 2) & (plan.donor < 0))[0][0])
    rec = {k: getattr(plan, k).copy() for k in arrays}
    donor = plan.donor.copy()
    if what == "a":
        rec["a"][idx] = (rec["a"][idx] + 1) % V
    elif what == "u":
        rec["u"][idx] = (rec["u"][idx] + 1) % V
    elif what == "y":
        rec["y"][idx, 3] += 1
    elif what == "row":
        rec["row"][idx] += 1
    elif what == "drop":
        rec = {k: np.delete(v, idx, 0) for k, v in rec.items()}
        donor = np.full(len(rec["a"]), -1)
    else:                                                                            # a row the replay never had
        free = next(r for r in range(64) if r not in set(plan.row[plan.step == 2].tolist()))
        new = {"step": 2, "row": free, "a": rec["a"][idx], "u": rec["u"][idx], "y": rec["y"][idx]}
        rec = {k: np.insert(v, idx, new[k], 0) for k, v in rec.items()}
        order = np.argsort(rec["step"] * plan.B + rec["row"], kind="stable")
        rec, donor = {k: v[order] for k, v in rec.items()}, np.full(len(order), -1)
    bad = L1R.L1RPlan({**rec, "B": plan.B, "steps": plan.steps}, donor)
    with monkeypatch.context() as mp:
        attach_plan(mp, bad)
        h = Harness(tmp_path / "bad", arm="m2", B=64, accum=4)
        with pytest.raises(L1R.L1RMismatch, match="differ from the pre-training replay"):
            h.run(4)
    assert h.trainer.step == 2                                                       # stopped at the first differing step


# ============================================================================== AC6 / AC7: other arms, run identity
def test_arms_m0_to_m7_are_unchanged_and_m2r_is_a_separate_registry_entry():
    assert sorted(ARMS) == [f"m{i}" for i in range(8)] and ARMS["m2"] == ("frozen_wrong", None)
    assert W2M.L1R_ARMS == {"m2r": ("frozen_wrong", None)} and "m2r" not in ARMS and W2M.ALL_ARMS["m2r"] == ARMS["m2"]


def test_m2_proposals_are_bit_for_bit_the_pre_refactor_code():
    tok, net = tiny_case()
    new = MechArm("m2", tok, net, torch.ones(V) / V, prefix_fn(tok), "cpu", 5, CorruptionCfg(mode="mech").rates(12),
                  frozen=MechArm.frozen_copy(net))
    old = MechArm("m2", tok, net, torch.ones(V) / V, prefix_fn(tok), "cpu", 5, CorruptionCfg(mode="mech").rates(12),
                  frozen=MechArm.frozen_copy(net))
    rng = np.random.default_rng(5)
    for _ in range(3):
        x = torch.from_numpy(Stream().batch(rng, 12))
        canon = new.canonical(x)
        got = new.proposals(canon, None)
        B, g, l0 = canon.shape[0], old.repl_gen, old.l0                              # the code of proposals() before the refactor
        probs0 = old.L0[None].expand(B * l0, -1).clone()
        probs0.scatter_(1, canon[:, :l0].reshape(-1, 1), 0.0)
        r0 = torch.multinomial(probs0, 1, generator=g).view(B, l0)
        logits = old._logits(old.frozen, old.frozen, canon)
        logits.scatter_(-1, canon[:, l0:, None], float("-inf"))
        probs = torch.softmax(logits / old.policy_temp, -1)
        samp = torch.multinomial(probs.flatten(0, 1), 1, generator=g).view(logits.shape[:-1])
        assert torch.equal(got, torch.cat([r0, samp], 1))
        assert torch.equal(new.repl_gen.get_state(), old.repl_gen.get_state())


def test_run_key_distinguishes_m2r_from_m2_and_the_config_is_m2s_with_three_changes():
    funcs, _, _ = script_nodes("cur")
    ns = {"OmegaConf": OmegaConf}
    run_nodes([funcs["_run_key"]], ns)
    m2 = OmegaConf.load(ROOT / "configs/scale/L_ts_m2_frozen_pw.yaml")
    m2r = OmegaConf.load(ROOT / "configs/scale/L_ts_m2r_frozen_pw_l1reassign.yaml")
    assert ns["_run_key"](m2) != ns["_run_key"](m2r)
    flipped = OmegaConf.create(OmegaConf.to_container(m2))
    flipped.corruption.arm = "m2r"
    assert ns["_run_key"](flipped) != ns["_run_key"](m2)                             # the arm alone distinguishes the runs
    flipped.train.grad_accum = 4
    assert ns["_run_key"](flipped) != ns["_run_key"](m2)
    assert ns["_run_key"](flipped) == ns["_run_key"](m2r)                            # the arm (and accumulation) is the only difference
    a, b = OmegaConf.to_container(m2), OmegaConf.to_container(m2r)
    assert b.pop("l1r") == {"coverage_min": 0.80}
    assert a["corruption"].pop("arm") == "m2" and b["corruption"].pop("arm") == "m2r"
    assert a["train"].pop("grad_accum") == 8 and b["train"].pop("grad_accum") == 4
    assert a == b                                                                    # same recipe, steps, seed, eval cadence


def test_train_gen_wires_the_replay_before_training_and_only_for_m2r():
    src = (ROOT / "scripts/train_gen.py").read_text(encoding="utf-8")
    code = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
    i_replay = next(i for i, l in enumerate(code) if "L1R.replay(" in l)
    i_loop = next(i for i, l in enumerate(code) if l.startswith("while trainer.step"))
    assert i_replay < i_loop
    assert any('mech.arm == "m2r"' in l for l in code[:i_replay])


# ============================================================================================== the coverage script
def test_coverage_script_runs_at_small_size_and_writes_a_path_free_json(tmp_path):
    out = tmp_path / "cov.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    r = subprocess.run([PY, str(ROOT / "scripts/l1r_coverage.py"), "--steps", "3", "--batch", "32", "--out", str(out)],
                       capture_output=True, text=True, env={**env, "PYTHONPATH": str(ROOT / "src")}, timeout=600, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-800:]
    j = json.load(open(out))
    assert j["canonical_encode_device"] == "cpu" and j["steps"] == 3 and j["batch_size"] == 32
    assert 0 < j["n_root_corrupted_rows"] <= 96 and 0 <= j["coverage"] <= 1
    assert j["coverage"] == pytest.approx(j["n_selected"] / j["n_root_corrupted_rows"])
    assert str(tmp_path) not in out.read_text() and str(ROOT) not in out.read_text()


def test_the_replays_dummy_draw_has_the_shape_and_dtype_of_the_trainings_draw(monkeypatch):
    """On CUDA the generator advance depends on the probability tensor's numel and dtype, which the CPU stream test cannot
    see: pin them here (real `proposals` call vs `skip_fine` with the vocabulary train_gen passes, `net.max_vocab`)."""
    tok, net = tiny_case()
    arm = MechArm("m2", tok, net, torch.ones(V) / V, prefix_fn(tok), "cpu", 0, CorruptionCfg(mode="mech").rates(12),
                  frozen=MechArm.frozen_copy(net))
    seen = []
    orig = MechArm.draw_fine
    monkeypatch.setattr(MechArm, "draw_fine", lambda self, p: (seen.append((tuple(p.shape), p.dtype, p.device, p.is_contiguous())), orig(self, p))[1])
    x = torch.from_numpy(Stream().batch(np.random.default_rng(0), 12))
    arm.proposals(arm.canonical(x), arm.draw_plan(12))
    arm.skip_fine(12, int(net.max_vocab))
    assert len(seen) == 2 and seen[0] == seen[1] and seen[0][0] == (12, 640, int(net.max_vocab))


@pytest.mark.parametrize("B,accum", [(32, 4), (30, 4)])
def test_the_replay_makes_the_same_fine_draws_as_a_real_run_micro_batch_by_micro_batch(monkeypatch, B, accum):
    """CPU multinomial advances the generator by a constant, whatever the shape, so the stream comparison above cannot see the
    shape; on CUDA the advance follows the tensor's numel. Compare the draws' shapes / dtypes one by one instead."""
    seen = []
    orig = MechArm.draw_fine
    monkeypatch.setattr(MechArm, "draw_fine", lambda self, p: (seen.append((tuple(p.shape), p.dtype, p.device, p.is_contiguous())), orig(self, p))[1])
    real_run_roots("m2r", 3, B, 4, accum)
    n_real, real = len(seen), list(seen)
    seen.clear()
    tok, net = tiny_case()
    arm, rng = dry_arm(tok, net, 3), np.random.default_rng(3)
    L1R.replay(lambda: torch.from_numpy(Stream().batch(rng, B)), arm, 4, accum, out_vocab=int(net.max_vocab))
    assert n_real == 4 * accum and seen == real
    assert len({s[0] for s in real}) == (2 if B % accum else 1)                     # the uneven last micro-batch is covered


def test_the_gate_accepts_a_coverage_exactly_at_the_minimum(tmp_path):
    plan = L1R.build_plan(fake_rec([(0, 1), (1, 0), (2, 3), (5, 6)]), 0)
    assert plan.coverage == 0.5
    L1R.gate(plan, 0.5, str(tmp_path))
    assert (tmp_path / "l1r_plan.npz").exists()
