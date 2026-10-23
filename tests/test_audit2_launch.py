"""Codex audit 2 tests, with external models replaced by tiny stubs.

Written as strict-xfail counterexamples against b397656 (docs/reviews/codex_audit2.md). After the R028 fixes they are
regression tests of the working tree: the xfail marks were removed, and tests whose fixture depended on the old code
structure were adapted to the fixed interfaces (noted per test).
"""
import sys
import ast
import io
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from audit2_support import definitions, jacobi_module, source
from ncmr.sudoku import BOS, cells, to_solution_ids

SDK = "scripts/train_sudoku_L.py"
WMT = "scripts/train_wmt14_L.py"


def consistency_fixture(kind="ordinary"):
    jt = jacobi_module()
    observed = []

    def logits(net, z, codes, pad_mask=None):
        if kind == "precision":
            new = torch.ones_like(codes) if torch.is_autocast_enabled("cpu") else torch.full_like(codes, 2)
        elif kind == "zero":
            new = torch.zeros_like(codes)
        else:
            new = torch.cat([torch.ones_like(codes[:, :1]), codes[:, :-1]], 1)
        return torch.nn.functional.one_hot(new, 3).float() * 10

    jt.parallel_logits = logits
    net = torch.nn.Linear(1, 1)

    def loss_logits(net, z, codes, pm):
        observed.append(codes.clone())
        return logits(net, z, codes, pm)

    ns = dict(torch=torch, net=net, m_states=4, S=2, K=2, dev="cpu",
              batch=lambda n: (torch.zeros(n, 1, dtype=torch.long), None),
              prefix=lambda src: (src.float()[:, :, None], torch.zeros_like(src, dtype=torch.bool)),
              jacobi_decode=jt.jacobi_decode, parallel_logits=loss_logits,
              traj_stats={"passes": 0., "n": 0, "states_before_fp": 0., "degenerate": 0},
              ce=lambda lg, target: target.clone())
    definitions(WMT, ["consist_loss"], ns)
    return ns, observed, jt, logits


def test_nondegenerate_sampling_is_strictly_pre_fixed_point_and_target_is_fixed():
    ns, seen, jt, lg = consistency_fixture()
    torch.manual_seed(0)
    target, n_used = ns["consist_loss"](64, torch.zeros(16, 1, dtype=torch.long))
    assert n_used == 64
    inp = seen[0]
    assert (inp != target).any(1).all()
    assert torch.equal(lg(None, None, target).argmax(-1), target)
    assert set(map(tuple, inp.tolist())) == {(0, 0), (1, 0)}


def test_consistency_target_is_the_fp32_fixed_point():
    ns, _, _, _ = consistency_fixture("precision")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        target, _ = ns["consist_loss"](4, torch.zeros(1, 1, dtype=torch.long))
    assert torch.equal(target, torch.full_like(target, 2))   # the fp32 map, although the caller runs under bf16


def test_no_fixed_point_state_is_sampled_as_a_pre_fixed_point_state():
    # adapted (R028 policy J1b): degenerate sources are dropped and counted, never trained on a fixed-point state
    ns, seen, _, _ = consistency_fixture("zero")
    loss, n_used = ns["consist_loss"](4, torch.zeros(1, 1, dtype=torch.long))
    assert loss is None and n_used == 0 and not seen
    assert ns["traj_stats"]["degenerate"] == 1


def pool_fixture(seed=7):
    rng = np.random.default_rng(seed)
    pool_rng = np.random.default_rng(seed + 1000)
    def batch(n, data_rng=None):
        p = torch.from_numpy((rng if data_rng is None else data_rng).integers(0, 10000, (n, 2)))
        return p, p.clone()
    ns = dict(torch=torch, batch=batch, S=2, dev="cpu", pool_rng=pool_rng)
    definitions(SDK, ["Pool"], ns)
    return ns, rng, batch


def test_restoring_pool_does_not_advance_saved_data_rng():
    ns, rng, batch = pool_fixture()
    pool = ns["Pool"](4)
    state = pool.state_dict()
    rng_state = rng.bit_generator.state
    expected = batch(4)[0]
    ns2, resumed_rng, resumed_batch = pool_fixture()
    resumed_rng.bit_generator.state = rng_state
    restored = ns2["Pool"](4, state=state)
    assert torch.equal(restored.prompt, pool.prompt)
    assert torch.equal(resumed_batch(4)[0], expected)


def test_final_repair_checkpoint_contains_pool_state():
    tree = ast.parse(source(SDK))
    final_save = next(n for n in tree.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "not a.eval_only")
    saves = []
    marker = {"codes": torch.tensor([[1, 2]])}
    pool = SimpleNamespace(state_dict=lambda: marker)
    ns = dict(a=SimpleNamespace(eval_only=False, run_dir="unused"), os=os,
              trainer=SimpleNamespace(save=lambda *a, **kw: saves.append(kw)),
              torch=SimpleNamespace(save=lambda *a, **kw: None),
              modules={"model": SimpleNamespace(state_dict=lambda: {})},
              rng=np.random.default_rng(0), pool=pool,
              gen=torch.Generator(), pool_rng=np.random.default_rng(1))
    # If a repaired version introduces a helper, execute that exact helper too.
    helper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint"]
    exec(compile(ast.Module(body=helper + [final_save], type_ignores=[]), SDK, "exec"), ns)
    assert saves and "pool" in saves[-1].get("extra", {})
    assert torch.equal(saves[-1]["extra"]["pool"]["codes"], marker["codes"])


def test_corruption_rng_resume_continues_uninterrupted_draws():
    # adapted: executes the generator creation and the restore block that follows it
    tree = ast.parse(source(SDK))
    i = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "gen" for t in n.targets))
    compiled = compile(ast.Module(body=tree.body[i:i + 2], type_ignores=[]), SDK, "exec")
    ns = dict(torch=torch, dev="cpu", seed=7, trainer=SimpleNamespace(step=0), corr=SimpleNamespace(mode="uniform"),
              resume_extra={})
    exec(compiled, ns)
    torch.rand(40, generator=ns["gen"])
    saved = ns["gen"].get_state()
    expected = torch.rand(10, generator=ns["gen"])
    ns.update(trainer=SimpleNamespace(step=1), resume_extra={"corruption_rng": saved})
    exec(compiled, ns)
    assert torch.equal(torch.rand(10, generator=ns["gen"]), expected)
    ns.update(resume_extra={})
    with pytest.raises(RuntimeError):
        exec(compiled, ns)


def test_pool_advance_reset_and_nonreset_rows():
    ns, rng, batch = pool_fixture()
    pool = ns["Pool"](3)
    pool.age[:] = torch.tensor([0, 1, 15])
    new = torch.tensor([[1, 0], [2, 1], [1, 2]])
    stats = {"solved": 0, "aged": 0, "advanced": 0}
    ns.update(pool=pool, pool_stats=stats, rc={"max_age": 16}, net=None,
              fb=lambda z, guess, active: z, wte=lambda p: p.float(),
              parallel_logits=lambda *a: torch.nn.functional.one_hot(new, 3).float(),
              decode_ids=lambda codes: codes.clone(),
              checks=lambda ids, prompt: torch.tensor([True, False, False]),
              canon=lambda sol: torch.zeros(3, 2, dtype=torch.long), F=torch.nn.functional)
    definitions(SDK, ["pool_loss"], ns)
    ns["pool_loss"]()
    assert torch.equal(pool.codes, torch.tensor([[0, 0], [2, 1], [0, 0]]))
    assert pool.active.tolist() == [False, True, False]
    assert pool.age.tolist() == [0, 2, 0]
    assert stats == {"solved": 1, "aged": 1, "advanced": 3}


def test_wmt_same_seed_next_step_data_order_matches():
    # adapted: both arms draw one common batch(bs) per step and consist_loss draws no data of its own
    tree = ast.parse(source(WMT))
    consist = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "consist_loss")
    assert "batch(" not in ast.unparse(consist)
    loop = next(n for n in tree.body if isinstance(n, ast.While))
    calls = [c for c in ast.walk(loop) if isinstance(c, ast.Call) and ast.unparse(c.func) == "batch"]
    assert [ast.unparse(c) for c in calls] == ["batch(bs)"]


def test_sdk_pool_resets_do_not_change_teacher_forcing_order():
    ns, _, batch = pool_fixture()
    ns2, _, control_batch = pool_fixture()
    ns["Pool"](128)
    assert torch.equal(batch(128)[0], control_batch(128)[0])


def test_wmt_realized_generator_proxy_is_matched_when_trajectories_converge_early():
    tree = ast.parse(source(WMT))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "fl_step" for t in n.targets))
    ns = dict(fwd=1, bs=128, K=11, n_c=64, m_states=4)
    exec(compile(ast.Module(body=[node], type_ignores=[]), WMT, "exec"), ns)
    assert 2000 * ns["fl_step"] == 3000 * (3 * 128)  # the registered proxy matches
    # adapted: the teacher runs a full budget, so a batch that converges in two passes still executes K + 1 passes
    jt = jacobi_module()
    calls = []
    def logits(net, z, codes, pad_mask=None):
        calls.append(len(codes))
        return torch.nn.functional.one_hot(torch.ones_like(codes), 2).float()
    jt.parallel_logits = logits
    out = jt.jacobi_decode(None, lambda c, h: None, 16, 1, "cpu", 12, keep_history=True, full_budget=True)
    assert out["nfe"].tolist() == [2] * 16 and len(out["history"]) == 12
    realized = 2000 * (3 * 128 + sum(calls))
    assert realized == 3000 * (3 * 128)
    consist = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "consist_loss")
    assert "full_budget=True" in ast.unparse(consist)


@pytest.mark.parametrize("path", [SDK, WMT])
def test_eval_only_does_not_load_a_trained_checkpoint(path):
    from omegaconf import OmegaConf
    from ncmr.corruption import CorruptionCfg
    tree = ast.parse(source(path))
    resumed = []
    class Trainer:
        def __init__(self, *a, **kw):
            self.step = 0
        def maybe_resume(self, modules, rng):
            # Represents loading compatible fine-tuned weights from ckpt.pt.
            resumed.append(True)
            self.step = 10
            return {}
    cfg = OmegaConf.create(dict(arm="ctrl", train=dict(seed=0, steps=20, batch_size=4,
                               lr=1e-3, warmup=1, weight_decay=0., ckpt_every=10)))
    net = torch.nn.Linear(1, 1)
    net.prefix_len = 1
    tok = SimpleNamespace(parameters=lambda: [], embed_dim=1,
                          vq=SimpleNamespace(levels=[1, 2], vqvae_vocab_sizes=[2, 2]))
    tok.eval = lambda: tok
    bundle = SimpleNamespace(net=net, vqvae=tok, pre_tokenizer=SimpleNamespace(eos_token_id=8, pad_token_id=9),
                             cfg=SimpleNamespace(model=SimpleNamespace(n_embd=1, n_layer=1)))
    fake_os = SimpleNamespace(path=SimpleNamespace(join=os.path.join, exists=lambda p: True), makedirs=lambda *a, **k: None)
    fake_np = SimpleNamespace(random=np.random, load=lambda p: dict(src=np.zeros((10, 1), dtype=np.int32), tgt=np.zeros((10, 1), dtype=np.int32)))
    ns = dict(a=SimpleNamespace(config="unused", run_dir="unused", data="unused.npz", seed=0,
                               calib_steps=0, eval_only=True), ROOT="unused", torch=torch,
              OmegaConf=SimpleNamespace(load=lambda p: cfg, save=lambda *a: None),
              os=fake_os, np=fake_np, json=json, open=lambda *a, **k: io.StringIO('{"src": [], "ref": []}'),
              pick_device=lambda: "cpu", load_ncm=lambda *a, **k: bundle,
              SudokuTrain=lambda *a: None, Trainer=Trainer, CorruptionCfg=CorruptionCfg,
              FL=SimpleNamespace(ncm_gen_fwd_per_seq=lambda *a, **k: 1), n_params=lambda m: 1)
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "cfg" for t in n.targets))
    # adapted: the fixed scripts define a digest() helper before the guard; stop at the first other function
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef) and n.name != "digest")
    ns.update(sys=sys, hashlib=SimpleNamespace(file_digest=lambda f, alg: SimpleNamespace(hexdigest=lambda: "x")))
    with pytest.raises(SystemExit):   # the directory guard refuses this released-model evaluation
        exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]), path, "exec"), ns)
    assert not resumed


def test_verifier_checks_rows_columns_boxes_and_clues_without_solution():
    from ncmr.sudoku import is_solution
    ns = dict(torch=torch, cells=cells, is_solution=is_solution)
    definitions(SDK, ["checks"], ns)
    board = [(r * 3 + r // 3 + c) % 9 + 1 for r in range(9) for c in range(9)]
    ids = torch.tensor([to_solution_ids(board)])
    puzzle = [0] * 81
    puzzle[0] = board[0]
    prompt = torch.tensor([[BOS] + to_solution_ids(puzzle) + [BOS]])
    assert ns["checks"](ids, prompt).item()
    bad = ids.clone(); bad[0, 0] = bad[0, 1]
    assert not ns["checks"](bad, prompt).item()
    shifted = torch.tensor([to_solution_ids([v % 9 + 1 for v in board])])
    assert not ns["checks"](shifted, prompt).item()


def test_wmt_target_text_eos_pad_and_length_filter():
    class Tokenizer:
        def __call__(self, texts, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [[int(x) for x in t.split()] for t in texts]}
    ns = dict(tok=Tokenizer(), P=3, L=4, EOS=8, PAD=9, np=np)
    definitions("scripts/prep_wmt14.py", ["encode"], ns)
    src, tgt, keep = ns["encode"]([("1 2 3 4", "4 5"), ("1", "1 2 3 4"), ("", "1"), ("1", "")])
    assert src.tolist() == [[1, 2, 3]]
    assert tgt.tolist() == [[4, 5, 8, 9]]
    assert keep == [0]


def test_prep_fills_requested_count_after_filtering():
    tree = ast.parse(source("scripts/prep_wmt14.py"))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "sel" for t in n.targets))
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "val" for t in n.targets))
    def encode(rows):
        # Nine tenths are filtered, but the full corpus has enough eligible rows.
        keep = [r for r in rows if r % 10 == 0]
        x = np.array(keep).reshape(-1, 1)
        return x, x, []
    ns = dict(np=np, rng=np.random.default_rng(0), train=list(range(40000)), a=SimpleNamespace(n=1000, shards=[0]),
              encode=encode, sys=sys)
    exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]), "prep_slice", "exec"), ns)
    assert len(ns["src"]) == 1000


def test_jacobi_caps_include_detection_and_freeze_stopped_rows():
    jt = jacobi_module()
    def logits(net, z, codes, pad_mask=None):
        new = torch.cat([torch.ones_like(codes[:, :1]), codes[:, :-1]], 1)
        new[0] = 0  # first row is fixed at initialization; detection still takes two passes
        return torch.nn.functional.one_hot(new, 2).float()
    jt.parallel_logits = logits
    out = jt.jacobi_decode(None, lambda c, h: None, 2, 4, "cpu", 5, keep_history=True)
    assert out["nfe"].tolist() == [2, 5]
    assert out["stop"].tolist() == [1, 1]
    for cap in range(1, 6):
        fresh = jt.jacobi_decode(None, lambda c, h: None, 2, 4, "cpu", cap)
        assert torch.equal(out["history"][cap - 1], fresh["codes"])
        assert torch.equal(out["nfe"].clamp(max=cap), fresh["nfe"])


def test_latest_jacobi_equals_released_sequential_on_cpu(released_sudoku):
    bundle, puzzles = released_sudoku
    jt = jacobi_module()
    net, tok = bundle.net.eval(), bundle.vqvae.eval()
    p = puzzles[:4]
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        seq = net.generate(None, p, temperature=0., top_k=0, top_p=1., cfg_scale=0., cfg_null_eos=None)
        prefix = tok.encoder.wte(p)
        out = jt.jacobi_decode(net, lambda c, h: prefix, 4, seq.shape[1], "cpu", len(tok.vq.levels) + 1)
        replay = jt.parallel_logits(net, prefix, out["codes"]).argmax(-1)
    assert torch.equal(seq, out["codes"])
    assert torch.equal(replay, out["codes"])
    assert out["stop"].eq(1).all()


def test_sudoku_reports_registered_nfe_tails_at_every_cap():
    class Net(torch.nn.Module):
        def generate(self, _, p, **kw):
            return torch.zeros_like(p)
    def decode(*args, **kw):
        return dict(history=[torch.zeros(3, 1, dtype=torch.long)] * 3,
                    nfe=torch.tensor([2, 3, 3]), stop=torch.tensor([1, 1, 0]))
    ns = dict(torch=torch, np=np, net=Net(), fb=None, dev="cpu", S=1,
              cells=lambda row: tuple(row), decode_ids=lambda c: c,
              jacobi_decode=decode, checks=lambda *a: torch.zeros(3, dtype=torch.bool))
    definitions(SDK, ["evaluate"], ns)
    out = ns["evaluate"](np.zeros((3, 1), dtype=np.int64), np.zeros((3, 1), dtype=np.int64), caps=(1, 2, 3))
    for mode in ("check", "nocheck"):
        for cap in (1, 2, 3):
            assert {"cap_hit_count", "p90_nfe", "max_nfe"} <= out[f"d2_{mode}"][f"cap{cap}"].keys()


def test_wmt_persists_capped_hypotheses_for_registered_comparisons():
    tree = ast.parse(source(WMT))
    dump = next(n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == "json.dump" and "hyp_test.json" in ast.unparse(n))
    saved = []
    capped = {6: ["capped hypothesis"]}
    ns = dict(json=SimpleNamespace(dump=lambda payload, stream: saved.append(payload)),
              open=lambda *a, **k: io.StringIO(), os=os,
              a=SimpleNamespace(run_dir="unused"), hyp=["exact hypothesis"], hyp_cap=capped,
              final={"test": {"nfe_per_sentence": [8]}})
    exec(compile(ast.Module(body=[dump], type_ignores=[]), WMT, "exec"), ns)
    assert saved[0].get("hyp_test_capped") == capped
