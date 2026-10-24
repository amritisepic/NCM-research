"""Working-tree audit probes. Expected defects are strict assertion-only xfails.

Run with Torch 2.11 (the W1 version); no script CLI or production loop is run.
"""
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from audit2_support import definitions
from ncmr.corruption import CorruptionCfg, build_hierarchy, soft_ce
from ncmr.models import build_ncm_generator
from ncmr.train import Trainer
from ncmr import train as train_module
from test_audit_hierarchy import tiny_text_tokenizer
from test_audit_regressions import policy_fixture as old_policy_fixture

ROOT = Path(__file__).resolve().parents[1]
DEFECT = dict(strict=True, raises=AssertionError)


def policy_fixture(passes=1):
    policy, ctx, seen = old_policy_fixture(passes)
    if hasattr(train_module, "fp32_eval"):
        inspect.unwrap(policy).__globals__["fp32_eval"] = train_module.fp32_eval
    return policy, ctx, seen


def text_case(tmp_path, passes=1):
    tok = tiny_text_tokenizer(learned_l0=True, seed=42)
    for p in tok.parameters():
        p.requires_grad_(False)
    net = build_ncm_generator(tok, n_layer=1, n_head=2, n_embd=32, dropout=0.1)
    tr = Trainer(str(tmp_path), net.parameters(), max_lr=5e-5, total_steps=4,
                 device="cpu", amp=True)
    levels = list(tok.vq.levels)
    x = torch.randint(0, 32, (2, 128))
    ns = dict(torch=torch, F=F, net=net, fnet=net, tok=tok, trainer=tr, device="cpu",
              policy_passes=passes, policy_temp=1.0, L0=torch.ones(32)/32, l0=1,
              K=len(levels), levels=levels, val_x=x, corr=CorruptionCfg(mode="onpolicy"),
              prefix_emb=lambda b: tok.encoder.wte(torch.zeros(b, 1, dtype=torch.long)),
              build_hierarchy=build_hierarchy, soft_ce=soft_ce,
              policy_gen=torch.Generator().manual_seed(104729))
    if hasattr(train_module, "fp32_eval"):
        ns["fp32_eval"] = train_module.fp32_eval
    definitions("scripts/train_gen.py", ["policy_codes", "compute_loss", "val_nll"], ns)
    return ns, x


@pytest.mark.parametrize("retarget", [False, True])
def test_canonical_policy_identity_released(released_sudoku, retarget):
    bundle, puzzles = released_sudoku
    tok, x = bundle.vqvae, puzzles[:2, 1:-1].contiguous()
    canonical = tok.encode(x)
    clean = build_hierarchy(tok, x, CorruptionCfg())
    cfg = CorruptionCfg(mode="onpolicy", mean_rate=1, spread=0, retarget=retarget)
    actual = build_hierarchy(tok, x, cfg, canonical, torch.Generator().manual_seed(9))
    for key in ["inputs", "targets", "canonical", "used", "z_used"]:
        assert torch.equal(actual[key], clean[key]), key
    assert torch.equal(actual["inputs"], tok.vq.idx_to_ncm_input(canonical))


def test_retarget_changes_labels_and_unmasked_finer_inputs(released_sudoku):
    bundle, puzzles = released_sudoku
    tok, x = bundle.vqvae, puzzles[:2, 1:-1].contiguous()
    canonical = tok.encode(x)
    pol = canonical.clone()
    pol[:, :1] = (pol[:, :1] + 1) % 16
    cfg = CorruptionCfg(mode="onpolicy", retarget=True,
                        level_rates=(1.0,) + (0.0,) * (len(tok.vq.levels)-1))
    actual = build_hierarchy(tok, x, cfg, pol, torch.Generator().manual_seed(9))
    assert not torch.equal(actual["targets"][:, 1:], canonical[:, 1:])
    assert torch.equal(actual["used"][:, 1:], actual["targets"][:, 1:])
    assert torch.equal(actual["inputs"], tok.vq.idx_to_ncm_input(actual["used"]))
    clean_labels = build_hierarchy(tok, x, CorruptionCfg(mode="onpolicy", level_rates=cfg.level_rates), pol)
    assert torch.equal(clean_labels["targets"], canonical)


def test_two_pass_uses_entire_previous_policy(monkeypatch):
    policy, ctx, seen = policy_fixture(2)
    outputs = []
    original = torch.multinomial
    level0_calls = []
    def sample(probabilities, *args, **kwargs):
        value = original(probabilities, *args, **kwargs)
        if probabilities.ndim == 2:
            outputs.append(value.reshape(2, -1).clone())
        else:
            level0_calls.append(value.clone())
        return value
    monkeypatch.setattr(torch, "multinomial", sample)
    with ctx:
        final = policy(torch.zeros(2, 3, dtype=torch.long), torch.Generator().manual_seed(0))
    assert torch.equal(seen[1], torch.cat([final[:, :1], outputs[0]], 1))
    assert torch.equal(final[:, 1:], outputs[1])
    assert len(level0_calls) == 1
    assert not torch.is_autocast_enabled("cpu")


def test_text_parallel_alignment_matches_forced_sequential(tmp_path, monkeypatch):
    ns, x = text_case(tmp_path)
    net, tok = ns["net"].eval(), ns["tok"]
    canonical = tok.encode(x)
    with torch.no_grad():
        full = net(torch.cat([ns["prefix_emb"](len(x)), tok.vq.idx_to_ncm_input(canonical)], 1))
    seen, off = [], [1]
    def force(logits, *args):
        seen.append(logits.float())
        codes = canonical[:, off[0]:off[0]+logits.shape[1]]
        off[0] += logits.shape[1]
        return codes
    monkeypatch.setattr(net, "_sample_level", force)
    result = net.generate(canonical[:, :1], torch.zeros(len(x), 1, dtype=torch.long),
                          temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0)
    assert torch.equal(result, canonical)
    torch.testing.assert_close(torch.cat(seen, 1), full.float(), rtol=1e-4, atol=1e-5)


@pytest.mark.xfail(reason="A3-1: policy forward inherits training AMP and TF32", **DEFECT)
def test_text_policy_forward_is_fp32_with_tf32_off(tmp_path):
    ns, x = text_case(tmp_path)
    seen = []
    hook = ns["net"].register_forward_hook(lambda m, a, y: seen.append(
        (y.dtype, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)))
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        with ns["trainer"].ctx:
            ns["policy_codes"](x, torch.Generator().manual_seed(0))
            assert torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cpu")
        assert seen == [(torch.float32, False, False)]
    finally:
        hook.remove()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


@pytest.mark.xfail(reason="A3-1: val_nll enters trainer.ctx and leaves TF32 enabled", **DEFECT)
def test_val_nll_is_fp32_scoring(tmp_path):
    ns, _ = text_case(tmp_path)
    seen = []
    hook = ns["net"].register_forward_hook(lambda m, a, y: seen.append(
        (y.dtype, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)))
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        out = ns["val_nll"]()
        assert out["val_nll"] > 0
        assert seen == [(torch.float32, False, False)]
    finally:
        hook.remove()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


@pytest.mark.xfail(reason="A3-2: categorical policy samples consume the training global RNG", **DEFECT)
def test_policy_preserves_global_training_rng():
    policy, ctx, _ = policy_fixture(1)
    before = torch.get_rng_state().clone()
    with ctx:
        policy(torch.zeros(2, 3, dtype=torch.long), torch.Generator().manual_seed(0))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.xfail(reason="A3-2: net.train(was) overwrites mixed submodule modes", **DEFECT)
def test_policy_preserves_each_module_mode(tmp_path):
    ns, x = text_case(tmp_path)
    ns["net"].train()
    ns["net"].transformer.blocks[0].eval()
    before = [m.training for m in ns["net"].modules()]
    ns["policy_codes"](x, torch.Generator().manual_seed(0))
    assert before == [m.training for m in ns["net"].modules()]


def test_all_trainable_parameters_receive_onpolicy_gradients(tmp_path):
    """Torch 2.11 fixes the older no_grad/autocast weight-cache detachment bug."""
    ns, x = text_case(tmp_path)
    ns["net"].train()
    with ns["trainer"].ctx:
        loss, _ = ns["compute_loss"](x, torch.Generator().manual_seed(0))
    loss.backward()
    missing = [name for name, p in ns["net"].named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_val_nll_uses_canonical_targets_independent_of_arm(tmp_path):
    ns, x = text_case(tmp_path)
    ns["trainer"]._ctx_kw = None
    ns["net"].eval()
    with torch.no_grad():
        inputs, target = ns["tok"].get_ncm_inputs(x)
        logits = ns["net"](torch.cat([ns["prefix_emb"](len(x)), inputs], 1)).float()
        expected = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum").item() / len(x)
    control = ns["val_nll"]()["val_nll"]
    ns["corr"] = CorruptionCfg(mode="onpolicy", retarget=True)
    requant = ns["val_nll"]()["val_nll"]
    assert control == requant
    assert abs(control - expected) < 1e-3


def test_sample_evaluation_preserves_global_rng_and_tf32():
    def sampling(*args):
        torch.manual_seed(10000)
        return torch.rand(10)
    ns = definitions("scripts/train_gen.py", ["sample_metrics"], dict(
        torch=torch, device="cpu", _sample_metrics=sampling))
    state = torch.get_rng_state().clone()
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    ns["sample_metrics"](10, [10000])
    assert torch.equal(state, torch.get_rng_state())
    assert flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_identical_config_cpu_resume_is_exact(tmp_path):
    ns, x = text_case(tmp_path, passes=2)
    # CPU fp32, stochastic policy + dropout + corruption + NumPy data stream.
    ns["trainer"]._ctx_kw = None
    net, tr = ns["net"], ns["trainer"]
    gen, rng = torch.Generator().manual_seed(71), np.random.default_rng(73)
    def step():
        xb = x[rng.integers(0, len(x), size=2)]
        loss, _ = ns["compute_loss"](xb, gen)
        tr.backward_step(loss, list(net.parameters()))
        return float(loss.detach())
    step()
    tr.save({"model": net}, rng, {"corruption_rng": gen.get_state(),
                                  "policy_rng": ns["policy_gen"].get_state()})
    expected_loss = step()
    expected = copy.deepcopy(net.state_dict())
    expected_rng = torch.get_rng_state().clone(), gen.get_state().clone()
    extra = tr.maybe_resume({"model": net}, rng)
    gen.set_state(extra["corruption_rng"])
    ns["policy_gen"].set_state(extra["policy_rng"])
    assert step() == expected_loss
    assert all(torch.equal(expected[k], v) for k, v in net.state_dict().items())
    assert torch.equal(expected_rng[0], torch.get_rng_state())
    assert torch.equal(expected_rng[1], gen.get_state())


@pytest.mark.xfail(reason="A3-3: run key accepts a changed stochastic microbatch partition", **DEFECT)
def test_resume_key_rejects_changed_microbatch_partition():
    ns = definitions("scripts/train_gen.py", ["_run_key"], {"OmegaConf": OmegaConf})
    cfg = OmegaConf.load(ROOT / "configs/scale/L_ts_onpol.yaml")
    before = ns["_run_key"](cfg)
    cfg.train.grad_accum = 1
    assert before != ns["_run_key"](cfg)


@pytest.mark.xfail(reason="A3-3: a checkpoint without config skips identity validation", **DEFECT)
def test_resume_without_config_fails_closed(tmp_path):
    tree = ast.parse((ROOT / "scripts/train_gen.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.If)
                and "_run_key(" in ast.unparse(n))
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"checkpoint identity guard only")
    ns = definitions("scripts/train_gen.py", ["_run_key"], dict(
        OmegaConf=OmegaConf, os=os, sys=__import__("sys"), ckpt_path=str(ckpt),
        cfg_path=str(tmp_path / "config.yaml"),
        cfg=OmegaConf.load(ROOT / "configs/scale/L_ts_onpol.yaml"),
        a=SimpleNamespace(calib_steps=0, run_dir="audit-run")))
    refused = False
    try:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "scripts/train_gen.py", "exec"), ns)
    except (RuntimeError, SystemExit):
        refused = True
    assert refused


def test_microbatch_partition_changes_masks(released_sudoku):
    bundle, puzzles = released_sudoku
    x = puzzles[:2, 1:-1].contiguous()
    cfg = CorruptionCfg(mode="uniform")
    full = build_hierarchy(bundle.vqvae, x, cfg, generator=torch.Generator().manual_seed(7))
    gen = torch.Generator().manual_seed(7)
    parts = [build_hierarchy(bundle.vqvae, row[None], cfg, generator=gen)["used"] for row in x]
    assert not torch.equal(full["used"], torch.cat(parts))


@pytest.mark.xfail(reason="A3-3: Sudoku base checkpoint identity is only its byte size", **DEFECT)
def test_sudoku_manifest_rejects_same_size_base_change(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name in ["config.yaml", "tokenizer.pt", "generator.pt"]:
        (base / name).write_bytes(b"base")
    data = tmp_path / "data.npz"
    data.write_bytes(b"data")
    tree = ast.parse((ROOT / "scripts/train_sudoku_L.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "manifest" for t in n.targets))
    ns = dict(OmegaConf=OmegaConf, cfg=OmegaConf.create({"arm": "onpolicy"}),
              a=SimpleNamespace(data=str(data)), _base=str(base), os=os,
              digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
              pick_device=lambda: "cpu")
    code = compile(ast.Module(body=[node], type_ignores=[]), "scripts/train_sudoku_L.py", "exec")
    exec(code, ns)
    before = copy.deepcopy(ns["manifest"])
    (base / "tokenizer.pt").write_bytes(b"edit")
    exec(code, ns)
    assert before != ns["manifest"]


def test_sudoku_canonical_policy_has_identical_prefix_query_and_targets(released_sudoku):
    from ncmr.jacobi_train import parallel_logits
    bundle, puzzles = released_sudoku
    tok, net = bundle.vqvae, bundle.net.eval()
    prompt, sol = puzzles[:1], puzzles[:1, 1:-1]
    idx = tok.encode(sol)
    with torch.no_grad():
        reference = parallel_logits(net, tok.encoder.wte(prompt), idx)
        expected = F.cross_entropy(reference.float().flatten(0, 1), idx.flatten())
    ns = definitions("scripts/train_sudoku_L.py", ["wte", "canon", "tf_loss"], dict(
        torch=torch, F=F, net=net, tok=tok, dev="cpu", build_hierarchy=build_hierarchy,
        corr=CorruptionCfg(mode="onpolicy", mean_rate=1.0, spread=0),
        gen=torch.Generator().manual_seed(0),
        parallel_logits=lambda *a: F.one_hot(idx, num_classes=16).float()))
    with torch.no_grad():
        actual = ns["tf_loss"](prompt, sol)
    assert torch.equal(actual, expected)


@pytest.mark.xfail(reason="A3-3: missing corruption RNG is silently reseeded", **DEFECT)
def test_resume_without_corruption_rng_fails_closed():
    tree = ast.parse((ROOT / "scripts/train_gen.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "trainer.step")
    ns = dict(trainer=SimpleNamespace(step=250), resume_extra={},
              gen=torch.Generator(), seed=0)
    refused = False
    try:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "scripts/train_gen.py", "exec"), ns)
    except (RuntimeError, SystemExit):
        refused = True
    assert refused


@pytest.mark.xfail(reason="A3-4: notebook collection metadata has no code identity", **DEFECT)
def test_notebook_collection_carries_source_identity():
    tree = ast.parse((ROOT / "notebooks/make_notebooks.py").read_text())
    collect = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "COLLECT" for t in n.targets))
    payloads = []
    class JsonSpy:
        @staticmethod
        def dump(payload, stream):
            payloads.append(payload)
    class Open:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def close(self): pass
    # Execute only the metadata dump; no filesystem or notebook commands run.
    stmt = next(n for n in ast.parse(collect).body if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "json.dump")
    ns = dict(json=JsonSpy, open=lambda *a, **k: Open(), gpu="cpu", out="results",
              time=SimpleNamespace(ctime=lambda: "time"), CODE_SHA="audit-source",
              ATTEMPTS=[], datetime=datetime, timezone=timezone)
    exec(compile(ast.Module(body=[stmt], type_ignores=[]), "notebooks/make_notebooks.py", "exec"), ns)
    assert any(k in payloads[0] for k in ("git_sha", "code_sha", "source_revision"))
