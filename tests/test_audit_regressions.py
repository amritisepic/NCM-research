"""Regression tests for codex-audit findings F2, F3, F7 (fixed in R024; they were strict xfails before).

The policy helper is extracted from the actual script's AST so argparse, model
loading and the training loop do not execute. Only its external models are tiny
stubs.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ncmr.corruption import CorruptionCfg, build_hierarchy
from ncmr.train import fp32_eval
from test_audit_hierarchy import tiny_text_tokenizer


def policy_fixture(passes=2):
    source = Path(__file__).resolve().parents[1] / "scripts/train_gen.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "policy_codes")
    seen = []

    def inputs(codes):
        seen.append(codes.clone())
        return torch.zeros(codes.shape[0], 2, 1)

    class Net(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2, 2)

    ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    net = Net()
    ns = dict(torch=torch, net=net, fnet=net, device="cpu", policy_passes=passes, fp32_eval=fp32_eval,
              policy_temp=1.0, l0=1, L0=torch.tensor([0.0, 1.0]),
              trainer=SimpleNamespace(ctx=ctx), prefix_emb=lambda b: torch.zeros(b, 1, 1),
              tok=SimpleNamespace(_encode_pre_quant=lambda x: x.float(), vq=SimpleNamespace(
                  z_to_idx=lambda x: torch.zeros(x.shape[0], 3, dtype=torch.long), idx_to_ncm_input=inputs)))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "scripts/train_gen.py", "exec"), ns)
    return ns["policy_codes"], ctx, seen


def test_policy_does_not_leak_cpu_autocast():
    policy, ctx, _ = policy_fixture(1)
    assert not torch.is_autocast_enabled("cpu")
    try:
        with ctx:
            policy(torch.zeros(2, 3, dtype=torch.long), torch.Generator().manual_seed(0))
        assert not torch.is_autocast_enabled("cpu")
    finally:
        # The reproducer intentionally exposes a leak; leave no state for other tests.
        torch.set_autocast_enabled("cpu", False)


def test_second_policy_pass_uses_sampled_level_zero():
    policy, ctx, seen = policy_fixture(2)
    try:
        with ctx:
            result = policy(torch.zeros(2, 3, dtype=torch.long), torch.Generator().manual_seed(0))
        assert len(seen) == 2
        assert torch.equal(seen[1][:, :1], result[:, :1])
    finally:
        torch.set_autocast_enabled("cpu", False)


def test_soft_labels_do_not_depend_on_microbatch_partition():
    """With the frozen per-level gap that train_gen.py calibrates (W1b), targets are partition-invariant."""
    tok = tiny_text_tokenizer(seed=42)
    x = torch.randint(0, 32, (2, 128))
    gaps = build_hierarchy(tok, x, CorruptionCfg(mode="none", soft_tau=0.25, soft_topm=16))["soft_gap_per_level"]
    cfg = CorruptionCfg(mode="none", soft_tau=0.25, soft_topm=16, soft_gap=tuple(gaps))
    full = build_hierarchy(tok, x, cfg)
    separate = [build_hierarchy(tok, row[None], cfg) for row in x]
    assert torch.equal(full["soft_idx"], torch.cat([h["soft_idx"] for h in separate]))
    # fp32 rounding of batched distances is ~3e-6; the old batch-mean gap differed by ~5e-2 here
    torch.testing.assert_close(full["soft_w"], torch.cat([h["soft_w"] for h in separate]), rtol=0, atol=1e-5)


def test_trainer_ctx_nesting_does_not_leak_autocast(tmp_path):
    """F2 at the source: every access to Trainer.ctx is a fresh autocast object, so nesting is safe."""
    from ncmr.train import Trainer
    tr = Trainer(str(tmp_path), [torch.nn.Parameter(torch.zeros(2, 2))], max_lr=1e-3, total_steps=1, device="cpu")
    assert not torch.is_autocast_enabled("cpu")
    try:
        with tr.ctx:
            with tr.ctx:
                pass
            assert torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cpu")
    finally:
        torch.set_autocast_enabled("cpu", False)
