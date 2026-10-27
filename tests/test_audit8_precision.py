"""Audit 8: execute production precision boundaries. Ported from `codex/audit8` (c2dca74): the A8-1 and A8-2 tests
were strict xfails there and pass once fixed (docs/reviews/triage_audit8_2026-10-27.md).

No downloads. The released-tokenizer probe uses conftest's hash-pinned fixture.
The tiny scorer replaces external weights, retaining the production MAUVE path.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ncmr import eval_text
from ncmr.w2m import MechArm
from w2m_support import make_arm, run_nodes, script_nodes, tiny_case

@pytest.mark.parametrize("legacy_cache", [False, True])
def test_w2m_mauve_forwards_are_fp32_and_do_not_reuse_amp_references(tmp_path, monkeypatch, legacy_cache):
    seen, refs = [], []

    class Tokenizer:
        eos_token, eos_token_id = "eos", 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2]}

    class Scorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, ids, **kwargs):
            h = self.lin(torch.ones(*ids.shape, 4))
            seen.append((torch.is_autocast_enabled("cpu"), h.dtype))
            return SimpleNamespace(hidden_states=[h])

    lm = Scorer().eval()
    monkeypatch.setattr(eval_text, "_load_lm", lambda *a: (Tokenizer(), lm))
    monkeypatch.setattr(eval_text, "gen_ppl", lambda *a, **k: 1.0)

    def mauve(p, q, **kwargs):
        refs.append(q.copy())
        return 0.5

    monkeypatch.setattr(eval_text, "mauve_score", mauve)
    if legacy_cache:
        cache = tmp_path / "runs/cache/mauve_ref_val_2.npy"
        cache.parent.mkdir(parents=True)
        np.save(cache, np.full((2, 4), 999.0))  # sentinel for the old AMP reference
    ns = dict(os=os, np=np, ROOT=str(tmp_path), device="cpu", corr=SimpleNamespace(mode="mech"),
              draw_samples=lambda *a: ([[2, 3], [3, 4]], 11, 0.0),
              bpe=SimpleNamespace(encode=lambda s: [int(x) for x in s.split()],
                                  decode=lambda ids: " ".join(map(str, ids))),
              examples=lambda *a: [[2, 3], [3, 4]], cut_at_eos=lambda r: r)
    funcs, _, _ = script_nodes()
    run_nodes([funcs["_sample_metrics"]], ns)
    ns["_sample_metrics"](2, [42], mauve=True)
    assert seen and all(not amp and dtype == torch.float32 for amp, dtype in seen), seen
    assert refs and not any(np.any(q == 999.0) for q in refs)


def test_canonical_teacher_disables_tf32_and_restores_the_caller(released_sudoku, monkeypatch):
    bundle, puzzles = released_sudoku
    tok, net = bundle.vqvae, bundle.net
    seen = []
    encode, quantize = tok._encode_pre_quant, tok.vq.z_to_idx

    def flags():
        return (torch.is_autocast_enabled("cpu"), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32, torch.is_grad_enabled())

    def encoding(x):
        seen.append(flags())
        return encode(x)

    def quantization(z):
        seen.append(flags())
        return quantize(z)

    monkeypatch.setattr(tok, "_encode_pre_quant", encoding)
    monkeypatch.setattr(tok.vq, "z_to_idx", quantization)
    v = int(tok.vq.vqvae_vocab_sizes[0])
    m = MechArm("m0", tok, net, torch.ones(v) / v, None, "cpu", 0,
                [0.0] * len(tok.vq.levels))
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            codes = m.canonical(puzzles[:2, 1:-1].contiguous())
            assert torch.is_autocast_enabled("cpu")
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert codes.shape[0] == 2 and seen
        assert seen == [(False, False, False, False)] * 2, seen
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


@pytest.mark.parametrize("arm", ["m1", "m2", "m6"])
@pytest.mark.parametrize("compiled", [False, True])
def test_actual_proposal_forwards_restore_amp_modes_rng_and_gradients(arm, compiled):
    tok, net = tiny_case()
    net.train()
    net.transformer.blocks[0].mlp.eval()  # restoration must preserve mixed submodule modes
    m = make_arm(arm, tok, net)
    if compiled:
        # Exercises OptimizedModule dispatch on CPU without requiring a platform C++ compiler.
        m.fwd = torch.compile(net, backend="eager")
    proposer = m.frozen if arm == "m2" else net
    seen = []
    hook = proposer.register_forward_hook(lambda mod, inp, out: seen.append(
        (out.dtype, torch.is_autocast_enabled("cpu"), torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32, torch.is_grad_enabled(), any(q.training for q in mod.modules()))))
    x = torch.randint(0, 32, (2, 128), generator=torch.Generator().manual_seed(9))
    canon, plan = m.canonical(x), m.draw_plan(2)
    modes = [q.training for q in net.modules()]
    rng = torch.get_rng_state().clone()
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            m.proposals(canon, plan)
            assert torch.is_autocast_enabled("cpu")
            # An actual student backward in the SAME AMP region catches detached cached casts.
            inp = tok.vq.idx_to_ncm_input(canon)
            loss = m.fwd(torch.cat([m.prefix_emb(2), inp], 1)).float().square().mean()
            loss.backward()
        assert seen[0] == (torch.float32, False, False, False, False, False), seen
        assert [q.training for q in net.modules()] == modes
        assert not torch.is_autocast_enabled("cpu")
        assert net.transformer.wte.weight.grad is not None
        assert torch.isfinite(net.transformer.wte.weight.grad).all()
        assert float(net.transformer.wte.weight.grad.abs().sum()) > 0
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        # Student dropout consumes RNG, so measure proposal isolation separately.
        torch.set_rng_state(rng)
        before = torch.get_rng_state().clone()
        m.proposals(canon, plan)
        assert torch.equal(before, torch.get_rng_state())
    finally:
        hook.remove()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
