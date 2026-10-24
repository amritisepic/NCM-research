"""J2-F (docs/preregistration.md): the frozen-teacher consistency arm of scripts/train_wmt14_L.py.

Registered code gate: (1) the teacher's fixed points are the released model's exact-decode outputs, (2) the teacher's
weights never change and receive no grad, (3) the executed generator proxy equals J2's, (4) the arm is in the run
manifest and a run directory refuses a different arm; plus audit 4's A4-3 padding invariance (R036).
"""
import ast
import copy
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import ncmr.jacobi_train as jt
from test_audit4_protocol import ROOT, WMT, definitions, known_trajectory, source, step, tiny_wmt

FROZEN_CFG = ROOT / "configs/scale/L_wmt_consist_frozen.yaml"


def frozen_copy(net):
    teacher = copy.deepcopy(net).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def captured_targets(ns, sources, teacher_net):
    """consist_loss's training targets (the teacher's fixed points, one row per sampled state)."""
    seen = []
    real_ce = ns["ce"]
    ns["ce"] = lambda lg, target: (seen.append(target.clone()), real_ce(lg, target))[1]
    try:
        ns["consist_loss"](len(sources) * ns["m_states"], sources, teacher_net)
    finally:
        ns["ce"] = real_ce
    return seen[0]


def fixed_points(net, ns, sources):
    z, pm = ns["prefix"](sources)
    with torch.no_grad():
        out = jt.jacobi_decode(net.eval(), lambda c, h: z.float(), len(sources), ns["S"], "cpu", ns["K"] + 1,
                               pad_mask=pm, keep_history=True, full_budget=True, initial_fixed_ok=True)
    return out["codes"]


# (1) J2-F's targets are the released model's fixed points, also after the student has moved; J2's follow the student
def test_frozen_targets_are_released_fixed_points_after_the_student_moves(tmp_path):
    ns = tiny_wmt(tmp_path)
    net = ns["net"]
    released = copy.deepcopy(net)
    teacher = frozen_copy(net)
    src = torch.from_numpy(ns["tr_src"][:4].astype("int64"))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    moved = copy.deepcopy(net)
    net.train()
    m = ns["m_states"]
    frozen_t = captured_targets(ns, src, teacher)
    moving_t = captured_targets(ns, src, None)
    assert torch.equal(frozen_t, fixed_points(released, ns, src).repeat_interleave(m, 0))
    assert torch.equal(moving_t, fixed_points(moved, ns, src).repeat_interleave(m, 0))
    assert not torch.equal(frozen_t, moving_t)
    assert net.training and not teacher.training


# (2) the teacher never changes and gets no gradient, while the student trains; the loop passes the teacher through
def test_frozen_teacher_weights_never_change_and_get_no_grad(tmp_path):
    ns = tiny_wmt(tmp_path)
    ns["teacher"] = frozen_copy(ns["net"])
    before_t = {k: v.clone() for k, v in ns["teacher"].state_dict().items()}
    before_s = {k: v.clone() for k, v in ns["net"].state_dict().items()}
    calls = []
    real = ns["consist_loss"]
    ns["consist_loss"] = lambda n, s, t=None: (calls.append(t), real(n, s, t))[1]
    for _ in range(2):
        step(ns)
    assert calls and all(t is ns["teacher"] for t in calls)
    assert all(p.grad is None and not p.requires_grad for p in ns["teacher"].parameters())
    assert all(torch.equal(before_t[k], v) for k, v in ns["teacher"].state_dict().items())
    assert any(not torch.equal(before_s[k], v) for k, v in ns["net"].state_dict().items())


def production_teacher(net, arm="consist_frozen"):
    """Run the trainer's own teacher-construction block."""
    tree = ast.parse(source(WMT))
    i = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "arm == 'consist_frozen'")
    ns = dict(copy=copy, net=net, arm=arm)
    exec(compile(ast.Module(body=tree.body[i - 1:i + 1], type_ignores=[]), WMT, "exec"), ns)
    return ns["teacher"]


def test_production_teacher_is_a_frozen_eval_copy(tmp_path):
    ns = tiny_wmt(tmp_path)
    net = ns["net"].train()
    teacher = production_teacher(net)
    assert production_teacher(net, arm="consist") is None
    assert teacher is not net and not teacher.training and net.training
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not {id(p) for p in teacher.parameters()} & {id(p) for p in net.parameters()}
    assert all(torch.equal(a, b) for a, b in zip(teacher.state_dict().values(), net.state_dict().values()))


def test_teacher_is_copied_before_resume_and_is_not_checkpointed():
    tree = ast.parse(source(WMT))
    body = [ast.unparse(n) for n in tree.body]
    copy_at = next(i for i, s in enumerate(body) if "copy.deepcopy(net)" in s)
    resume_at = next(i for i, s in enumerate(body) if "trainer.maybe_resume" in s)
    assert copy_at < resume_at
    assert next(s for s in body if s.startswith("modules =")) == "modules = {'model': net}"


# (3) identical executed generator rows for J2 and J2-F, and the same registered proxy
@pytest.mark.parametrize("degenerate", [0, 1, 16])
def test_frozen_teacher_executes_the_registered_rows(monkeypatch, degenerate):
    rows = {}
    for arm in ("consist", "consist_frozen"):
        ns, calls, _ = known_trajectory(monkeypatch)
        sources = torch.ones(16, 1, dtype=torch.long)
        sources[:degenerate] = 0
        teacher = torch.nn.Linear(1, 1).eval() if arm == "consist_frozen" else None
        ns["consist_loss"](64, sources, teacher)
        rows[arm] = 3 * (64 + sum(c["rows"] for c in calls[12:])) + sum(c["rows"] for c in calls[:12])
        assert all(not c["grad"] and not c["amp"] and not c["training"] for c in calls[:12])
    assert rows["consist"] == rows["consist_frozen"] == 576


def test_frozen_arm_gets_the_consistency_half_batch_and_proxy():
    tree = ast.parse(source(WMT))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and ast.unparse(n).startswith("bs ="))
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and ast.unparse(n).startswith("fl_step ="))
    program = compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), WMT, "exec")
    out = {}
    for arm in ("ctrl", "consist", "consist_frozen"):
        ns = dict(tc=SimpleNamespace(batch_size=128), arm=arm, m_states=4, K=11, fwd=1)
        exec(program, ns)
        out[arm] = (ns["n_c"], ns["fl_step"])
    assert out["consist"] == out["consist_frozen"] == (64, 3 * 128 + 12 * 16)
    assert out["ctrl"] == (0, 3 * 128)
    with pytest.raises(ValueError):
        exec(program, dict(tc=SimpleNamespace(batch_size=128), arm="consist_frozn", m_states=4, K=11, fwd=1))


# (4) the arm is in the run manifest: a J2-F run directory refuses J2's config and vice versa
@pytest.mark.parametrize("first,second", [("consist_frozen", "consist"), ("consist", "consist_frozen")])
def test_run_directory_refuses_the_other_consistency_arm(tmp_path, first, second):
    base, run = tmp_path / "artifacts/wmt14", tmp_path / "run"
    base.mkdir(parents=True)
    run.mkdir()
    (tmp_path / "data.npz").write_bytes(b"training archive")
    (base / "config.yaml").write_text("original config", encoding="utf-8")
    for name in ["tokenizer.pt", "generator.pt"]:
        buf = io.BytesIO()
        torch.save(dict(model=dict(weight=torch.tensor([1., 2.]))), buf)
        (base / name).write_bytes(buf.getvalue())
    cfg = OmegaConf.load(FROZEN_CFG)
    cfg.arm = first
    ns = dict(ROOT=str(tmp_path), os=os, json=json, sys=sys, OmegaConf=OmegaConf,
              a=SimpleNamespace(run_dir=str(run), data=str(tmp_path / "data.npz"), eval_only=False, calib_steps=0),
              cfg=cfg, dev="cpu", ckpt_path=str(run / "ckpt.pt"), manifest_path=str(run / "run_manifest.json"),
              digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    tree = ast.parse(source(WMT))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.If) and
                 ast.unparse(n.test) == "a.eval_only and os.path.exists(ckpt_path)")
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and
               isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "OmegaConf.save")
    program = compile(ast.Module(body=tree.body[start:end], type_ignores=[]), WMT, "exec")
    exec(program, ns)
    assert json.loads((run / "run_manifest.json").read_text())["config"]["arm"] == first
    Path(ns["ckpt_path"]).write_bytes(b"checkpoint exists")
    exec(program, ns)                      # same arm: resume allowed
    ns["cfg"].arm = second
    with pytest.raises(SystemExit):
        exec(program, ns)


def test_frozen_config_differs_from_j2_only_in_the_arm():
    j2 = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/scale/L_wmt_consist.yaml"))
    j2f = OmegaConf.to_container(OmegaConf.load(FROZEN_CFG))
    assert j2.pop("arm") == "consist" and j2f.pop("arm") == "consist_frozen"
    assert j2 == j2f


# A4-3 (R036 test gap): zero-loss padding rows of degenerate sources leave the real rows' loss and gradients unchanged
@pytest.mark.parametrize("frozen", [False, True])
def test_padding_rows_leave_real_rows_loss_and_gradient_unchanged(tmp_path, frozen):
    ns = tiny_wmt(tmp_path)
    net = ns["net"].eval()                 # dropout off, so the comparison is exact up to batch-shape numerics
    teacher = frozen_copy(net) if frozen else None
    net.train = lambda mode=True: net      # keep the student in eval mode through consist_loss (after the copy)
    src = torch.from_numpy(ns["tr_src"][:4].astype("int64"))
    real_decode = ns["jacobi_decode"]

    def decode_with_degenerate(rows):
        def decode(*args, **kw):
            out = real_decode(*args, **kw)
            for r in rows:                 # all-zero fixed point from the start: J1b's degenerate source
                out["codes"][r] = 0
                for h in out["history"]:
                    h[r] = 0
            return out
        return decode

    def loss_and_grads(sources, degenerate_rows):
        ns["jacobi_decode"] = decode_with_degenerate(degenerate_rows)
        net.zero_grad()
        torch.manual_seed(0)               # the state draws: same uniform stream for the kept sources
        loss, used = ns["consist_loss"](len(sources) * ns["m_states"], sources, teacher)
        loss.backward()
        return loss.detach(), used, {k: p.grad.clone() for k, p in net.named_parameters() if p.grad is not None}

    padded = loss_and_grads(src, degenerate_rows=[3])          # last source degenerate: padded rows appended
    plain = loss_and_grads(src[:3], degenerate_rows=[])         # the same three real sources, no padding
    ns["jacobi_decode"] = real_decode
    assert padded[1] == plain[1] == 3 * ns["m_states"]
    assert torch.allclose(padded[0], plain[0], rtol=1e-5, atol=1e-6)
    assert padded[2].keys() == plain[2].keys()
    for k in plain[2]:
        assert torch.allclose(padded[2][k], plain[2][k], rtol=1e-4, atol=1e-6), k
