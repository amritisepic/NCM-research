"""Audit 4 CPU probes; default to the working tree so proposed fixes can XPASS.

AUDIT4_REF=c91126a pins the original subject. No CLI training or downloads.
Only assertion failures representing demonstrated defects are strict xfails.
"""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ncmr import jacobi_train as jt
from ncmr.models import generator_config
from ncmr.ncm_ref_loader import NextConceptTransformer, VQVAE
from ncmr.train import Trainer

ROOT = Path(__file__).resolve().parents[1]
WMT = "scripts/train_wmt14_L.py"
DEFECT = dict(strict=True, raises=AssertionError)


def source(path):
    ref = os.environ.get("AUDIT4_REF")
    if ref:
        return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT).decode()
    return (ROOT / path).read_text(encoding="utf-8")


def definitions(path, names, namespace):
    tree = ast.parse(source(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def jacobi_module():
    module = ModuleType("audit4_jacobi")
    exec(compile(source("src/ncmr/jacobi_train.py"), "src/ncmr/jacobi_train.py", "exec"), module.__dict__)
    return module


def known_trajectory(monkeypatch, m=4):
    module = jacobi_module()
    calls, training_inputs = [], []

    def logits(net, z, codes, pad_mask=None):
        calls.append(dict(rows=len(codes), grad=torch.is_grad_enabled(),
                          amp=torch.is_autocast_enabled("cpu"), training=net.training))
        bit = z[:, 0, 0].long().reshape(-1, 1)
        new = torch.cat([bit, codes[:, :-1]], 1)
        return F.one_hot(new, 2).float() * 10

    module.parallel_logits = logits

    def training_logits(net, z, codes, pad_mask=None):
        training_inputs.append(codes.clone())
        return logits(net, z, codes, pad_mask)

    ns = dict(torch=torch, net=torch.nn.Linear(1, 1), m_states=m, S=11, K=11, dev="cpu",
              prefix=lambda src: (src.float()[:, :, None], torch.zeros_like(src, dtype=torch.bool)),
              jacobi_decode=module.jacobi_decode, parallel_logits=training_logits,
              traj_stats=dict(passes=0., n=0, states_before_fp=0., degenerate=0),
              ce=lambda lg, target: target.clone())
    definitions(WMT, ["consist_loss"], ns)
    return ns, calls, training_inputs


def test_j1b_known_trajectory_indexing_uniform_support_fp32_and_full_budget(monkeypatch):
    ns, calls, seen = known_trajectory(monkeypatch, m=11)
    # Stratified draws cover every interval in [0,1), making every pre-FP index observable.
    monkeypatch.setattr(torch, "rand", lambda g, m, **kw: ((torch.arange(m) + .5) / m).repeat(g, 1))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        targets, used = ns["consist_loss"](22, torch.tensor([[0], [1]]))
    assert used == 11 and targets.eq(1).all()
    expected = torch.tensor([[1] * t + [0] * (11 - t) for t in range(11)])
    assert torch.equal(seen[0][:used], expected)  # input of pass t+1, including all-zero t=0
    assert len(calls) == 13 and all(c["rows"] == 2 for c in calls[:12])
    assert all(not c["amp"] and not c["grad"] and not c["training"] for c in calls[:12])
    assert calls[-1]["amp"] and calls[-1]["training"]
    # ported: the run-level totals (run_sources / run_degenerate, never reset) are additions; the interval stats are
    assert {k: v for k, v in ns["traj_stats"].items() if k != "passes" and not k.startswith("run_")} == dict(
        n=2, states_before_fp=11., degenerate=1)
    assert (ns["traj_stats"]["run_sources"], ns["traj_stats"]["run_degenerate"]) == (2, 1)
    assert not torch.is_autocast_enabled("cpu")


def test_j1b_random_draws_are_uniform_over_strictly_pre_fixed_states(monkeypatch):
    ns, _, seen = known_trajectory(monkeypatch)
    torch.manual_seed(173)
    _, used = ns["consist_loss"](40000, torch.ones(10000, 1, dtype=torch.long))
    counts = torch.bincount(seen[0].sum(1), minlength=12)
    assert used == 40000 and counts[11] == 0
    assert (counts[:11] - 40000 / 11).abs().max() < 230


def test_registered_loop_draws_128_rows_and_uses_rows_65_69_through_125(monkeypatch):
    tree = ast.parse(source(WMT))
    loop = next(n for n in tree.body if isinstance(n, ast.While))
    node = loop.body[0]
    data_calls, tf_rows, source_rows = [], [], []
    def batch(n):
        data_calls.append(n)
        x = torch.arange(1, n + 1).reshape(-1, 1)
        return x, x
    def tf_loss(src, tgt):
        tf_rows.extend(src[:, 0].tolist())
        return torch.tensor(1.)
    def consist(n, src):
        source_rows.extend(src[:, 0].tolist())
        return torch.tensor(2.), n
    ns = dict(trainer=SimpleNamespace(ctx=torch.autocast("cpu", enabled=False)), batch=batch,
              bs=128, n_c=64, m_states=4, tf_loss=tf_loss, consist_loss=consist)
    exec(compile(ast.Module(body=[node], type_ignores=[]), WMT, "exec"), ns)
    assert data_calls == [128] and tf_rows == list(range(1, 65))
    assert source_rows == list(range(65, 129, 4))


def tiny_wmt(tmp_path, amp=True):
    """Actual upstream NCM/VQ classes, WMT-style prefix L0 head, 11 levels, 64-token targets."""
    torch.manual_seed(42)
    levels = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    tok = VQVAE(in_vocab_size=32, context_length=64, embed_dim=8, vqvae_vocab_size=[16] * 11,
                n_layers=1, compression_factor=1, dropout=0, pre_quant_groupnorm=2,
                vector_quantizer_config=OmegaConf.create(dict(
                    clss="multiscale_residual_vector_quantizer", decay=.99, commitment_cost=.25,
                    quant_resi=dict(enabled=True, ratio=.5, share_mode=0),
                    levels=dict(use_manual_levels=True, manual_levels=levels)))).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    cfg = generator_config(tok, n_layer=1, n_head=2, n_embd=32, dropout=.1, prefix_len=4)
    cfg.predict_level0 = True
    cfg.prefix_pad_token_id = 31
    net = NextConceptTransformer(tok, cfg).train()
    trainer = Trainer(str(tmp_path), net.parameters(), max_lr=5e-5, total_steps=4, warmup=2,
                      device="cpu", amp=amp, weight_decay=.01, betas=(.9, .95))
    rng = np.random.default_rng(8)
    data = np.random.default_rng(11)
    src = data.integers(1, 29, (64, 4), dtype=np.int32)
    src[::2, -2:] = 31
    tgt = data.integers(1, 29, (64, 64), dtype=np.int32)
    tgt[:, 12] = 30
    tgt[:, 13:] = 31
    ns = dict(torch=torch, np=np, F=F, tok=tok, net=net, trainer=trainer, rng=rng, dev="cpu",
              tr_src=src, tr_tgt=tgt, D=dict(val_src=src[:4], val_tgt=tgt[:4]), PAD=31, EOS=30,
              K=11, S=sum(levels), P=4, m_states=4, bs=8, n_c=4,
              params=list(net.parameters()),
              parallel_logits=jt.parallel_logits, jacobi_decode=jt.jacobi_decode,
              traj_stats=dict(passes=0., n=0, states_before_fp=0., degenerate=0))
    definitions(WMT, ["prefix", "canon", "ce", "batch", "tf_loss", "consist_loss", "val_nll"], ns)
    return ns


def step(ns, optimize=True):
    loop = next(n for n in ast.parse(source(WMT)).body if isinstance(n, ast.While))
    # Execute the actual loss block and backward; optionally the optimizer call too.
    exec(compile(ast.Module(body=loop.body[:3 if optimize else 2], type_ignores=[]), WMT, "exec"), ns)
    return ns["loss"].item()


def test_every_trainable_parameter_gets_finite_nonzero_gradient_in_j2(tmp_path):
    ns = tiny_wmt(tmp_path)
    observed = []
    handle = ns["net"].register_forward_pre_hook(lambda net, args: observed.append(
        (torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"), net.training)))
    step(ns, optimize=False)
    handle.remove()
    params = dict(ns["net"].named_parameters())
    missing = [name for name, p in params.items() if p.requires_grad and
               (p.grad is None or not p.grad.isfinite().all() or not p.grad.abs().sum() > 0)]
    assert not missing, missing
    assert len(observed) == 14  # TF + exactly 12 teacher passes + consistency
    assert observed[0] == observed[-1] == (True, True, True)
    assert all(v == (False, False, False) for v in observed[1:-1])
    assert not torch.is_autocast_enabled("cpu")


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_j2_resume_preserves_weights_optimizer_data_sampling_and_dropout_rng(tmp_path):
    ns = tiny_wmt(tmp_path / "original")
    step(ns)
    ns["trainer"].save(dict(model=ns["net"]), ns["rng"])
    expected_losses = [step(ns), step(ns)]
    expected = dict(weights=copy.deepcopy(ns["net"].state_dict()),
                    opt=copy.deepcopy(ns["trainer"].opt.state_dict()),
                    data=copy.deepcopy(ns["rng"].bit_generator.state), torch=torch.get_rng_state())
    resumed = tiny_wmt(tmp_path / "original")  # deliberately consumes randomness before restore
    resumed["trainer"].maybe_resume(dict(model=resumed["net"]), resumed["rng"])
    resumed["val_nll"]()  # an additional validation call must not change training randomness
    assert [step(resumed), step(resumed)] == expected_losses
    actual = dict(weights=resumed["net"].state_dict(), opt=resumed["trainer"].opt.state_dict(),
                  data=resumed["rng"].bit_generator.state, torch=torch.get_rng_state())
    assert_nested_equal(actual, expected)


# fixed (was a strict xfail: A4-1: WMT val_nll uses the training autocast)
def test_val_nll_generator_forwards_are_fp32(tmp_path):
    ns = tiny_wmt(tmp_path)
    observed = []
    hook = ns["net"].register_forward_pre_hook(lambda net, args: observed.append(
        torch.is_autocast_enabled("cpu")))
    ns["val_nll"]()
    hook.remove()
    assert observed and not any(observed)


def manifest_case(tmp_path):
    base, run = tmp_path / "artifacts/wmt14", tmp_path / "run"
    base.mkdir(parents=True)
    run.mkdir()
    (tmp_path / "data.npz").write_bytes(b"training archive")
    (base / "config.yaml").write_text("original config", encoding="utf-8")
    for name in ["tokenizer.pt", "generator.pt"]:
        buf = io.BytesIO()
        torch.save(dict(model=dict(weight=torch.tensor([1., 2.]))), buf)
        (base / name).write_bytes(buf.getvalue())
    cfg = OmegaConf.load(ROOT / "configs/scale/L_wmt_consist.yaml")
    ns = dict(ROOT=str(tmp_path), os=os, json=json, sys=sys, OmegaConf=OmegaConf,
              a=SimpleNamespace(run_dir=str(run), data=str(tmp_path / "data.npz"), eval_only=False, calib_steps=0),
              cfg=cfg, dev="cpu", ckpt_path=str(run / "ckpt.pt"),
              manifest_path=str(run / "run_manifest.json"),
              digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    tree = ast.parse(source(WMT))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.If) and
                 ast.unparse(n.test) == "a.eval_only and os.path.exists(ckpt_path)")
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and
               isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "OmegaConf.save")
    program = compile(ast.Module(body=tree.body[start:end], type_ignores=[]), WMT, "exec")
    exec(program, ns)
    Path(ns["ckpt_path"]).write_bytes(b"checkpoint exists")
    return ns, program, base


@pytest.mark.parametrize("changed", ["arm", "config", "data", "base_config", "eval_only"])
def test_manifest_refuses_changed_registered_inputs_and_eval_only(tmp_path, changed):
    ns, program, base = manifest_case(tmp_path)
    if changed == "arm":
        ns["cfg"].arm = "ctrl"
    elif changed == "config":
        ns["cfg"].train.lr *= 2
    elif changed == "data":
        Path(ns["a"].data).write_bytes(b"other archive")
    elif changed == "base_config":
        (base / "config.yaml").write_text("other config", encoding="utf-8")
    else:
        ns["a"].eval_only = True
    with pytest.raises(SystemExit):
        exec(program, ns)


@pytest.mark.parametrize("name", ["tokenizer.pt", "generator.pt"])
# fixed (was a strict xfail: A4-2: manifest compares checkpoint sizes, not contents)
def test_manifest_refuses_changed_same_size_base_weights(tmp_path, name):
    ns, program, base = manifest_case(tmp_path)
    path = base / name
    original = path.read_bytes()
    buf = io.BytesIO()
    torch.save(dict(model=dict(weight=torch.tensor([3., 4.]))), buf)
    changed = buf.getvalue()
    assert changed != original and len(changed) == len(original)
    path.write_bytes(changed)
    rejected = False
    try:
        exec(program, ns)
    except (SystemExit, ValueError):
        rejected = True
    assert rejected, "different same-size base weights were accepted"


@pytest.mark.parametrize("degenerate", [
    0,
    1,   # fixed (was a strict xfail: A4-3: dropped rows still counted as backward work)
    16,   # fixed (was a strict xfail: A4-3: all-degenerate step claims unused work)
])
def test_realized_generator_rows_match_registered_proxy(monkeypatch, degenerate):
    ns, calls, _ = known_trajectory(monkeypatch)
    sources = torch.ones(16, 1, dtype=torch.long)
    sources[:degenerate] = 0
    _, used = ns["consist_loss"](64, sources)
    teacher_rows = sum(c["rows"] for c in calls[:12])
    training_rows = sum(c["rows"] for c in calls[12:])
    actual = 3 * (64 + training_rows) + teacher_rows
    assert actual == 576, dict(actual=actual, registered=576, states_used=used)


def test_exact_jacobi_matches_sequential_for_padded_eos_sources_and_saves_caps(tmp_path):
    ns = tiny_wmt(tmp_path)
    net, tok = ns["net"].eval(), ns["tok"]
    src = torch.from_numpy(ns["tr_src"][:4].astype(np.int64))
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        seq = net.generate(None, src, temperature=0., top_k=0, top_p=1., cfg_scale=0., cfg_null_eos=30)
        z, pm = ns["prefix"](src)
        out = jt.jacobi_decode(net, lambda c, h: z, len(src), ns["S"], "cpu", 12,
                               pad_mask=pm, keep_history=True, full_budget=True)
        assert torch.equal(seq, out["codes"])
        for cap in range(1, 13):
            fresh = jt.jacobi_decode(net, lambda c, h: z, len(src), ns["S"], "cpu", cap, pad_mask=pm)
            assert torch.equal(out["history"][cap - 1], fresh["codes"])
            assert torch.equal(out["nfe"].clamp(max=cap), fresh["nfe"])


def test_evaluation_and_final_json_keep_ordered_all_cap_hypotheses_and_w0_bleu(tmp_path):
    import sacrebleu
    ns = tiny_wmt(tmp_path)
    ns.update(vq=ns["tok"].vq, pt=SimpleNamespace(
        encode=lambda text: [int(t) for t in text.split()],
        decode=lambda ids, **kw: " ".join(f"word{t}" for t in ids)))
    definitions(WMT, ["detok", "evaluate"], ns)
    srcs = ["1 2", "3 30 4", "5 6 7 8 9"]
    refs = ["word1 word2 word3 word4", "word5 word6 word7 word8", "word9 word10 word11 word12"]
    observed = []
    hook = ns["net"].register_forward_pre_hook(lambda net, args: observed.append(
        (torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"))))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        metrics, hyp, capped = ns["evaluate"](srcs, refs, bs_=2)
    hook.remove()
    assert all(v == (False, False) for v in observed)
    assert metrics["fixed_point_equals_sequential_pct"] == 100
    assert metrics["fixed_point_differs_from_sequential_idx"] == []
    assert list(capped) == list(range(1, 13)) and all(len(h) == 3 for h in capped.values())
    for cap, h in capped.items():
        score = sacrebleu.corpus_bleu(h, [refs], lowercase=True)
        actual = metrics["capped"][str(cap)]
        assert actual["bleu_lc"] == score.score
        assert actual["len_ratio"] == score.sys_len / score.ref_len
        assert actual["passes_mean"] == float(np.minimum(metrics["nfe_per_sentence"], cap).mean())
    counts = metrics["nfe_per_sentence"][:]
    saves = [n for n in ast.parse(source(WMT)).body if isinstance(n, ast.Expr) and
             isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "json.dump" and
             any(name in ast.unparse(n) for name in ["hyp_test.json", "hyp_val.json"])]
    env = dict(json=json, os=os, a=SimpleNamespace(run_dir=str(tmp_path)),
               hyp=hyp, hyp_cap=capped, hyp_val=hyp, hyp_val_cap=capped,
               final=dict(test=copy.deepcopy(metrics), val=copy.deepcopy(metrics)))
    exec(compile(ast.Module(body=saves, type_ignores=[]), WMT, "exec"), env)
    for split in ["test", "val"]:
        saved = json.loads((tmp_path / f"hyp_{split}.json").read_text())
        assert saved[f"hyp_{split}"] == hyp
        assert saved[f"hyp_{split}_capped"] == {str(c): h for c, h in capped.items()}
        assert saved[f"nfe_{split}"] == counts


def test_j2_target_is_source_determined_at_every_code_position(tmp_path):
    ns = tiny_wmt(tmp_path)
    ns["net"].eval()
    # Identical visible source, including an EOS and right padding, repeated in one batch.
    src = torch.tensor([[7, 30, 31, 31]]).repeat(4, 1)
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        z, pm = ns["prefix"](src)
        out = jt.jacobi_decode(ns["net"], lambda c, h: z, 4, ns["S"], "cpu", 12,
                               pad_mask=pm, keep_history=True, full_budget=True)
        target = out["codes"]
        assert all(torch.equal(row, target[0]) for row in target)
        assert out["stop"].eq(1).all()
        # Replaying the fixed point preserves all positions, including codes representing EOS/padding.
        assert torch.equal(jt.parallel_logits(ns["net"], z, target, pm).argmax(-1), target)


# fixed (was a strict xfail: A4-4: first-pass fixed points are detected on pass two)
def test_initial_fixed_point_needs_one_detection_pass(monkeypatch):
    ns, calls, _ = known_trajectory(monkeypatch)
    loss, used = ns["consist_loss"](8, torch.zeros(2, 1, dtype=torch.long))
    assert used == 0 and len(calls) >= 12  # full teacher budget still executes
    assert ns["traj_stats"]["passes"] == 2  # two sources, one detection pass each


# fixed (was a strict xfail: A4-5: curve evaluations discard ordered hypotheses and pass counts)
def test_curve_evaluation_persists_ordered_comparator_inputs(tmp_path):
    tree = ast.parse(source(WMT))
    loop = next(n for n in tree.body if isinstance(n, ast.While))
    block = next(n for n in loop.body if isinstance(n, ast.If) and "tc.eval_every" in ast.unparse(n.test))
    hyp, capped, counts = ["a", "b"], {c: [f"a{c}", f"b{c}"] for c in range(1, 13)}, [8, 9]
    records = []
    ns = dict(trainer=SimpleNamespace(step=500, log=lambda rec: records.append(rec) or rec),
              tc=SimpleNamespace(eval_every=500, steps=2000), n_curve=2,
              val_text=dict(src=["1", "2"], ref=["a", "b"]),
              evaluate=lambda *a, **kw: (dict(nfe_per_sentence=counts[:]), hyp, capped),
              val_nll=lambda: 1., json=json, os=os, a=SimpleNamespace(run_dir=str(tmp_path)))
    exec(compile(ast.Module(body=[block], type_ignores=[]), WMT, "exec"), ns)
    payloads = records + [json.loads(p.read_text()) for p in tmp_path.rglob("*.json")]
    assert any(p.get("hyp_val") == hyp and
               p.get("hyp_val_capped") == {str(c): h for c, h in capped.items()} and
               p.get("nfe_val") == counts for p in payloads)


def test_hash_pinned_asset_parallel_fp32_teacher_still_matches_sequential(released_sudoku):
    # Shared primitive cross-check on released, hash-verified weights; not a WMT14 quality estimate.
    bundle, prompts = released_sudoku
    net, tok = bundle.net.eval(), bundle.vqvae.eval()
    x = prompts[:2]
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        seq = net.generate(None, x, temperature=0., top_k=0, top_p=1., cfg_scale=0., cfg_null_eos=None)
        z = tok.encoder.wte(x)
        result = jt.jacobi_decode(net, lambda c, h: z, len(x), seq.shape[1], "cpu", len(tok.vq.levels) + 1)
    assert torch.equal(result["codes"], seq)


def test_no_rng_draw_between_wmt_restore_and_first_training_step():
    tree = ast.parse(source(WMT))
    # ported: the restore's return value (the checkpoint's extra) is now kept, so the call may be an assignment
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, (ast.Expr, ast.Assign)) and
                 isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "trainer.maybe_resume")
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.While))
    executed = [n for n in tree.body[start + 1:end] if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    text = ast.unparse(ast.Module(body=executed, type_ignores=[]))
    assert all(name not in text for name in ["torch.rand", "rng.", "batch(", "manual_seed", "load_ncm"])
