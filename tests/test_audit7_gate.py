"""Audit 7 positive CPU probes for the new launch gate and eval selector.

No new defect was established in source review, so these are ordinary positive
probes, not xfails. Production functions are extracted from the checkout; the
legacy comparator is audit 6's unchanged evaluate implementation. Released
weights use conftest's hash gate. Execution status is recorded in the report.
"""
import ast
import copy
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ncmr.jacobi_train import jacobi_decode
from test_audit4_protocol import ROOT, WMT, assert_nested_equal, definitions, step
from test_j2f_frozen_teacher import production_teacher, teacher_check_ns


def _run(path):
    ns = teacher_check_ns(path)
    ns["teacher"] = production_teacher(ns["net"])
    ns["bs"], ns["n_c"] = 16, 8
    definitions(WMT, ["run_stats"], ns)
    return ns


def _reference(ns, path):
    val = ns["val_text"]
    ev, hyp, _ = ns["evaluate"](val["src"], val["ref"], sequential=True, caps=())
    path.write_text(json.dumps({"hyp_val": hyp, "nfe_val": ev["nfe_per_sentence"]}), encoding="utf-8")
    return path


def _state(ns):
    """Everything decoded modules or the next training step can observe."""
    modules = {"net": ns["net"], "tok": ns["tok"], "teacher": ns["teacher"],
               "teacher_tok": ns["teacher"].vqvae_local[0]}
    return copy.deepcopy({
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "modes": {name: [m.training for m in module.modules()] for name, module in modules.items()},
        "grads": {name: [None if p.grad is None else p.grad.clone() for p in module.parameters()]
                  for name, module in modules.items()},
        "opt": ns["trainer"].opt.state_dict(), "step": ns["trainer"].step,
        "data_rng": ns["rng"].bit_generator.state, "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "traj_stats": ns["traj_stats"],
        "precision": (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"),
                      torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32),
        "grad_enabled": torch.is_grad_enabled(),
    })


@pytest.mark.parametrize("failure", [None, "hypothesis", "passes", "missing"])
@pytest.mark.parametrize("outer_autocast", [False, True])
def test_launch_gate_leaves_rng_modules_and_precision_unchanged(tmp_path, failure, outer_autocast):
    ns = _run(tmp_path / "run")
    path = _reference(ns, tmp_path / "hyp_val.json")
    if failure == "missing":
        path = tmp_path / "absent.json"
    elif failure is not None:
        record = json.loads(path.read_text(encoding="utf-8"))
        if failure == "hypothesis":
            record["hyp_val"][0] += " changed"
        else:
            record["nfe_val"][0] += 1
        path.write_text(json.dumps(record), encoding="utf-8")

    forwards = []
    hook = ns["teacher"].register_forward_pre_hook(lambda m, args: forwards.append(
        (m.training, torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"), args[0].dtype)))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=outer_autocast):
            before = _state(ns)
            if failure is None:
                ns["check_teacher"](str(path), n=len(ns["val_text"]["src"]))
            else:
                with pytest.raises(SystemExit):
                    ns["check_teacher"](str(path), n=len(ns["val_text"]["src"]))
            assert_nested_equal(_state(ns), before)
    finally:
        hook.remove()
    if failure == "missing":
        assert not forwards
    else:
        assert forwards and all(v == (False, False, False, torch.float32) for v in forwards)
    assert ns["net"].training and not ns["teacher"].training


def test_launch_gate_preserves_next_steps_and_checkpoint_resume(tmp_path):
    baseline = _run(tmp_path / "baseline")
    reference = _reference(baseline, tmp_path / "hyp_val.json")
    initial_rng = torch.get_rng_state().clone()
    expected_losses = [step(baseline) for _ in range(3)]
    expected = _state(baseline)

    checked = _run(tmp_path / "checked")
    assert torch.equal(initial_rng, torch.get_rng_state())
    checked["check_teacher"](str(reference), n=6)
    assert step(checked) == expected_losses[0]
    modules = {"model": checked["net"]}
    checked["trainer"].save(modules, checked["rng"], extra={"traj_stats": checked["traj_stats"]})

    resumed = _run(tmp_path / "checked")
    extra = resumed["trainer"].maybe_resume({"model": resumed["net"]}, resumed["rng"])
    resumed["traj_stats"] = extra["traj_stats"]
    before = _state(resumed)
    resumed["check_teacher"](str(reference), n=6)
    assert_nested_equal(_state(resumed), before)
    assert [step(resumed), step(resumed)] == expected_losses[1:]
    assert_nested_equal(_state(resumed), expected)


def test_default_gate_replays_j0_first_128_as_one_identical_ordered_batch(tmp_path):
    ns = _run(tmp_path / "run")
    src = [f"{1 + i % 28} {1 + (i // 28) % 28} 3 4" for i in range(129)]
    refs = ["word1 word2"] * len(src)
    ns["val_text"] = {"src": src, "ref": refs}
    real_decode, batches = ns["jacobi_decode"], []

    def record_decode(model, prefix_fn, n, length, device, passes, **kw):
        batches.append({"rows": n, "length": length, "device": device, "passes": passes,
                        "prefix": prefix_fn(torch.zeros(n, length, dtype=torch.long), False).clone(),
                        "pad_mask": kw["pad_mask"].clone(),
                        "keep_history": kw["keep_history"], "initial_fixed_ok": kw["initial_fixed_ok"]})
        return real_decode(model, prefix_fn, n, length, device, passes, **kw)

    ns["jacobi_decode"] = record_decode
    # J0's full ordered evaluation crosses the 128-row batch boundary.
    ev, hyp, _ = ns["evaluate"](src, refs, sequential=True, caps=())
    assert [batch["rows"] for batch in batches] == [128, 1]
    j0_first_batch = batches[0]
    path = tmp_path / "hyp_val.json"
    path.write_text(json.dumps({"hyp_val": hyp, "nfe_val": ev["nfe_per_sentence"]}), encoding="utf-8")
    batches.clear()
    before = _state(ns)
    ns["check_teacher"](str(path))  # the actual default n=128 and bs_=128
    assert len(batches) == 1
    assert_nested_equal(batches[0], j0_first_batch)
    assert_nested_equal(_state(ns), before)


@pytest.mark.parametrize("sequential,caps,batch_size", [
    (True, None, 128), (True, None, 2), (True, (1, 4, 12), 128), (False, (), 128),
])
def test_existing_evaluate_calls_match_audit6_outputs_and_post_call_mode(tmp_path, sequential, caps, batch_size):
    ns = _run(tmp_path)
    # Read exactly the pre-delta implementation, rather than restating it.
    legacy = subprocess.check_output(["git", "show", "dd64aa8:" + WMT], cwd=ROOT).decode("utf-8")
    function = next(n for n in ast.parse(legacy).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    function.name = "legacy_evaluate"
    exec(compile(ast.Module(body=[function], type_ignores=[]), WMT, "exec"), ns)
    val = ns["val_text"]
    before = _state(ns)
    expected = ns["legacy_evaluate"](val["src"], val["ref"], bs_=batch_size, sequential=sequential, caps=caps)
    assert ns["net"].training
    assert_nested_equal(_state(ns), before)
    actual = ns["evaluate"](val["src"], val["ref"], bs_=batch_size, sequential=sequential, caps=caps)
    assert_nested_equal(actual, expected)
    assert ns["net"].training
    assert_nested_equal(_state(ns), before)


def test_launch_gate_decodes_hash_pinned_released_weights_without_mutating_them(tmp_path, released_sudoku):
    """Exercise real shared NCM/VQ paths; this is not a WMT14 replay claim."""
    bundle, puzzles = released_sudoku
    net, tok = copy.deepcopy((bundle.net, bundle.vqvae))
    net.train()
    teacher = production_teacher(net)
    levels = [int(v) for v in tok.vq.levels]
    src = [" ".join(str(int(t)) for t in row) for row in puzzles[:4]]
    ns = dict(torch=torch, np=np, os=os, sys=sys, json=json,
              net=net, tok=tok, teacher=teacher, vq=tok.vq, dev="cpu", K=len(levels), S=sum(levels),
              P=int(net.prefix_len), PAD=-1, EOS=-2, jacobi_decode=jacobi_decode,
              pt=SimpleNamespace(encode=lambda s: [int(t) for t in s.split()],
                                 decode=lambda ids, **kw: " ".join(str(int(t)) for t in ids)),
              val_text={"src": src, "ref": ["1 2 3 4"] * len(src)})
    definitions(WMT, ["prefix", "detok", "evaluate", "check_teacher"], ns)
    # This fixture is Sudoku, so omit WMT's sequential-generation parity claim.
    ev, hyp, _ = ns["evaluate"](src, ns["val_text"]["ref"], sequential=False, caps=())
    path = tmp_path / "hyp_val.json"
    path.write_text(json.dumps({"hyp_val": hyp, "nfe_val": ev["nfe_per_sentence"]}), encoding="utf-8")
    before = copy.deepcopy({"net": net.state_dict(), "tok": tok.state_dict(),
                            "teacher": teacher.state_dict(), "teacher_tok": teacher.vqvae_local[0].state_dict(),
                            "torch_rng": torch.get_rng_state()})
    ns["check_teacher"](str(path), n=len(src))
    assert_nested_equal({"net": net.state_dict(), "tok": tok.state_dict(),
                         "teacher": teacher.state_dict(), "teacher_tok": teacher.vqvae_local[0].state_dict(),
                         "torch_rng": torch.get_rng_state()}, before)
    assert net.training and not teacher.training and not tok.training and not teacher.vqvae_local[0].training
