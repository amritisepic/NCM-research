"""Audit 6 positive CPU evidence for the J2-F teacher and continuation gates.

These probes execute the trainer's own construction, consistency, loop and
checkpoint-statements. Released assets come only from the hash-pinned fixture.
"""
import ast
import copy

import torch

from test_audit4_protocol import (
    WMT, assert_nested_equal, definitions, source, step, tiny_wmt,
)
from test_j2f_frozen_teacher import production_teacher


def _restore_stats(ns, extra):
    tree = ast.parse(source(WMT))
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n).startswith("_rs ="))
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
               and ast.unparse(n).startswith("traj_stats ="))
    ns["resume_extra"] = extra
    exec(compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), WMT, "exec"), ns)
    definitions(WMT, ["run_stats"], ns)


def _frozen_run(path):
    ns = tiny_wmt(path)
    ns["bs"], ns["n_c"] = 16, 8
    before = torch.get_rng_state().clone()
    ns["teacher"] = production_teacher(ns["net"])
    assert torch.equal(before, torch.get_rng_state()), "constructing the teacher consumed the training RNG"
    ns["modules"] = {"model": ns["net"]}

    # Keep one real trajectory and one degenerate source per step. This makes
    # both the state draws and the nonzero run-level degenerate count observable.
    decode = ns["jacobi_decode"]

    def with_degenerate(*args, **kwargs):
        out = decode(*args, **kwargs)
        out["codes"][-1].zero_()
        for state in out["history"]:
            state[-1].zero_()
        out["nfe"][-1] = 1
        out["stop"][-1] = 1
        return out

    ns["jacobi_decode"] = with_degenerate
    _restore_stats(ns, {})
    return ns


def _save(ns):
    # Execute the production periodic checkpoint's own save expression.
    tree = ast.parse(source(WMT))
    loop = next(n for n in tree.body if isinstance(n, ast.While))
    block = next(n for n in loop.body if isinstance(n, ast.If)
                 and "tc.ckpt_every" in ast.unparse(n.test))
    exec(compile(ast.Module(body=block.body, type_ignores=[]), WMT, "exec"), ns)


def _resume(ns):
    tree = ast.parse(source(WMT))
    statement = next(n for n in tree.body if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Call)
                     and ast.unparse(n.value.func) == "trainer.maybe_resume")
    exec(compile(ast.Module(body=[statement], type_ignores=[]), WMT, "exec"), ns)
    _restore_stats(ns, ns["resume_extra"])


def _training_state(ns):
    return copy.deepcopy(dict(
        weights=ns["net"].state_dict(),
        optimizer=ns["trainer"].opt.state_dict(),
        data=ns["rng"].bit_generator.state,
        draws=torch.get_rng_state(),
        run_stats=ns["run_stats"](),
    ))


def test_frozen_resume_matches_uninterrupted_training_and_restores_run_totals(tmp_path):
    ns = _frozen_run(tmp_path / "run")
    released = copy.deepcopy(ns["teacher"].state_dict())
    step(ns)
    _save(ns)
    expected_losses = [step(ns), step(ns)]
    expected = _training_state(ns)
    assert expected["run_stats"] == {"sources": 6, "degenerate": 3}

    resumed = _frozen_run(tmp_path / "run")
    _resume(resumed)
    assert resumed["run_stats"]() == {"sources": 2, "degenerate": 1}
    assert_nested_equal(resumed["teacher"].state_dict(), released)
    # A validation call between restoration and continuation must consume no
    # state/dropout RNG and leave the student ready to train.
    resumed["val_nll"]()
    assert [step(resumed), step(resumed)] == expected_losses
    assert_nested_equal(_training_state(resumed), expected)
    assert_nested_equal(resumed["teacher"].state_dict(), released)
    assert not resumed["teacher"].training and resumed["net"].training


def test_frozen_and_moving_teacher_leave_identical_rng_and_update_when_maps_coincide(tmp_path):
    moving = tiny_wmt(tmp_path / "moving")
    moving["bs"], moving["n_c"] = 16, 8
    definitions(WMT, ["run_stats"], moving)
    initial_rng = torch.get_rng_state().clone()
    expected_loss = step(moving)
    expected = _training_state(moving)

    frozen = tiny_wmt(tmp_path / "frozen")
    frozen["bs"], frozen["n_c"] = 16, 8
    before_copy = torch.get_rng_state().clone()
    frozen["teacher"] = production_teacher(frozen["net"])
    definitions(WMT, ["run_stats"], frozen)
    assert torch.equal(before_copy, initial_rng)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert step(frozen) == expected_loss
    assert_nested_equal(_training_state(frozen), expected)


def test_released_teacher_has_isolated_buffers_and_tokenizer_storage(released_sudoku):
    bundle, _ = released_sudoku
    student = bundle.net
    rng = torch.get_rng_state().clone()
    teacher = production_teacher(student)
    assert torch.equal(rng, torch.get_rng_state())
    assert not teacher.training

    def assert_isolated(original, cloned):
        left = dict(original.named_parameters()) | dict(original.named_buffers())
        right = dict(cloned.named_parameters()) | dict(cloned.named_buffers())
        assert left.keys() == right.keys()
        for name, tensor in left.items():
            other = right[name]
            assert tensor.dtype == other.dtype and tensor.device == other.device, name
            assert torch.equal(tensor, other), name
            if tensor.numel():
                assert tensor.untyped_storage().data_ptr() != other.untyped_storage().data_ptr(), name

    assert_isolated(student, teacher)
    assert teacher.vqvae_local[0] is not student.vqvae_local[0]
    assert not teacher.vqvae_local[0].training
    assert_isolated(student.vqvae_local[0], teacher.vqvae_local[0])
