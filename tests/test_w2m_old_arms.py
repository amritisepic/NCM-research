"""W2M must not change any existing arm: the old step functions are syntactically unchanged, build_hierarchy without the
new arguments is identical to the pre-W2M implementation, and the old loop and the current loop give identical runs."""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from ncmr.corruption import CorruptionCfg, build_hierarchy
from w2m_support import Harness, script_nodes, tiny_case

HERE = Path(__file__).resolve().parent


def _ref_corruption():
    spec = importlib.util.spec_from_file_location("w2m_ref_corruption", HERE / "w2m_ref_corruption.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


OLD_FUNCS = ["_run_key", "_digest", "prefix_emb", "policy_codes", "compute_loss", "val_nll", "draw_samples",
             "sample_metrics", "_sample_metrics"]


@pytest.mark.parametrize("name", OLD_FUNCS)
def test_old_step_functions_are_syntactically_unchanged(name):
    ref, _, _ = script_nodes("ref")
    cur, _, _ = script_nodes("cur")
    assert ast.dump(ref[name]) == ast.dump(cur[name])


@pytest.mark.parametrize("cfg", [
    dict(mode="none"), dict(mode="uniform"), dict(mode="onpolicy"), dict(mode="uniform", retarget=True),
    dict(mode="onpolicy", retarget=True), dict(mode="uniform", soft_tau=0.5, soft_topm=4),
    dict(mode="uniform", level_rates=(1.0,) * 11 + (0.0,))])
def test_build_hierarchy_without_w2m_arguments_equals_the_pre_w2m_implementation(cfg):
    ref = _ref_corruption()
    tok, _ = tiny_case()
    x = torch.randint(0, 32, (6, 128), generator=torch.Generator().manual_seed(2))
    pol = torch.randint(0, 32, (6, 641), generator=torch.Generator().manual_seed(3))
    a = build_hierarchy(tok, x, CorruptionCfg(**cfg), pol, torch.Generator().manual_seed(9))
    b = ref.build_hierarchy(tok, x, ref.CorruptionCfg(**cfg), pol, torch.Generator().manual_seed(9))
    assert a.keys() == b.keys()
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def test_a_supplied_canonical_equals_the_internal_one_and_masks_override_the_rates():
    tok, _ = tiny_case()
    x = torch.randint(0, 32, (6, 128), generator=torch.Generator().manual_seed(2))
    canon = tok.vq.z_to_idx(tok._encode_pre_quant(x))
    clean = build_hierarchy(tok, x, CorruptionCfg(mode="none"))
    given = build_hierarchy(tok, x, CorruptionCfg(mode="mech"), canonical=canon, masks=[torch.zeros(6, l, dtype=torch.bool) for l in tok.vq.levels],
                            policy_codes=canon)
    for k in ("inputs", "targets", "canonical", "used", "z_used"):
        assert torch.equal(clean[k], given[k]), k
    masks = [torch.zeros(6, l, dtype=torch.bool) for l in tok.vq.levels]
    masks[2][1, 3] = True
    repl = canon.clone()
    repl[:, 9 + 3] = (canon[:, 9 + 3] + 1) % 32                    # level 2 starts at 1 + 8 = 9
    out = build_hierarchy(tok, x, CorruptionCfg(mode="mech", mean_rate=0.0), policy_codes=repl, masks=masks, canonical=canon)
    assert int((out["used"] != canon).sum()) == 1 and out["used"][1, 12] == repl[1, 12]
    with pytest.raises(AssertionError):
        build_hierarchy(tok, x, CorruptionCfg(mode="mech", retarget=True), policy_codes=repl, masks=masks, canonical=canon)


@pytest.mark.parametrize("mode", ["uniform", "onpolicy"])
def test_old_arms_run_identically_under_the_pre_w2m_loop_and_the_current_loop(tmp_path, mode):
    old = Harness(tmp_path / "old", mode=mode, kind="ref", accum=2).run(3)
    cur = Harness(tmp_path / "cur", mode=mode, kind="cur", accum=2).run(3)
    assert [r["loss"] for r in old.losses()] == [r["loss"] for r in cur.losses()]
    assert len(cur.losses()) == 3
    so, sc = old.state(), cur.state()
    for k in so["model"]:
        assert torch.equal(so["model"][k], sc["model"][k]), k
    for k in ("torch_rng", "gen", "policy_gen"):
        assert torch.equal(so[k], sc[k]), k
    # checkpoints: the same keys, no W2M additions
    a = torch.load(tmp_path / "old" / "ckpt.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "cur" / "ckpt.pt", map_location="cpu", weights_only=False)
    assert set(a["extra"]) == set(b["extra"]) == {"corruption_rng", "policy_rng"}
    assert set(a["modules"]) == set(b["modules"]) == {"model"}
    # log lines carry no W2M fields
    assert not any("realized" in r for r in cur.losses())


@pytest.mark.parametrize("mode", ["uniform", "onpolicy"])
def test_old_arms_still_resume_exactly(tmp_path, mode):
    whole = Harness(tmp_path / "w", mode=mode).run(4)
    Harness(tmp_path / "s", mode=mode).run(2)
    rest = Harness(tmp_path / "s", mode=mode, resume=True).run(4)
    a, b = whole.state(), rest.state()
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert torch.equal(a["gen"], b["gen"]) and torch.equal(a["policy_gen"], b["policy_gen"])


def test_existing_configs_are_untouched_by_the_w2m_branch():
    import subprocess
    out = subprocess.run(["git", "diff", "--name-only", "38a08fa", "--", "configs/scale"], cwd=HERE.parent,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("base commit not available")
    changed = [p for p in out.stdout.split() if "L_ts_m" not in p]
    assert changed == [], changed
