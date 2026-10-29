"""PD/PDa: execute matched training, exact marginal preservation, repair and atomic continuation on CPU."""
import copy
import json

import pytest
import torch
from omegaconf import OmegaConf

from ncmr.w2m import pd_permutation
from w2m_support import Harness, ROOT, run_nodes, script_nodes


@pytest.mark.parametrize("canonical,proposals", [([0], [1]), ([0, 1], [1, 0]),
                                                   ([0] * 20, [1, 2] * 10),
                                                   ([0, 1, 2, 0, 2], [1, 2, 0, 2, 1])])
def test_pda_adversarial_groups_terminate_preserve_multiset_and_exclude_canonical(canonical, proposals):
    c, p = torch.tensor(canonical), torch.tensor(proposals)
    for seed in range(32):
        pi = pd_permutation(p, c, torch.Generator().manual_seed(seed))
        assert torch.equal(pi.sort().values, torch.arange(len(p)))
        assert torch.equal(p[pi].sort().values, p.sort().values)
        assert not (p[pi] == c).any()
        if len(p) == 1 or (len(p) == 2 and canonical == [0, 1]):
            assert torch.equal(pi, torch.arange(len(p)))
        elif len(set(canonical)) == 1:
            assert not (pi == torch.arange(len(p))).any()


def trace(tmp_path, arm, until=2, resume=False):
    h = Harness(tmp_path, arm=arm, B=16, accum=4, ckpt_every=1, resume=resume, at=(1, 2))
    records, proposals, batches = [], [], []
    original_record, original_proposals, original_batch = h.mech.record, h.mech.proposals, h.ns["stream"].batch

    def record(used, targets, plan):
        records.append((used.clone(), targets.clone(), plan.clean.clone(), [m.clone() for m in plan.masks]))
        return original_record(used, targets, plan)

    def propose(canon, plan):
        before = torch.get_rng_state().clone()
        out = original_proposals(canon, plan)
        assert torch.equal(before, torch.get_rng_state())
        proposals.append(out.clone())
        return out

    def batch(rng, n):
        x = original_batch(rng, n)
        batches.append(x.copy())
        return x

    h.mech.record, h.mech.proposals, h.ns["stream"].batch = record, propose, batch
    h.run(until)
    return h, records, proposals, batches


def test_pd_executed_trainer_matches_p_data_masks_proposals_and_rng(tmp_path):
    p, d = [trace(tmp_path / arm, arm) for arm in ("m2", "pd")]
    assert all((a == b).all() for a, b in zip(p[3], d[3]))
    assert len(p[2]) == len(d[2]) == 8
    assert all(torch.equal(a, b) for a, b in zip(p[2], d[2]))
    levels = p[0].mech.levels
    reassigned, masked = [0] * len(levels), [0] * len(levels)
    # Four micro-batches form one optimizer batch: compare every masked level/position group at that scope.
    for start in (0, 4):
        rp, rd = p[1][start:start + 4], d[1][start:start + 4]
        up, ud = [torch.cat([r[0] for r in records]) for records in (rp, rd)]
        cp, cd = [torch.cat([r[1] for r in records]) for records in (rp, rd)]
        assert torch.equal(cp, cd)
        assert torch.equal(torch.cat([r[2] for r in rp]), torch.cat([r[2] for r in rd]))
        masks = [torch.cat([r[3][k] for r in rp]) for k in range(len(levels))]
        assert all(torch.equal(m, torch.cat([r[3][k] for r in rd])) for k, m in enumerate(masks))
        assert torch.equal(ud != cd, torch.cat(masks, 1))
        offset = 0
        for k, length in enumerate(levels):
            if 2 <= k <= 5:
                for pos in range(length):
                    rows = masks[k][:, pos]
                    assert torch.equal(up[rows, offset + pos].sort().values, ud[rows, offset + pos].sort().values)
                masked[k] += int(masks[k].sum())
            else:
                assert torch.equal(up[:, offset:offset + length], ud[:, offset:offset + length])
            offset += length
    for key in ("mask_rng", "repl_rng", "flag_rng"):
        assert torch.equal(p[0].mech.state_dict()[key], d[0].mech.state_dict()[key])
    assert p[0].data_rng.bit_generator.state == d[0].data_rng.bit_generator.state
    assert torch.equal(p[0].state()["torch_rng"], d[0].state()["torch_rng"])
    coverage = d[0].mech.realized()["pd_coverage"]
    assert coverage["masked_per_level"] == masked
    assert coverage["overall"] == sum(coverage["reassigned_per_level"]) / sum(masked)
    assert all(coverage["per_level"][k] == coverage["reassigned_per_level"][k] / masked[k] for k in range(2, 6))
    assert d[0].sample_calls == p[0].sample_calls == [1, 2]
    assert d[0].losses()[-1]["realized"]["pd_coverage"] == coverage


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    else:
        assert a == b


def test_pd_atomic_resume_equals_uninterrupted(tmp_path):
    complete = trace(tmp_path / "complete", "pd", until=4)[0]
    trace(tmp_path / "resume", "pd", until=2)
    resumed = trace(tmp_path / "resume", "pd", until=4, resume=True)[0]
    assert_state_equal(complete.state(), resumed.state())
    assert complete.data_rng.bit_generator.state == resumed.data_rng.bit_generator.state
    assert [r["loss"] for r in complete.losses()] == [r["loss"] for r in resumed.losses()]
    assert_state_equal(complete.trainer.opt.state_dict(), resumed.trainer.opt.state_dict())


def test_pd_missing_rng_or_coverage_refuses_resume(tmp_path):
    h = Harness(tmp_path, arm="pd")
    state = h.mech.state_dict()
    for key in ("pd_rng", "pd_masked", "pd_reassigned"):
        broken = copy.deepcopy(state)
        del (broken if key == "pd_rng" else broken["stats"])[key]
        with pytest.raises(RuntimeError, match="exact resume"):
            h.mech.load_state_dict(broken)


def test_pd_permutation_coverage_counts_row_identity_even_when_codes_equal(tmp_path):
    h = Harness(tmp_path, arm="pd", B=16)
    m = h.mech
    plan = m.draw_plan(16)
    canonical = torch.zeros(16, m.S, dtype=torch.long)
    proposals = torch.ones_like(canonical)
    m.permute_proposals(canonical, proposals, plan)
    coverage = m.realized()["pd_coverage"]
    # All non-singleton groups derange completely although every incoming code has the same value.
    expect = [sum(int(mask[:, j].sum()) for j in range(mask.shape[1]) if int(mask[:, j].sum()) > 1)
              if 2 <= k <= 5 else 0 for k, mask in enumerate(plan.masks)]
    assert coverage["reassigned_per_level"] == expect


def test_pd_configs_are_same_w2m_recipe_at_registered_steps_and_eval():
    configs = [OmegaConf.load(ROOT / f"configs/scale/L_ts_pd_{key}.yaml") for key in "pdu"]
    for cfg, arm in zip(configs, ("m2", "pd", "m3")):
        reference = OmegaConf.load(ROOT / f"configs/scale/L_ts_{'m3_uniform_wrong' if arm == 'm3' else 'm2_frozen_pw'}.yaml")
        assert cfg.train.steps == 175 and cfg.train.seed == 0 and cfg.corruption.arm == arm
        cfg.corruption.arm = reference.corruption.arm
        reference.train.steps = 175
        reference.sample_eval.at = [100]
        assert OmegaConf.to_container(cfg) == OmegaConf.to_container(reference)


@pytest.mark.parametrize("change", ["arm", "config", "base"])
def test_pd_recorded_run_rejects_changed_identity(tmp_path, change):
    from test_audit3_protocol import _text_manifest_ns
    if change == "base":
        ns, node, guard, run, art = _text_manifest_ns(tmp_path)
        run(node)
        ns.update(ckpt_path=str(tmp_path / "ckpt.pt"), manifest_path=str(tmp_path / "run_manifest.json"))
        (tmp_path / "run_manifest.json").write_text(json.dumps(ns["manifest"]))
        guard()
        (art / "generator.pt").write_bytes(b"changed")
        run(node)
        with pytest.raises(SystemExit):
            guard()
        return
    import ast
    funcs, _, _ = script_nodes()
    tree = ast.parse((ROOT / "scripts/train_gen.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.If) and "_run_key(" in ast.unparse(n))
    cfg = OmegaConf.load(ROOT / "configs/scale/L_ts_pd_d.yaml")
    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, cfg_path)
    ns = dict(OmegaConf=OmegaConf, cfg=cfg, cfg_path=str(cfg_path), _recorded=True,
              a=type("Args", (), {"calib_steps": 0, "run_dir": "run"})(), os=__import__("os"), sys=__import__("sys"))
    run_nodes([funcs["_run_key"], node], ns)
    if change == "arm":
        cfg.corruption.arm = "m2"
    else:
        cfg.train.lr *= 2
    with pytest.raises(SystemExit, match="different run"):
        run_nodes([node], ns)
