"""Audit 8 positive checks of production orchestration and independent metric oracles."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ncmr import eval_text, w2m
from ncmr.data import cut_at_eos
from ncmr.ncm_ref_loader import ForwardCounter
from w2m_support import Harness, prefix_fn, run_nodes, script_nodes, tiny_case

ROOT = Path(__file__).resolve().parents[1]


def load_script(path):
    spec = importlib.util.spec_from_file_location("audit8_" + Path(path).stem, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("arm", sorted(w2m.ARMS))
def test_real_loop_masks_do_not_depend_on_micro_batch_partition(tmp_path, arm):
    states, batches = [], []
    for accum in (1, 2, 4):
        h = Harness(tmp_path / str(accum), arm=arm, accum=accum)
        draws = []
        original = h.mech.draw_plan

        def record(n):
            p = original(n)
            draws.append((p.clean.clone(), [m.clone() for m in p.masks]))
            return p

        h.mech.draw_plan = record
        h.run(2)
        states.append(h.mech.mask_gen.get_state().clone())
        batches.append(draws)
    assert all(torch.equal(states[0], s) for s in states[1:])
    for step in range(2):
        for other in batches[1:]:
            assert torch.equal(batches[0][step][0], other[step][0])
            assert all(torch.equal(a, b) for a, b in zip(batches[0][step][1], other[step][1]))


def test_frozen_copy_owns_parameters_buffers_tokenizer_and_attention_cache():
    tok, net = tiny_case()
    net.transformer.blocks[0].attn.cached_k = torch.ones(1, 2, 3, 4)
    frozen = w2m.MechArm.frozen_copy(net)
    for student, teacher in ((net, frozen), (tok, frozen.vqvae_local[0])):
        assert student is not teacher
        for accessor in ("named_parameters", "named_buffers"):
            a, b = dict(getattr(student, accessor)()), dict(getattr(teacher, accessor)())
            assert a.keys() == b.keys()
            for name in a:
                assert torch.equal(a[name], b[name])
                if a[name].numel():
                    assert a[name].data_ptr() != b[name].data_ptr(), name
    a, b = net.transformer.blocks[0].attn.cached_k, frozen.transformer.blocks[0].attn.cached_k
    assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    assert not any(m.training for m in frozen.modules())
    assert not any(p.requires_grad for p in frozen.parameters())


@pytest.mark.parametrize("arm", ["m1", "m5"])
def test_real_sampling_and_val_leave_the_next_training_step_identical(tmp_path, monkeypatch, arm):
    # Only external LM scoring is replaced; real generation, val NLL, RNG fork and loop all execute.
    def score(*args, **kwargs):
        torch.rand(9)  # simulate the scorer's lazy initialisation
        return {"genppl_gpt2-large": 1.0}

    monkeypatch.setattr(eval_text, "evaluate_samples", score)
    result = []
    funcs, _, _ = script_nodes()
    for name, at in (("plain", ()), ("evaluated", (1, 2))):
        h = Harness(tmp_path / name, arm=arm, at=at)
        h.ns.update(val_x=torch.randint(0, 32, (2, 128), generator=torch.Generator().manual_seed(17)),
                    ForwardCounter=ForwardCounter, cut_at_eos=cut_at_eos,
                    bpe=SimpleNamespace(encode=lambda s: [int(x) for x in s.split()],
                                        decode=lambda ids: " ".join(map(str, ids))))
        run_nodes([funcs[n] for n in ("val_nll", "draw_samples", "sample_metrics", "_sample_metrics")], h.ns)
        h.run(3)
        result.append((h.state(), [r["loss"] for r in h.losses()], h.data_rng.bit_generator.state))
    a, b = result
    assert a[1:] == b[1:]
    assert all(torch.equal(a[0]["model"][k], b[0]["model"][k]) for k in a[0]["model"])
    for k in ("torch_rng", "gen", "policy_gen"):
        assert torch.equal(a[0][k], b[0][k]), k
    for k in ("mask_rng", "repl_rng", "flag_rng"):
        assert torch.equal(a[0]["mech"][k], b[0]["mech"][k]), k
    assert a[0]["mech"]["stats"] == b[0]["mech"]["stats"]
    if arm == "m5":
        assert torch.equal(a[0]["flag"], b[0]["flag"])


def test_kl_matches_independent_direction_and_finer_level_oracle():
    tok, net = tiny_case()
    levels = [int(l) for l in tok.vq.levels]
    canon = torch.cat([torch.randint(0, 32, (3, l)) for l in levels], 1)
    perturb = w2m.make_perturbations(canon, levels, [32] * len(levels), (1, 4, 6), 3)
    got = w2m.context_dependence_kl(net, tok, prefix_fn(tok), canon, perturb, "cpu", bs=2)
    with torch.no_grad(), w2m.fp32_eval(net, "cpu"):
        clean = net(torch.cat([prefix_fn(tok)(3), tok.vq.idx_to_ncm_input(canon)], 1))
        for j, (pos, shift) in perturb.items():
            changed = canon.clone()
            idx = sum(levels[:j]) + pos
            changed[torch.arange(3), idx] = (changed[torch.arange(3), idx] + shift) % 32
            inp = tok.vq.idx_to_ncm_input(changed)
            pert = net(torch.cat([prefix_fn(tok)(3), inp], 1))
            select = torch.cat([torch.full((l,), k > j, dtype=torch.bool) for k, l in enumerate(levels[1:], 1)])
            target, lp = clean[:, select].softmax(-1), pert[:, select].log_softmax(-1)
            expected = F.kl_div(lp, target, reduction="none").sum(-1).double().mean()
            assert got[j] == pytest.approx(float(expected), rel=2e-4, abs=2e-8)


def test_mechanism_orchestrator_uses_heldout_true_flags_and_zero_flag_probe(monkeypatch):
    script = load_script("scripts/mech_eval_w2m.py")
    n, split = 8, 6
    tok, net = tiny_case()
    contexts = {kind: {"inputs": torch.full((n, 4, 8), float(i)),
                        "flags": torch.arange(n * 4).view(n, 4).float(),
                        "targets": torch.arange(n * 4).view(n, 4),
                        "contaminated": torch.arange(n * 4).view(n, 4) % 2 == 0,
                        "masks": [torch.zeros(n, 1, dtype=torch.bool)]}
                for i, kind in enumerate(("plausible_wrong", "uniform_wrong"))}
    observed = []
    monkeypatch.setattr(w2m, "context_dependence_kl", lambda *a: {1: 0.1})

    def hidden(net, prefix, inputs, device, **kwargs):
        assert kwargs == {"hidden": True, "keep_logits": False}  # no flag or flag vector
        return None, torch.arange(n * 4 * 2).view(n, 4, 2).float()

    def probe(htr, ytr, hte, yte):
        assert htr.shape[0] == split and hte.shape[0] == n - split
        assert int(hte[0, 0, 0]) == split * 4 * 2
        return 0.5

    def flag_use(net, flag, prefix, ctx, device):
        observed.append(ctx)
        return {"delta_per_window": -1.0}

    monkeypatch.setattr(w2m, "forward_logits", hidden)
    monkeypatch.setattr(w2m, "fit_probe_auc", probe)
    monkeypatch.setattr(w2m, "flag_use", flag_use)
    script.evaluate_model(net, w2m.FlagVector(8), tok, prefix_fn(tok), None, {}, contexts, split, "cpu", 0)
    assert len(observed) == 1
    for k in ("inputs", "flags", "targets"):
        assert torch.equal(observed[0][k], contexts["plausible_wrong"][k][split:])


def test_flag_use_has_registered_sign_and_uses_true_flags():
    tok, net = tiny_case()
    levels = [int(l) for l in tok.vq.levels]
    canon = torch.cat([torch.randint(0, 32, (3, l)) for l in levels], 1)
    ctx = {"inputs": tok.vq.idx_to_ncm_input(canon), "flags": torch.rand(3, sum(levels[1:])),
           "targets": canon[:, 1:]}
    flag = w2m.FlagVector(tok.embed_dim)
    flag.w.data.fill_(0.7)
    got = w2m.flag_use(net, flag, prefix_fn(tok), ctx, "cpu", bs=3)
    with torch.no_grad(), w2m.fp32_eval(net, "cpu"):
        logits = [net(torch.cat([prefix_fn(tok)(3), inp], 1)) for inp in
                  (ctx["inputs"] + ctx["flags"][..., None] * flag.w, ctx["inputs"])]
        ce = [F.cross_entropy(l.flatten(0, 1), ctx["targets"].flatten(), reduction="none").view(3, -1).sum(1)
              for l in logits]
    assert got["delta_per_window"] == pytest.approx(float((ce[0] - ce[1]).mean()), abs=1e-4)


def test_w2m_notebooks_equal_generator_and_own_config_seed():
    nb = load_script("notebooks/make_notebooks.py")
    for arm, (cfg, _, _) in nb.W2M_ARMS.items():
        name = "w2m_L_ts_" + arm
        actual = json.loads((ROOT / "notebooks" / (name + ".ipynb")).read_text(encoding="utf-8"))
        assert actual == nb.make(name, nb.RUNS[name])
        assert nb.RUNS[name]["seeds"] == [0]
        assert len(nb.RUNS[name]["cmds"]) == 1
        assert "configs/scale/" + cfg + ".yaml" in nb.RUNS[name]["cmds"][0]
    name = "w2m_mech_eval"
    actual = json.loads((ROOT / "notebooks" / (name + ".ipynb")).read_text(encoding="utf-8"))
    assert actual == nb.make(name, nb.RUNS[name])
