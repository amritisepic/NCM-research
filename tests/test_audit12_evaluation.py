"""Audit 12: exercise production evaluation boundaries without scorer downloads.

These are positive integrity checks; no evaluation defect was found. The CPU
helpers execute the production function nodes, using the established W2M tiny
model harness. Published artifacts provide the actual six-run protocol evidence.
"""
import json
from collections import Counter
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from ncmr import data, eval_text
from ncmr.train import Trainer, fp32_eval
from w2m_support import prefix_fn, run_nodes, script_nodes, tiny_case


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SEEDS = [42, *range(15)]


def test_six_published_runs_have_identical_evaluation_inputs():
    manifests, protocol = [], []
    for arm, stem in (("m2", "frozen_pw"), ("m3", "uniform_wrong")):
        for seed in (1, 2, 3):
            folder = ROOT / "results/colab" / f"w2c1_L_ts_{arm}"
            prefix = folder / f"L_ts_{arm}_{stem}_s{seed}"
            read = lambda suffix: json.loads(Path(str(prefix) + suffix).read_text(encoding="utf-8"))
            result, samples = read("__result.json"), read("__samples_final.json")
            manifests.append(read("__run_manifest.json"))
            config, final = result["config"], result["final_eval"]
            assert result["steps"] == 600 and config["train"]["seed"] == seed
            assert result["eval_precision"] == "fp32 sampling and scoring, TF32 off"
            protocol.append((config["sample_eval"], config["train"]["eval_every"],
                             config["train"]["eval_n"], config["train"]["grad_accum"]))
            assert config["sample_eval"]["final_seeds"] == SAMPLE_SEEDS
            assert config["sample_eval"]["final_n"] == 256
            assert config["sample_eval"]["at"] == [100, 300, 600]
            assert config["train"]["eval_n"] == 1000
            assert config["train"]["grad_accum"] == 4
            assert samples["groups"] == [s for s in SAMPLE_SEEDS for _ in range(256)]
            assert Counter(samples["groups"]) == dict.fromkeys(SAMPLE_SEEDS, 256)
            assert len(samples["ids"]) == final["n"] == 4096
            assert final["mauve_n"] == [4096, 4096]
            assert len(final["val_nll_per_level"]) == 11
            for scorer in ("gpt2-large", "TinyStories-33M"):
                groups = final[f"genppl_{scorer}_per_group"]
                assert len(groups) == 16
                assert sum(groups) / 16 == pytest.approx(final[f"genppl_{scorer}"])
            events = [json.loads(line) for line in
                      Path(str(prefix) + "__provenance.jsonl").read_text(encoding="utf-8").splitlines()]
            assert [e["event"] for e in events] == ["start", "end"]
            assert events[0]["had_ckpt"] is False
            assert events[0]["micro_batch_64"] is True
            assert events[1]["exit"] == 0 and events[1]["last_logged_step"] == 600
    assert all(m == manifests[0] for m in manifests)
    assert all(p == protocol[0] for p in protocol)


def test_per_group_scores_use_sorted_sample_seed_order(monkeypatch):
    """Saved blocks start at seed 42; score vectors instead use 0..14,42."""
    seen = []

    def scorer(texts, name, device):
        seen.append((name, texts))
        return float(texts[0])

    monkeypatch.setattr(eval_text, "gen_ppl", scorer)
    ids = [[s] for s in SAMPLE_SEEDS for _ in range(2)]
    groups = [s for s in SAMPLE_SEEDS for _ in range(2)]
    result = eval_text.evaluate_samples(ids, lambda s: str(s[0]), groups,
                                        device="cpu", mauve_amp=False)
    expected = list(map(float, sorted(SAMPLE_SEEDS)))
    for scorer_name in ("gpt2-large", "TinyStories-33M"):
        assert result[f"genppl_{scorer_name}_per_group"] == expected
    assert len(seen) == 32 and all(len(texts) == 2 for _, texts in seen)


@pytest.mark.parametrize("raises", [False, True])
def test_sample_metrics_isolates_rng_and_tf32_even_on_failure(raises):
    """Exercise the wrapper around all generator, scorer and MAUVE calls."""
    funcs, _, _ = script_nodes()
    seen = []

    def inner(n, seeds, mauve):
        seen.append((torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.allow_tf32,
                     torch.is_autocast_enabled("cpu")))
        torch.manual_seed(999)
        torch.rand(20)
        if raises:
            raise RuntimeError("probe")
        return n, seeds, mauve

    ns = {"torch": torch, "device": "cpu", "_sample_metrics": inner}
    run_nodes([funcs["sample_metrics"]], ns)
    rng = torch.get_rng_state().clone()
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        if raises:
            with pytest.raises(RuntimeError, match="probe"):
                ns["sample_metrics"](256, SAMPLE_SEEDS, True)
        else:
            assert ns["sample_metrics"](256, SAMPLE_SEEDS, True) == (256, SAMPLE_SEEDS, True)
        assert seen == [(False, False, False)]
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def test_clean_nll_scores_every_trainable_level_in_fp32(tmp_path, monkeypatch):
    tok, net = tiny_case()
    levels = list(map(int, tok.vq.levels))
    # Distinct artificial stories make parity errors and changed item order visible.
    stories = [[i + 2] * 128 for i in range(12)]
    monkeypatch.setattr(data, "valid_stories", lambda: stories)
    val = torch.from_numpy(data.examples("val", 6))
    test = torch.from_numpy(data.examples("test", 6))
    assert val[:, 0].tolist() == [2, 4, 6, 8, 10, 12]
    assert test[:, 0].tolist() == [3, 5, 7, 9, 11, 13]
    params = [p for p in net.parameters() if p.requires_grad]
    trainer = Trainer(str(tmp_path), params, 1e-4, 2, amp=True)
    prefix = prefix_fn(tok)
    seen = []
    hook = net.register_forward_hook(lambda mod, inp, out: seen.append(
        (out.dtype, torch.is_autocast_enabled("cpu"),
         torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         any(m.training for m in mod.modules()))))
    funcs, _, _ = script_nodes()
    ns = dict(torch=torch, F=F, K=len(levels), levels=levels, val_x=val,
              fp32_eval=fp32_eval, net=net, fnet=net, tok=tok,
              device="cpu", prefix_emb=prefix)
    run_nodes([funcs["val_nll"]], ns)
    net.train()
    try:
        for _ in range(3):
            # Repeated fresh Trainer contexts exercise the former R024 leak class.
            with trainer.ctx:
                with trainer.ctx:
                    result = ns["val_nll"]()
                    assert torch.is_autocast_enabled("cpu")
            assert not torch.is_autocast_enabled("cpu")
        assert seen == [(torch.float32, False, False, False, False)] * 3
        assert net.training
    finally:
        hook.remove()
    with torch.no_grad(), fp32_eval(net, "cpu"):
        inputs, target = tok.get_ncm_inputs(val)
        logits = net(torch.cat([prefix(len(val)), inputs], 1)).float()
        losses = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
        per_level = [float(v.sum() / len(val)) for v in losses.split(levels[1:], 1)]
    assert result["val_nll"] == pytest.approx(sum(per_level))
    assert result["val_nll_per_level"] == pytest.approx(per_level, abs=0.0051)
    assert sum(result["val_nll_per_level"]) == pytest.approx(result["val_nll"], abs=0.0051 * (len(levels) - 1))
