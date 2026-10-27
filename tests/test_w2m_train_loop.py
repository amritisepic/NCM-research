"""W2M: the real train_gen.py loop on tiny CPU objects -- resume equals an uninterrupted run, RNG isolation, checkpoint
contents, sample-eval schedule, configs, FLOP accounting, frozen-copy order."""
import re
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from ncmr.w2m import A3_DISAGREEMENT, ARMS, PROPOSAL_FORWARDS, m7_level_rates
from w2m_support import Harness

ROOT = Path(__file__).resolve().parents[1]
ALL_ARMS = sorted(ARMS)
CONFIGS = {"m0": "L_ts_m0_uniform", "m1": "L_ts_m1_policy", "m2": "L_ts_m2_frozen_pw", "m3": "L_ts_m3_uniform_wrong",
           "m4": "L_ts_m4_frozen_pw_flags", "m5": "L_ts_m5_frozen_pw_shuffled", "m6": "L_ts_m6_policy_usedparent",
           "m7": "L_ts_m7_uniform_a3rates"}


def kw(arm):
    out = {"arm": arm}
    if arm == "m7":
        out["level_rates"] = m7_level_rates(A3_DISAGREEMENT, [32] * 12)
    if arm in ("m4", "m5"):
        out["flags_w"] = torch.full((8,), 0.2)
    return out


def assert_same_state(a, b):
    assert a["model"].keys() == b["model"].keys()
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert (a["flag"] is None) == (b["flag"] is None)
    if a["flag"] is not None:
        assert torch.equal(a["flag"], b["flag"])
    for k in ("mask_rng", "repl_rng", "flag_rng"):
        assert torch.equal(a["mech"][k], b["mech"][k]), k
    assert a["mech"]["stats"] == b["mech"]["stats"]
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert torch.equal(a["gen"], b["gen"]) and torch.equal(a["policy_gen"], b["policy_gen"])


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_resume_equals_an_uninterrupted_run(tmp_path, arm):
    whole = Harness(tmp_path / "whole", arm=arm, **{k: v for k, v in kw(arm).items() if k != "arm"}).run(4)
    first = Harness(tmp_path / "split", arm=arm, **{k: v for k, v in kw(arm).items() if k != "arm"}).run(2)
    assert (tmp_path / "split" / "ckpt.pt").exists() and first.trainer.step == 2
    rest = Harness(tmp_path / "split", arm=arm, resume=True, **{k: v for k, v in kw(arm).items() if k != "arm"})
    assert rest.trainer.step == 2
    rest.run(4)
    assert_same_state(whole.state(), rest.state())
    assert [r["loss"] for r in whole.losses()[2:]] == [r["loss"] for r in rest.losses()[2:]]
    assert len(whole.losses()) == 4 and len(rest.losses()) == 4
    assert whole.data_rng.bit_generator.state == rest.data_rng.bit_generator.state


def test_resume_is_sensitive_to_the_frozen_copy_being_taken_after_the_restore(tmp_path):
    """Negative control for the test above: a frozen proposer copied AFTER the restore is the trained model, and the
    resumed run then differs from the uninterrupted one."""
    k = {kk: v for kk, v in kw("m2").items() if kk != "arm"}
    whole = Harness(tmp_path / "whole", arm="m2", **k).run(4)
    Harness(tmp_path / "split", arm="m2", **k).run(2)
    bad = Harness(tmp_path / "split", arm="m2", resume=True, frozen_after_restore=True, **k).run(4)
    assert any(not torch.equal(whole.state()["model"][n], bad.state()["model"][n]) for n in whole.state()["model"])


def test_train_gen_copies_the_frozen_generator_before_any_checkpoint_restore():
    src = (ROOT / "scripts/train_gen.py").read_text(encoding="utf-8")
    code = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
    i_frozen = next(i for i, l in enumerate(code) if "frozen_copy(net)" in l)
    i_resume = next(i for i, l in enumerate(code) if "trainer.maybe_resume(" in l)
    assert i_frozen < i_resume


def test_checkpoint_holds_every_new_rng_and_the_flag_vector(tmp_path):
    h = Harness(tmp_path, arm="m5", **{k: v for k, v in kw("m5").items() if k != "arm"}).run(2)
    st = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    ex = st["extra"]
    assert {"corruption_rng", "policy_rng", "mech"} <= set(ex)
    assert {"mask_rng", "repl_rng", "flag_rng", "stats"} <= set(ex["mech"])
    assert set(st["modules"]) == {"model", "flag"} and st["modules"]["flag"]["w"].shape == (8,)
    assert torch.equal(st["modules"]["flag"]["w"], h.flagmod.w.detach())
    assert float(st["modules"]["flag"]["w"].abs().sum()) > 0
    assert ex["mech"]["stats"]["n"] == 16                         # two steps of 8 rows


def test_resume_without_the_mech_state_fails_closed(tmp_path):
    Harness(tmp_path, mode="onpolicy").run(2)                    # a checkpoint of an old arm (no "mech" in extra)
    with pytest.raises(RuntimeError, match="W2M"):
        Harness(tmp_path, arm="m1", resume=True)


def test_proposals_and_flags_leave_the_dropout_and_data_streams_identical_across_arms(tmp_path):
    ref = Harness(tmp_path / "m0", arm="m0").run(2)
    for arm in ALL_ARMS[1:]:
        h = Harness(tmp_path / arm, arm=arm, **{k: v for k, v in kw(arm).items() if k != "arm"}).run(2)
        assert torch.equal(ref.state()["torch_rng"], h.state()["torch_rng"]), arm
        assert ref.data_rng.bit_generator.state == h.data_rng.bit_generator.state, arm
        assert ref.mech.mask_gen.get_state().equal(h.mech.mask_gen.get_state()), arm


def test_curve_samples_follow_sample_eval_at_and_old_every_semantics(tmp_path):
    h = Harness(tmp_path / "a", arm="m0", at=(2, 3)).run(4)
    assert h.sample_calls == [2, 3]
    old = Harness(tmp_path / "b", mode="uniform", every=2).run(4)
    assert old.sample_calls == [2, 4]


def test_proposal_forwards_enter_the_flop_proxy():
    assert {a: PROPOSAL_FORWARDS[ARMS[a][0]] for a in ARMS} == {"m0": 0, "m1": 1, "m2": 1, "m3": 0, "m4": 1, "m5": 1, "m6": 1, "m7": 0}
    src = (ROOT / "scripts/train_gen.py").read_text(encoding="utf-8")
    assert re.search(r"fl_step = \(3 \+ .*mech\.proposal_forwards", src)


def test_configs_are_w1s_recipe_for_600_steps_and_differ_only_in_the_arm():
    ctrl = OmegaConf.load(ROOT / "configs/scale/L_ts_ctrl.yaml")
    arms = set()
    for arm, name in CONFIGS.items():
        c = OmegaConf.load(ROOT / f"configs/scale/{name}.yaml")
        assert c.corruption.mode == "mech" and c.corruption.arm == arm and c.init == "released"
        arms.add(c.corruption.arm)
        assert (c.corruption.mean_rate, c.corruption.spread, c.corruption.retarget) == (0.5, 0.5, False)
        assert c.corruption.clean_row_frac == 0.25
        assert c.train.steps == 600
        for k in ("batch_size", "grad_accum", "lr", "warmup", "min_lr_frac", "weight_decay", "beta2", "seed", "eval_n"):
            assert c.train[k] == ctrl.train[k], k
        assert list(c.sample_eval.at) == [100, 300, 600]
        assert c.train.eval_every == 100 and c.train.ckpt_every <= 100
        for k in ("n", "final_seeds", "final_n", "mauve"):
            assert c.sample_eval[k] == ctrl.sample_eval[k], k
        assert list(c.sample_eval.final_seeds) == [42] + list(range(15))
    assert arms == set(ARMS)
    m7 = OmegaConf.load(ROOT / f"configs/scale/{CONFIGS['m7']}.yaml")
    assert [float(v) for v in m7.corruption.d_target] == list(A3_DISAGREEMENT)
    assert all("d_target" not in OmegaConf.load(ROOT / f"configs/scale/{n}.yaml").corruption for a, n in CONFIGS.items() if a != "m7")
