"""Audit 12: pin the W2C inputs and independently verify the registered analysis.

The numerical/sign boundaries already live in test_r056_build.py. These tests
check the actual sixteen-group records and keep exploratory seed zero out of
the registered reading. No model download or GPU is needed.
"""
import hashlib
import importlib.util
import json
import math
import shutil
import statistics

import pytest

from conftest import ROOT


RUNS = {"m2": "L_ts_m2_frozen_pw", "m3": "L_ts_m3_uniform_wrong"}
PINS = {
    ("m2", 1): "d079dc520e2cbc1b93530c3701aafc7e90b99f3b03ec78ad50a9b3423109c989",
    ("m2", 2): "a837c0dd61bdee8413d693700cf844b8f85f9b711a9b7367690051eb3b85829e",
    ("m2", 3): "33b9c036e6ad57f51469ef5760f840c4d9d03cefcdf8a8672707c600c0fd2a9e",
    ("m3", 1): "bd6898e17fc1c3a63363a2ba2bfd97201f296b47da4640b971525a69c644d959",
    ("m3", 2): "f7761c8391762747f18df6848c7a18c081b45d0ad1043409175830d9fbcca15d",
    ("m3", 3): "98cf9e5d95c24e07cf88a4869b57e3f888ed6371510d7b73ffb02a0dda54f66c",
}
SAMPLE_SEEDS = [42, *range(15)]


def result_path(arm, seed):
    return ROOT / "results/colab" / f"w2c1_L_ts_{arm}" / f"{RUNS[arm]}_s{seed}__result.json"


def pinned_result(arm, seed):
    # Pin Git's LF bytes while allowing Windows checkout newline conversion.
    data = result_path(arm, seed).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == PINS[arm, seed]
    return json.loads(data)


def analyzer():
    spec = importlib.util.spec_from_file_location("audit12_analyze_w2c", ROOT / "scripts/analyze_w2c.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arm,seed", sorted(PINS))
def test_recorded_group_order_and_group_means(arm, seed):
    record = pinned_result(arm, seed)
    assert record["steps"] == 600
    assert record["config"]["train"]["seed"] == seed
    assert record["config"]["sample_eval"]["final_seeds"] == SAMPLE_SEEDS
    assert record["config"]["sample_eval"]["n"] == 256
    assert record["final_eval"]["n"] == 4096
    for scorer in ("gpt2-large", "TinyStories-33M"):
        key = f"genppl_{scorer}"
        values = record["final_eval"][key + "_per_group"]
        assert len(values) == len(SAMPLE_SEEDS)
        assert statistics.mean(values) == pytest.approx(record["final_eval"][key], abs=1e-12)


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_recorded_differences_and_paired_standard_errors(seed):
    module = analyzer()
    analysis = module.analyze_w2c(str(ROOT / "results/colab"), str(ROOT / "results"))
    saved = json.loads((ROOT / "results/w2c_analysis.json").read_text())["w2c_stage1"]
    row = analysis["per_seed"][seed]
    m2, m3 = [pinned_result(arm, seed)["final_eval"] for arm in ("m2", "m3")]
    for scorer, label in (("gpt2-large", "gpt2"), ("TinyStories-33M", "ts33m")):
        key = f"genppl_{scorer}_per_group"
        differences = [a - b for a, b in zip(m2[key], m3[key])]
        assert len(differences) == 16
        assert all(d > 0 for d in differences)
        assert row[f"D_{label}"] == pytest.approx(statistics.mean(differences), abs=1e-12)
        expected_se = statistics.stdev(differences) / 4
        assert row[f"D_{label}_se_paired"] == pytest.approx(expected_se, abs=1e-12)
        assert saved["per_seed"][str(seed)][f"D_{label}_se_paired"] == pytest.approx(expected_se, abs=1e-12)
    assert row["dNLL"] == pytest.approx(m2["val_nll"] - m3["val_nll"], abs=1e-12)
    assert row["dMAUVE"] == pytest.approx(m2["mauve"] - m3["mauve"], abs=1e-12)


def test_recorded_pooled_seed_sd_and_registered_readings():
    module = analyzer()
    analysis = module.analyze_w2c(str(ROOT / "results/colab"), str(ROOT / "results"))
    means = {
        arm: [statistics.mean(pinned_result(arm, seed)["final_eval"]["genppl_gpt2-large_per_group"])
              for seed in (1, 2, 3)]
        for arm in ("m2", "m3")
    }
    pooled = math.sqrt((statistics.variance(means["m2"]) + statistics.variance(means["m3"])) / 2)
    assert analysis["seeds_present"] == [1, 2, 3]
    assert analysis["pooled_seed_sd"] == pytest.approx(pooled, abs=1e-12)
    assert analysis["damage"]["reading"] == "replicates"
    assert all(analysis["damage"]["replicates_conditions"].values())
    assert not any(analysis["damage"]["overturned_conditions"].values())
    assert analysis["dissociation"] == "holds"
    assert analysis["criterion_c_candidate_claim"] is True


def test_exploratory_seed_zero_cannot_enter_registered_pool(tmp_path):
    module = analyzer()
    baseline = module.analyze_w2c(str(ROOT / "results/colab"), str(ROOT / "results"))
    for arm in ("m2", "m3"):
        dest = tmp_path / f"w2c1_L_ts_{arm}"
        dest.mkdir()
        for seed in (1, 2, 3):
            source = result_path(arm, seed)
            shutil.copyfile(source, dest / source.name)
        # A tempting fourth seed in the same directory must not affect any
        # mean, variance, SE, or reading. Its invalid JSON also detects reads.
        (dest / f"{RUNS[arm]}_s0__result.json").write_text("exploratory seed zero must not be read")
    actual = module.analyze_w2c(str(tmp_path), str(ROOT / "results"))
    for key in ("seeds_present", "pooled_seed_sd", "mean_D_gpt2", "mean_D_ts33m",
                "mean_dNLL", "mean_dMAUVE", "damage", "dissociation", "criterion_c_candidate_claim"):
        assert actual[key] == baseline[key]
