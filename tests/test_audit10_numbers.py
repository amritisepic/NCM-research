"""Audit 10: reproduce the quoted baseline numbers from hash-pinned results.

These checks need no model downloads. Shared CPU setup comes from conftest.py.
The continuation/controller defect probes live in test_audit10_e1b.py.
"""
import hashlib
import importlib.util
import json
import math
import statistics

import pytest

from conftest import ROOT


PINS = {
    "results/colab/w2m_L_ts_m0/L_ts_m0_uniform_s0__result.json":
        "f0b084154c3201e43ac1966e9ac02c62aee6848305b175aae65a47141145e1eb",
    "results/colab/w2m_L_ts_m1/L_ts_m1_policy_s0__result.json":
        "343ca5cb5543033b57c23cc7dbfab12c42851af913578ef29f5d2c4a1d3c6291",
    "results/colab/w2m_L_ts_m2/L_ts_m2_frozen_pw_s0__result.json":
        "902fc2935e210201dcfc4410f965c2a5ed742f9634cfd199394c7f8e4e438f47",
    "results/colab/w2m_L_ts_m3/L_ts_m3_uniform_wrong_s0__result.json":
        "64ac4f32492ab3e5560cf1477857f7028eaadd6fae9b46b6cf5f4f7bcad39191",
    "results/colab/w2m_L_ts_m4/L_ts_m4_frozen_pw_flags_s0__result.json":
        "2c0b9171f0a033561243d9ac125e39eeffe329e5277824bbb15ac16780650cbc",
    "results/colab/w2m_L_ts_m6/L_ts_m6_policy_usedparent_s0__result.json":
        "e69c871933e94c91e4810c4cc649b5d27157b09a55f8959b52161bf2705fb298",
    "results/colab/w2m_L_ts_m7/L_ts_m7_uniform_a3rates_s0__result.json":
        "47b24c6eb894bf3609e89c20942f947f6a9e01d533a0ca83d9b1d0a23fa5c810",
    "results/colab/e1_equivalence/e1_sdk__result.json":
        "e936925fcb6935230f253ecc4220e12b33428dda3234db03ec8f1190b3df64e5",
    "results/colab/e1_equivalence/e1_ts__result.json":
        "5b134622a64121d9e835a8bae483c505768efbc63e68688a7e7ac5781942c5db",
    "results/colab/e1_equivalence/e1_wmt__result.json":
        "c66d29ed71fb17458413fb1ef683c54cb16413ca3b2d1bb8a131f14d2f6e478e",
    "results/w2m_sample_stats.json":
        "ddc0441dae4f9e3f9d17ebff63525f4e02443cc625173c3da78e430b82f8f7ce",
}


def pinned(rel):
    data = (ROOT / rel).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINS[rel], rel
    return json.loads(data)


def arm(name):
    rel = next(p for p in PINS if p.startswith(f"results/colab/w2m_L_ts_{name}/"))
    return pinned(rel)["final_eval"]


@pytest.mark.parametrize("rel", sorted(PINS))
def test_result_digests(rel):
    pinned(rel)


@pytest.mark.parametrize("name", ("m0", "m1", "m2", "m3", "m4", "m6", "m7"))
def test_w2m_saved_metrics_equal_the_sixteen_group_mean(name):
    result = arm(name)
    for scorer in ("gpt2-large", "TinyStories-33M"):
        key = f"genppl_{scorer}"
        groups = result[key + "_per_group"]
        assert len(groups) == 16
        assert statistics.mean(groups) == pytest.approx(result[key], abs=1e-12)


def test_registered_m2_m3_numbers_are_paired():
    m2, m3 = arm("m2"), arm("m3")
    key = "genppl_gpt2-large_per_group"
    delta = [a - b for a, b in zip(m2[key], m3[key])]
    assert statistics.mean(m2[key]) == pytest.approx(42.62201603601117)
    assert statistics.mean(m3[key]) == pytest.approx(21.685751521329436)
    assert statistics.mean(delta) == pytest.approx(20.936264514681724)
    assert statistics.stdev(delta) / math.sqrt(16) == pytest.approx(0.37397848490525193)
    assert m2["val_nll"] - m3["val_nll"] == pytest.approx(4.0353515624999545)


@pytest.mark.parametrize("name,entropy,mauve,gain,dup", [
    ("m0", 3.9889928771011984, 0.8651534635491999, 2.0537094162569147, 8.256599790840509),
    ("m1", 4.026305910544495, 0.4677402503328998, 1.6977889985475039, 18.215080212666702),
    ("m2", 4.017823559746418, 0.41180738767904934, 1.691359477684186, 21.268134848199754),
    ("m3", 3.991797690541272, 0.8653978107993214, 2.058692901232113, 8.298451597500067),
    ("m4", 4.018794832369645, 0.7199696693992106, 1.8943070655292849, 13.87485245673993),
    ("m6", 4.021290159583036, 0.45386873219826224, 1.6917471870559764, 18.753708741455448),
    ("m7", 3.9933030883981897, 0.860142341602631, 2.052358105949187, 8.34684672539375),
])
def test_saved_diversity_baselines(name, entropy, mauve, gain, dup):
    result = arm(name)
    stats = pinned("results/w2m_sample_stats.json")["sets"][name.upper()]
    assert result["entropy"] == pytest.approx(entropy)
    assert result["mauve"] == pytest.approx(mauve)
    assert stats["bigram_gain"] == pytest.approx(gain)
    assert stats["adjacent_dup_per_1k"] == pytest.approx(dup)


def load_e1b():
    spec = importlib.util.spec_from_file_location("audit10_e1b_numbers", ROOT / "scripts/e1b_continue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename,model,n,cuts,censored,wide", [
    ("e1_wmt__result.json", "J0", 128, 3, 18, 40),
    ("e1_ts__result.json", "A0", 64, 4, 6, 8),
    ("e1_sdk__result.json", "S0", 128, 4, 0, 0),
])
def test_e1b_selection_from_unchanged_e1a_delta(filename, model, n, cuts, censored, wide):
    module = load_e1b()
    data = pinned(f"results/colab/e1_equivalence/{filename}")
    assert len(data["rows"][model]) == n
    assert len([k for k in data["cuts"] if k >= 8]) == cuts
    delta = module.e1_delta(data)
    assert delta == pytest.approx(0.05 / (n * cuts))
    cens, widths = module.select_items(data["rows"][model], delta)
    assert (len(cens), len(widths)) == (censored, wide)


def test_wmt_literal_four_cut_denominator_selects_an_extra_item():
    """The registration's four-cut denominator conflicts with its count and unchanged-E1a rule."""
    module = load_e1b()
    data = pinned("results/colab/e1_equivalence/e1_wmt__result.json")
    rows = data["rows"]["J0"]
    cens, wide = module.select_items(rows, 0.05 / (128 * 4))
    _, e1a_wide = module.select_items(rows, module.e1_delta(data))
    assert (len(cens), len(wide)) == (18, 41)
    assert set(wide) - set(e1a_wide) == {43}
