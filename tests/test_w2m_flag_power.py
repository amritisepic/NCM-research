"""Pins R043's flag-saturation and probe-confound numbers (log.md R043; docs/preregistration.md "W2Ma") to the simulation in
scripts/w2m_flag_power.py, each to its stated rounding. CPU, a few seconds."""
import importlib.util
import json
import os

import pytest
import torch

from ncmr import w2m as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("w2m_flag_power", os.path.join(ROOT, "scripts", "w2m_flag_power.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


@pytest.fixture(scope="module")
def sat():
    return P.saturation(4096, 0.25, 0)


@pytest.fixture(scope="module")
def probe():
    return P.probe_setting(4096, 5)


def test_saturation_setting_has_exactly_a_quarter_clean_rows(sat):
    assert sat["rows"] == 4096 and sat["clean_rows"] == 1024


def test_l0_replaced_in_about_756_percent_of_corrupted_rows(sat):
    # R043 states 75.6%; the simulation gives 75.65% (2324 of 3072 rows), which rounds to 75.7: a one-row difference, pinned as is
    assert round(sat["l0_replaced_corrupted_rows"] * 3072) == 2324
    assert abs(sat["l0_replaced_corrupted_rows"] * 100 - 75.6) < 0.1


def test_level1_flag_equals_l0_replaced_and_is_constant_within_a_row(sat):
    assert sat["level1_flag_constant_within_row"] is True
    assert sat["flag_eq1_per_level_corrupted_rows"][0] == sat["l0_replaced_corrupted_rows"]


def test_flags_are_at_least_969_from_level_2_and_100_percent_from_level_5(sat):
    per = sat["flag_eq1_per_level_corrupted_rows"]                     # index 0 = level 1
    assert len(per) == 11
    assert all(round(p * 100, 1) >= 96.9 for p in per[1:])
    assert round(per[1] * 100, 1) == 96.9
    assert all(round(p * 100, 1) == 100.0 for p in per[4:])            # levels 5..11


def test_996_percent_of_corrupted_row_flag_slots_are_1(sat):
    assert round(sat["flag_eq1_all_slots_corrupted_rows"] * 100, 1) == 99.6


def test_truthful_and_shuffled_flags_differ_by_00014_per_slot_in_corrupted_rows(sat):
    assert round(sat["mean_abs_truthful_minus_shuffled_corrupted_rows"], 4) == 0.0014


def test_probe_contexts_are_9968_percent_contaminated_with_20_negatives_per_window_91_percent_at_level_1(probe):
    assert probe["window_positions"] == 640
    assert round(probe["contaminated_fraction"] * 100, 2) == 99.68
    assert round(probe["negatives_per_window"], 1) == 2.0
    assert round(probe["negatives_at_level1_fraction"] * 100) == 91


def test_a_score_that_only_knows_level_1_reaches_auc_0952_and_0957_on_l0_clean_windows(probe):
    assert round(probe["auc_level_only_all_windows"]["indicator_not_level1"], 3) == 0.952
    assert round(probe["auc_level_only_l0_clean_windows"]["indicator_not_level1"], 3) == 0.957


def test_simulation_uses_the_repos_rates_and_mask_flags():
    from ncmr.corruption import CorruptionCfg
    r = CorruptionCfg(mode="mech", mean_rate=0.5, spread=0.5).rates(12)
    assert all(abs(a - b) < 1e-9 for a, b in zip(r, P.RATES))
    clean, masks = P.draw_masks(64, 0.25, 0)
    assert int(clean.sum()) == 16 and all(not bool(m[clean].any()) for m in masks) and not bool(masks[-1].any())
    f = W.mask_flags(P.stand_in_vq(), masks)
    assert f.shape == (64, sum(P.LEVELS[1:])) and float(f[clean].abs().sum()) == 0.0


def _close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == pytest.approx(b, abs=1e-3) if isinstance(a, float) else a == b


def test_the_committed_json_matches_the_simulation(sat, probe):
    with open(os.path.join(ROOT, "results", "w2m_flag_power.json")) as f:
        j = json.load(f)
    assert _close(j["saturation_25pct_clean_seed0"], json.loads(json.dumps(sat)))
    assert _close(j["probe_setting_no_clean_seed5"], json.loads(json.dumps(probe)))
