"""Audit 9: independent arithmetic on immutable result blobs; no model downloads."""
import hashlib
import json
from pathlib import Path
from statistics import mean

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "J0": "w2c_L_wmt_ctrl/L_wmt_released",
    "J1": "w2c_L_wmt_ctrl/L_wmt_ctrl_s0",
    "J2": "w2c_L_wmt_consist/L_wmt_consist_s0",
    "J2-F": "w2c_L_wmt_consist_frozen/L_wmt_consist_frozen_s0",
}
PINS = {
    "results/s0_neighbour_gain.json": "00d9ed73efec8260c5b94beb946ff3cd081647e71d948937351175b0c3e3edd5",
    "results/w2m_flag_power.json": "d702cfbcb87d23c130c8170a320884f624184760e6f6d6d51911535197621a3a",
    "results/j2f_stop_rule_val_s0.json": "48f9805ed865acc36ca322edbd4894e46e4d73f9d4d37840bcb7113da2be14ca",
}
PINS.update({"results/colab/" + ARMS[a] + "__hyp_val.json": h for a, h in {
    "J0": "646d186fd6b27a00aadf71a3db485e3636481875497e1ec4f1ba92b52e20701a",
    "J1": "796be9370c68387ec627a29fa7a2710198992a64251ea11eaa984a8e8b79f184",
    "J2": "d7774e8693f8fe0bdfce480e187610d693ff174df2fc3d41a4a0382ab9cd7693",
    "J2-F": "819371e8dd731c76ac9ef546bb1cb8732e017189eef691f6e5e2cd4bea60c5dd",
}.items()})


def read_pinned(path):
    raw = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == PINS[path], path
    return json.loads(raw)


def independent_rule(h, split, m):
    out = []
    for i, exact in enumerate(h["nfe_" + split]):
        cap = h["hyp_" + split + "_capped"]
        assert cap[str(exact)][i] == h["hyp_" + split][i]
        stop = exact
        for t in range(m + 1, exact + 1):
            if len({cap[str(t - d)][i] for d in range(m + 1)}) == 1:
                stop = t
                break
        out.append((stop, cap[str(stop)][i]))
    return out


@pytest.mark.parametrize("arm,want", [
    ("J0", (9.024, 10.045)), ("J1", (9.759, 10.707333333333333)),
    ("J2", (7.1306666666666665, 8.313)), ("J2-F", (7.799333333333333, 9.083333333333334)),
])
def test_r042_passes_independently_reproduce_both_rules(arm, want):
    from ncmr.cached_decode import stable_stop
    h = read_pinned("results/colab/" + ARMS[arm] + "__hyp_val.json")
    record = read_pinned("results/j2f_stop_rule_val_s0.json")["arms"][arm]
    assert len(h["nfe_val"]) == 3000
    for m, expected in enumerate(want, 1):
        outputs = independent_rule(h, "val", m)
        assert mean(t for t, _ in outputs) == expected == record[f"stable x{m}"]["passes"]
        for i, exact in enumerate(h["nfe_val"]):
            texts = [h["hyp_val_capped"][str(t)][i] for t in range(1, exact + 1)]
            assert stable_stop(texts, exact, m) == outputs[i]


def test_r045_arithmetic_and_gate_from_level_results():
    s = read_pinned("results/s0_neighbour_gain.json")
    for k, row in s["levels"].items():
        assert row["val_windows"] == 1024
        assert row["base_nll_mean"] - row["with_term_nll_mean"] == pytest.approx(row["gain_mean"], abs=1e-10)
        assert row["gain_per_code"] == row["gain_mean"] / row["positions"]
        assert row["gain_pct_of_base"] == 100 * row["gain_mean"] / row["base_nll_mean"]
        assert row["best_epoch"] == min(range(len(row["dev_nll_per_epoch"])), key=row["dev_nll_per_epoch"].__getitem__) + 1
    base = sum(s["levels"][str(k)]["base_nll_mean"] for k in range(1, 6))
    gain = sum(s["levels"][str(k)]["gain_mean"] for k in range(1, 6))
    # The aggregate was summed in float32 over windows before taking its double mean.
    assert base == pytest.approx(s["sum_levels_1to5"]["base_nll_mean"], abs=2e-8)
    assert gain == pytest.approx(6.217371253296733, abs=1e-12)
    assert gain < 10 and gain < 0.05 * base
    assert s["gate"]["part1"]["verdict"] == "fail"
    assert round(s["levels"]["4"]["gain_per_code"], 3) == 0.015
    assert sum(r["gain_mean"] > 1 for r in s["levels"].values()) == 2


def test_r043_arithmetic_from_saved_simulation():
    s = read_pinned("results/w2m_flag_power.json")
    sat, probe = s["saturation_25pct_clean_seed0"], s["probe_setting_no_clean_seed5"]
    assert sat["rows"] - sat["clean_rows"] == 3072
    assert round(sat["l0_replaced_corrupted_rows"] * 3072) == 2324
    assert round(100 * sat["l0_replaced_corrupted_rows"], 1) == 75.7
    assert round(100 * sat["flag_eq1_all_slots_corrupted_rows"], 1) == 99.6
    assert round(sat["mean_abs_truthful_minus_shuffled_corrupted_rows"], 4) == 0.0014
    assert probe["negatives_per_window"] == pytest.approx((1 - probe["contaminated_fraction"]) * 640, abs=2e-5)
    assert round(100 * probe["contaminated_fraction"], 2) == 99.68
    assert round(100 * probe["negatives_at_level1_fraction"]) == 91
    assert round(probe["auc_level_only_all_windows"]["indicator_not_level1"], 3) == 0.952
    assert round(probe["auc_level_only_l0_clean_windows"]["indicator_not_level1"], 3) == 0.957
