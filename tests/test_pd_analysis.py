"""PD's same-step analysis and every registered boundary, CPU only."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_pd_test", ROOT / "scripts/analyze_pd.py")
PD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PD)


def reading(**kw):
    args = dict(gap=0.15, r_gpt2=0.25, r_ts33m=0.25, dup_d=10.0, dup_p=10.0,
                mauve_d=0.78, mauve_p=0.80, entropy_d=3.95, entropy_u=4.0, coverage=0.80)
    args.update(kw)
    return PD.read_pd(**args)


def test_pd_validity_includes_point15_and_refuses_below():
    assert reading()["pilot_valid"] is True
    below = reading(gap=np.nextafter(0.15, -math.inf))
    assert below["pilot_valid"] is False and below["reading"] == "underpowered"
    assert reading(gap=0.149, r_gpt2=0.0)["reading"] == "underpowered"


def test_pd_rescue_includes_every_registered_equality():
    assert reading()["reading"] == "rescue"


@pytest.mark.parametrize("key", ["r_gpt2", "r_ts33m"])
def test_pd_rescue_requires_point25_under_both_scorers(key):
    assert reading(**{key: np.nextafter(0.25, -math.inf)})["reading"] == "inconclusive"


def test_pd_kill_includes_point10_but_not_above():
    assert reading(r_gpt2=0.10)["reading"] == "killed"
    assert reading(r_gpt2=np.nextafter(0.10, math.inf))["reading"] == "inconclusive"


@pytest.mark.parametrize("key,floor", [("entropy_d", 3.95), ("mauve_d", 0.78)])
def test_pd_diversity_kill_only_beyond_registered_margin(key, floor):
    assert reading(**{key: floor})["reading"] == "rescue"
    assert reading(**{key: np.nextafter(floor, -math.inf)})["reading"] == "killed"
    # A partial ratio rescue remains inconclusive; it is not the registered kill.
    assert reading(r_ts33m=0.20, **{key: floor - 0.1})["reading"] == "inconclusive"


def test_pd_duplicates_above_P_preclude_rescue_but_are_not_a_kill():
    assert reading(dup_d=np.nextafter(10.0, math.inf))["reading"] == "inconclusive"


def test_pd_coverage_includes_point80_and_gates_any_reading_below():
    assert reading()["reading"] == "rescue"
    assert reading(coverage=np.nextafter(0.80, -math.inf), r_gpt2=0.0)["reading"] == "inconclusive"


def results():
    out = {}
    noise = np.linspace(-0.2, 0.2, 16)
    for arm, loss in (("p", 3.0), ("d", 2.9), ("u", 2.6)):
        config = yaml.safe_load((ROOT / f"configs/scale/L_ts_pd_{arm}.yaml").read_text(encoding="utf-8"))
        fe = {"n": 4096, "nfe_per_sample": 11, "entropy": 4.0, "mauve": 0.8, "dup_adj1_per_1k": 10.0}
        for key in PD.SCORERS.values():
            values = np.exp(loss + noise)
            fe[key + "_per_group"] = values.tolist()
            fe[key] = float(values.mean())
        out[arm] = {"steps": 175, "config": config, "w2m_arm": {"p": "m2", "d": "pd", "u": "m3"}[arm],
                    "eval_precision": "fp32 sampling and scoring, TF32 off", "final_eval": fe}
    out["d"]["pd"] = {"overall": 0.8, "per_level": [0, 0, .8, .8, .8, .8, 0, 0, 0, 0, 0, 0],
                       "masked_per_level": [0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0],
                       "reassigned_per_level": [0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0]}
    return out


def test_pd_means_group_log_ppl_before_averaging_and_pairs_contrasts():
    data = results()
    analysis = PD.analyze_results(data)
    for score in analysis["scorers"].values():
        assert score["L"]["p"] == pytest.approx(3.0)
        assert score["L"]["p"] != pytest.approx(math.log(data["p"]["final_eval"][PD.SCORERS["gpt2"]]))
        assert score["P_minus_U"] == pytest.approx(.4)
        assert score["r"] == pytest.approx(.25)
        assert score["P_minus_U_se_paired"] == pytest.approx(0, abs=1e-15)
        assert score["r_se_paired_jackknife"] == pytest.approx(0, abs=1e-15)
    # The exact ratio can straddle .25 by rounding; boundary readings are tested above.
    assert analysis["available"] and analysis["coverage"] == .8


def test_pd_ratio_se_is_paired_jackknife():
    p = np.linspace(3, 4, 16)
    u = np.linspace(2, 2.8, 16)
    d = p - np.linspace(.1, .7, 16)
    ratio, se = PD.ratio_stats(p, d, u)
    leaves = [((np.delete(p, i) - np.delete(d, i)).mean()
               / (np.delete(p, i) - np.delete(u, i)).mean()) for i in range(16)]
    assert ratio == pytest.approx((p - d).mean() / (p - u).mean())
    assert se == pytest.approx(math.sqrt(15 / 16 * np.square(leaves - np.mean(leaves)).sum()))


@pytest.mark.parametrize("mutation", ["steps", "arm", "seed", "base", "groups", "precision", "coverage", "partition"])
def test_pd_refuses_unregistered_or_inconsistent_inputs(mutation):
    data = results()
    if mutation == "steps":
        data["u"]["steps"] = 600
    elif mutation == "arm":
        data["d"]["w2m_arm"] = "m3"
    elif mutation == "seed":
        data["p"]["config"]["train"]["seed"] = 1
    elif mutation == "base":
        data["p"]["config"]["init"] = "scratch"
    elif mutation == "groups":
        data["p"]["final_eval"][PD.SCORERS["gpt2"] + "_per_group"].pop()
    elif mutation == "precision":
        data["p"]["eval_precision"] = "bf16"
    elif mutation == "coverage":
        data["d"]["pd"]["overall"] = 1.0
    elif mutation == "partition":
        data["p"]["config"]["train"]["grad_accum"] = 4 if data["d"]["config"]["train"]["grad_accum"] == 8 else 8
    with pytest.raises(ValueError):
        PD.analyze_results(data)


def test_pd_absent_result_reports_missing_inputs(tmp_path):
    result = PD.analyze(tmp_path)
    assert result["available"] is False and len(result["missing"]) == 3
    assert all(not Path(path).is_absolute() for path in result["missing"])
