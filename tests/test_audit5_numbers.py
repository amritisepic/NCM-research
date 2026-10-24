"""Read-only recomputation of W1d's numerical motivation and R020 oracle shares."""
import json
import math
from pathlib import Path
from statistics import mean, stdev

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARMS = [
    ("w1_L_ts_ctrl", "L_ts_released16", 16.16, 0.11, 0.827),
    ("w1_L_ts_ctrl", "L_ts_ctrl_s0", 16.60, 0.08, 0.881),
    ("w1_L_ts_requant", "L_ts_requant_s0", 16.29, 0.09, 0.915),
    ("w1_L_ts_onpol", "L_ts_onpol_s0", 41.44, 0.43, 0.496),
    ("w1_L_ts_b1", "L_ts_b1_s0", 36.81, 0.26, 0.673),
    ("w1_L_ts_onpol2", "L_ts_onpol2_s0", 32.17, 0.18, 0.543),
    ("w1_L_ts_soft", "L_ts_soft_s0", 17.46, 0.10, 0.832),
]


def final_eval(folder, run):
    return json.loads(
        (ROOT / "results/colab" / folder / (run + "__result.json")).read_text(encoding="utf-8")
    )["final_eval"]


@pytest.mark.parametrize("folder,run,ppl,se,mauve", ARMS)
def test_r030_reported_sample_metrics(folder, run, ppl, se, mauve):
    result = final_eval(folder, run)
    groups = result["genppl_gpt2-large_per_group"]
    assert len(groups) == 16 and all(math.isfinite(x) for x in groups)
    assert mean(groups) == pytest.approx(result["genppl_gpt2-large"], abs=1e-10)
    assert stdev(groups) == pytest.approx(result["genppl_gpt2-large_sd"], abs=1e-10)
    assert round(mean(groups), 2) == ppl
    assert round(stdev(groups) / math.sqrt(len(groups)), 2) == se
    assert round(mean(result["mauve_per_seed"]), 3) == mauve


def test_w1d_motivation_uses_paired_sampling_error():
    a0, a1, a2 = [final_eval(*arm[:2]) for arm in ARMS[:3]]
    scores = [a["genppl_gpt2-large_per_group"] for a in (a0, a1, a2)]
    d21 = [a - b for a, b in zip(scores[2], scores[1])]
    d20 = [a - b for a, b in zip(scores[2], scores[0])]
    assert mean(d21) == pytest.approx(-0.3144907619, abs=1e-9)
    assert stdev(d21) / 4 == pytest.approx(0.0795606437, abs=1e-9)
    assert mean(d20) == pytest.approx(0.1253646980, abs=1e-9)
    assert stdev(d20) / 4 == pytest.approx(0.0757278711, abs=1e-9)
    assert a2["mauve"] - a1["mauve"] == pytest.approx(0.0335936929, abs=1e-9)
    assert a2["entropy"] - a1["entropy"] == pytest.approx(0.0001457625, abs=1e-9)


def test_r023_corrected_oracle_shares():
    result = json.loads(
        (ROOT / "results/colab/w0_calib_diag/diag_tf_levels_released_ts.json").read_text(encoding="utf-8")
    )
    p = {int(j): v["genppl_gpt2-large"] for j, v in result["by_j"].items()}
    gap = p[-1] - p[11]
    assert gap == pytest.approx(7.3812543732, abs=1e-9)
    assert 100 * (p[0] - p[5]) / gap == pytest.approx(86.3409270, abs=1e-6)
    assert 100 * (p[6] - p[11]) / gap == pytest.approx(7.2739105, abs=1e-6)
    assert 100 * (p[7] - p[10]) / gap == pytest.approx(3.9637418, abs=1e-6)
