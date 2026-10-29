"""Registered PD/PDa pilot readings from saved 175-step final evaluations.

No samples are generated. Group losses are log(per-group Gen-PPL), never the
log of the arithmetic mean Gen-PPL. Ratio uncertainty is a paired delete-one
jackknife over the sixteen shared groups.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCORERS = {"gpt2": "genppl_gpt2-large", "ts33m": "genppl_TinyStories-33M"}
GROUP_SEEDS = [42, *range(15)]


def paired_diff(a, b):
    """Mean and paired standard error of a minus b over aligned groups."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.shape != (16,) or not np.isfinite(d).all():
        raise ValueError("PD requires sixteen finite paired groups")
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def ratio_stats(p, d, u):
    """Ratio of mean contrasts and its paired delete-one jackknife SE."""
    p, d, u = (np.asarray(a, dtype=float) for a in (p, d, u))
    numerator, denominator = p - d, p - u
    gap = float(denominator.mean())
    if gap == 0:
        return None, None
    ratio = float(numerator.mean() / gap)
    leave_gap = (denominator.sum() - denominator) / (len(p) - 1)
    if np.any(leave_gap == 0):
        return ratio, None
    leave = ((numerator.sum() - numerator) / (len(p) - 1)) / leave_gap
    se = math.sqrt((len(p) - 1) / len(p) * float(((leave - leave.mean()) ** 2).sum()))
    return ratio, se


def read_pd(gap, r_gpt2, r_ts33m, dup_d, dup_p, mauve_d, mauve_p, entropy_d, entropy_u, coverage):
    """Apply PD's registered inclusive boundaries; coverage gates every reading."""
    valid = bool(gap >= 0.15)
    conditions = {
        "coverage_ge_0.80": coverage >= 0.80,
        "pilot_gap_ge_0.15": valid,
        "r_gpt2_ge_0.25": r_gpt2 is not None and r_gpt2 >= 0.25,
        "r_ts33m_ge_0.25": r_ts33m is not None and r_ts33m >= 0.25,
        "duplicates_le_P": dup_d <= dup_p,
        "mauve_ge_P_minus_0.02": mauve_d >= mauve_p - 0.02,
        "entropy_ge_U_minus_0.05": entropy_d >= entropy_u - 0.05,
        "r_gpt2_le_0.10": r_gpt2 is not None and r_gpt2 <= 0.10,
    }
    conditions = {key: bool(value) for key, value in conditions.items()}
    ratio_rescue = conditions["r_gpt2_ge_0.25"] and conditions["r_ts33m_ge_0.25"]
    diversity = conditions["mauve_ge_P_minus_0.02"] and conditions["entropy_ge_U_minus_0.05"]
    if not conditions["coverage_ge_0.80"]:
        reading = "inconclusive"
        reason = "coverage below registered gate"
    elif not valid:
        reading = "underpowered"
        reason = "same-step P-U log-loss contrast below registered gate"
    elif conditions["r_gpt2_le_0.10"] or (ratio_rescue and not diversity):
        reading = "killed"
        reason = "rescue at most 0.10 or ratio rescue requires a diversity loss"
    elif ratio_rescue and diversity and conditions["duplicates_le_P"]:
        reading = "rescue"
        reason = "all registered rescue conditions hold"
    else:
        reading = "inconclusive"
        reason = "neither registered rescue nor kill conditions hold"
    return {"reading": reading, "pilot_valid": valid, "conditions": conditions, "reason": reason}


def _config_key(config):
    """Cadence does not define training; sampling protocol remains checked."""
    config = json.loads(json.dumps(config))
    for key in ("eval_every", "ckpt_every", "log_every"):
        config.get("train", {}).pop(key, None)
    # The registered notebook uses --micro_batch64; the unmodified W2M
    # configuration uses 32. Both partitions are accepted only when matched
    # across arms (checked in analyze_results).
    accum = config.get("train", {}).get("grad_accum")
    if accum in (4, 8):
        config["train"]["grad_accum"] = "matched W2M partition"
    return config


def validate_result(result, arm):
    """Reject a substituted arm, seed, step count, training or sampling recipe."""
    if result.get("steps") != 175:
        raise ValueError(f"{arm}: PD requires the same-step 175-step reference")
    expected = yaml.safe_load((ROOT / f"configs/scale/L_ts_pd_{arm}.yaml").read_text(encoding="utf-8"))
    if _config_key(result.get("config", {})) != _config_key(expected):
        raise ValueError(f"{arm}: saved config differs from registered PD recipe")
    expected_arm = {"p": "m2", "d": "pd", "u": "m3"}[arm]
    if result.get("w2m_arm") != expected_arm:
        raise ValueError(f"{arm}: saved arm differs from PD recipe")
    if result.get("eval_precision") != "fp32 sampling and scoring, TF32 off":
        raise ValueError(f"{arm}: PD requires fp32 sampling/scoring with TF32 off")
    fe = result["final_eval"]
    if fe.get("n") != 16 * 256 or fe.get("nfe_per_sample") != 11:
        raise ValueError(f"{arm}: PD final evaluation requires 16 x 256 samples and 11 NFE")
    groups = {}
    for scorer, key in SCORERS.items():
        values = np.asarray(fe[key + "_per_group"], dtype=float)
        if values.shape != (16,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{arm}: sixteen positive finite groups required for {scorer}")
        if not math.isclose(float(values.mean()), float(fe[key]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{arm}: reported Gen-PPL differs from its group mean")
        groups[scorer] = np.log(values)
    for key in ("entropy", "mauve", "dup_adj1_per_1k"):
        if not math.isfinite(float(fe[key])):
            raise ValueError(f"{arm}: finite {key} required")
    return groups


def _coverage(result):
    coverage = result.get("pd", result.get("realized_disagreement", {}).get("pd_coverage"))
    if not isinstance(coverage, dict):
        raise ValueError("D: PD coverage counts are required")
    assigned = np.asarray(coverage.get("reassigned_per_level"), dtype=float)
    masked = np.asarray(coverage.get("masked_per_level"), dtype=float)
    if (assigned.shape != (12,) or masked.shape != (12,) or not np.isfinite(assigned).all()
            or not np.isfinite(masked).all() or np.any(assigned < 0) or np.any(assigned > masked)
            or np.any(assigned != np.floor(assigned)) or np.any(masked != np.floor(masked))
            or masked[2:6].sum() == 0):
        raise ValueError("D: invalid PD coverage counts")
    overall = float(assigned[2:6].sum() / masked[2:6].sum())
    if not math.isclose(float(coverage["overall"]), overall, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("D: reported coverage disagrees with counts")
    per_level = np.divide(assigned, masked, out=np.zeros(12), where=masked != 0)
    if not np.allclose(coverage["per_level"], per_level, rtol=1e-12, atol=1e-12):
        raise ValueError("D: per-level coverage disagrees with counts")
    return overall


def analyze_results(results):
    groups = {arm: validate_result(results[arm], arm) for arm in "pdu"}
    partitions = {results[arm]["config"]["train"]["grad_accum"] for arm in "pdu"}
    if len(partitions) != 1:
        raise ValueError("P/D/U must use the same micro-batch partition")
    coverage = _coverage(results["d"])
    scorers = {}
    for scorer in SCORERS:
        p, d, u = (groups[arm][scorer] for arm in "pdu")
        gap, gap_se = paired_diff(p, u)
        change, change_se = paired_diff(p, d)
        ratio, ratio_se = ratio_stats(p, d, u)
        scorers[scorer] = {
            "L": {arm: float(groups[arm][scorer].mean()) for arm in "pdu"},
            "L_se": {arm: float(groups[arm][scorer].std(ddof=1) / 4) for arm in "pdu"},
            "P_minus_U": gap, "P_minus_U_se_paired": gap_se,
            "P_minus_D": change, "P_minus_D_se_paired": change_se,
            "r": ratio, "r_se_paired_jackknife": ratio_se,
        }
    fe = {arm: results[arm]["final_eval"] for arm in "pdu"}
    reading = read_pd(scorers["gpt2"]["P_minus_U"], scorers["gpt2"]["r"], scorers["ts33m"]["r"],
                      fe["d"]["dup_adj1_per_1k"], fe["p"]["dup_adj1_per_1k"],
                      fe["d"]["mauve"], fe["p"]["mauve"], fe["d"]["entropy"], fe["u"]["entropy"], coverage)
    return {"available": True, "registration": "PD/PDa", "steps": 175, "seed": 0,
            "coverage": coverage, "coverage_per_level": results["d"]["pd"]["per_level"]
            if "pd" in results["d"] else results["d"]["realized_disagreement"]["pd_coverage"]["per_level"],
            "scorers": scorers, "entropy": {a: fe[a]["entropy"] for a in "pdu"},
            "mauve": {a: fe[a]["mauve"] for a in "pdu"},
            "duplicates_per_1k": {a: fe[a]["dup_adj1_per_1k"] for a in "pdu"}, **reading}


def analyze(root):
    root = Path(root)
    files = {arm: root / f"pd_L_ts_{arm}" / f"L_ts_pd_{arm}_s0__result.json" for arm in "pdu"}
    missing = [f"pd_L_ts_{arm}/L_ts_pd_{arm}_s0__result.json" for arm, path in files.items() if not path.is_file()]
    if missing:
        return {"available": False, "registration": "PD/PDa", "missing": missing}
    result = analyze_results({arm: json.loads(path.read_text(encoding="utf-8")) for arm, path in files.items()})
    result["files"] = {arm: f"pd_L_ts_{arm}/L_ts_pd_{arm}_s0__result.json" for arm in "pdu"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT / "results/colab"))
    parser.add_argument("--out", default=str(ROOT / "results/pd_analysis.json"))
    args = parser.parse_args()
    result = analyze(args.root)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print("PD:", result.get("reading", "not available"))


if __name__ == "__main__":
    main()
