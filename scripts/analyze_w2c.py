"""W2C stage 1, M3-0 and L1R (docs/preregistration.md, R056 package): the registered readings, computed from the saved
`results/colab/<notebook>/<run>__result.json` files (nothing is re-sampled). Each reading is computed only when all of its
inputs exist; otherwise it is reported as "not available" with the missing files, and nothing crashes.

  python scripts/analyze_w2c.py [--root results/colab] [--stats_dir results] [--out results/w2c_analysis.json]
                                [--l1r_coverage 0.93]

Inputs (notebook name = directory under results/colab/):
  W2C stage 1  w2c1_L_ts_m2 / w2c1_L_ts_m3: L_ts_m2_frozen_pw_s{1,2,3}, L_ts_m3_uniform_wrong_s{1,2,3};
               duplicates per 1k from results/w2c1_sample_stats.json (`scripts/w1_sample_stats.py --set w2c1`, as registered).
  M3-0         w2p_L_ts_m3_c0: L_ts_m3_uniform_wrong_c0_s0, against W2M's seed-0 M3 (w2m_L_ts_m3).
  L1R          l1r_L_ts_m2r: L_ts_m2r_frozen_pw_l1reassign_s0, against W2M's seed-0 M2 and M3; duplicates per 1k from
               results/l1r_sample_stats.json (`--set l1r`); the coverage gate from the run's result.json (`l1r_coverage`,
               `coverage` or `l1r.coverage`) or --l1r_coverage.
Groups: the 16 final-eval sample groups (sample seeds 42 and 0-14) are the same for every run, so differences are paired.
Thresholds are the registration's; the reading functions are pure (tests/test_r056_build.py checks them at the boundaries).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPT2, TS33 = "genppl_gpt2-large", "genppl_TinyStories-33M"
SEEDS = (1, 2, 3)

# W2C stage 1 (registered)
D_EACH_MIN = 10.0            # damage replicates: every D_s >= 10
D_MEAN_SD_FACTOR = 2.0       # ... and mean D > 2 s_p
D_MEAN_OVERTURN = 5.0        # overturned if mean D < 5
NLL_HOLDS = 10.0             # dissociation holds if |mean dNLL| <= 10 nats/window
NLL_REMOVED = 20.0           # removed if mean dNLL > 20
# M3-0 (registered)
I_PREDICTION = 3.0           # I >= 3: Astra A1's prediction; I < 1: the clean rows are weakened as the principal cause
I_WEAKENED = 1.0
# L1R (registered)
RESCUE_MET = 0.25            # >= 25% rescue under both scorers
RESCUE_KILL = 0.10           # <= 10% (GPT-2 large): killed
ENTROPY_TOL = 0.05           # entropy >= M3's - 0.05 (3.94)
COVERAGE_MIN = 0.80          # run-time coverage gate


# ------------------------------------------------------------------------------------------------ pure functions
def paired_diff(a, b):
    """Mean and paired SE (sd of the group-wise differences / sqrt(n)) of a - b over groups shared by the two runs."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    assert d.shape == np.shape(a) and len(d) > 1, "paired groups must align"
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def pooled_seed_sd(g_m2, g_m3):
    """s_p = sqrt((var_s G(M2_s) + var_s G(M3_s)) / 2): sample variances (ddof = 1) over the seeds of the per-seed Gen-PPL."""
    return math.sqrt((float(np.var(g_m2, ddof=1)) + float(np.var(g_m3, ddof=1))) / 2.0)


def read_damage(d_s, d_ts_s, ddup_s, s_p, mean_d_ts, mean_dmauve):
    """W2C damage reading from per-seed D_s (GPT-2 large), per-seed TinyStories-33M D_s and dDup_s, the pooled seed SD,
    the seed-mean TinyStories-33M difference and the seed-mean MAUVE difference (M2 - M3).

    replicates: every D_s >= 10, mean D > 2 s_p, every seed D_s(TS-33M) > 0 and dDup_s > 0.
    overturned: mean D < 5, or the seed-mean M2 - M3 difference has the opposite sign (M2 better) under TinyStories-33M
    (mean D_ts < 0) or MAUVE (mean dMAUVE > 0; higher MAUVE is better). The registration leaves the case where both the
    replicates conditions and a MAUVE reversal hold open; here overturned wins (the claim is not made on a reversed score).
    Anything else: mixed."""
    mean_d = float(np.mean(d_s))
    rep = {"every_D_ge_10": all(d >= D_EACH_MIN for d in d_s), "mean_D_gt_2sp": mean_d > D_MEAN_SD_FACTOR * s_p,
           "every_D_ts_gt_0": all(d > 0 for d in d_ts_s), "every_dDup_gt_0": all(d > 0 for d in ddup_s)}
    over = {"mean_D_lt_5": mean_d < D_MEAN_OVERTURN, "ts33m_reversed": mean_d_ts < 0, "mauve_reversed": mean_dmauve > 0}
    if any(over.values()):
        reading = "overturned"
    elif all(rep.values()):
        reading = "replicates"
    else:
        reading = "mixed"
    return {"reading": reading, "mean_D": mean_d, "replicates_conditions": rep, "overturned_conditions": over}


def read_dissociation(mean_dnll):
    """holds if |mean dNLL| <= 10 nats/window; removed if mean dNLL > 20; else not established."""
    if abs(mean_dnll) <= NLL_HOLDS:
        return "holds"
    if mean_dnll > NLL_REMOVED:
        return "removed"
    return "not established"


def read_m30(i_mean):
    """I = G(M3, seed 0) - G(M3-0): >= 3 prediction; < 1 weakened; between inconclusive."""
    if i_mean >= I_PREDICTION:
        return "prediction"
    if i_mean < I_WEAKENED:
        return "weakened"
    return "inconclusive"


def rescue(g_m2, g_m2r, g_m3):
    """r = (G(M2) - G(M2R)) / (G(M2) - G(M3)); None when M2 is not worse than M3 (no gap to rescue)."""
    gap = g_m2 - g_m3
    return None if gap <= 0 else (g_m2 - g_m2r) / gap


def read_l1r(r_gpt2, r_ts, entropy_m2r, entropy_m3, mauve_m2r, mauve_m2, dup_m2r, dup_m2, coverage):
    """L1R reading. r_*: rescue per scorer (None if undefined); dup_*: duplicates per 1k (None if not yet measured);
    coverage: selected / root-corrupted rows (None if not recorded).
    killed: coverage < 0.80, or r_gpt2 <= 0.10, or a gain (r_gpt2 > 0.10) with entropy < M3's - 0.05.
    met: r >= 0.25 under both scorers, entropy >= M3's - 0.05, MAUVE >= M2's, duplicates per 1k <= M2's.
    Otherwise between (reported, no expansion); `incomplete` when everything else is met but duplicates per 1k are missing.
    A run that produced a result passed the run-time coverage gate, so a missing coverage value does not block `met`."""
    ent_floor = entropy_m3 - ENTROPY_TOL
    conds = {"coverage_ge_0.80": None if coverage is None else coverage >= COVERAGE_MIN,
             "r_gpt2_le_0.10": None if r_gpt2 is None else r_gpt2 <= RESCUE_KILL,
             "entropy_ge_floor": entropy_m2r >= ent_floor,
             "r_gpt2_ge_0.25": None if r_gpt2 is None else r_gpt2 >= RESCUE_MET,
             "r_ts33m_ge_0.25": None if r_ts is None else r_ts >= RESCUE_MET,
             "mauve_ge_m2": mauve_m2r >= mauve_m2,
             "dup_le_m2": None if dup_m2r is None or dup_m2 is None else dup_m2r <= dup_m2}
    base = [bool(conds[k]) for k in ("r_gpt2_ge_0.25", "r_ts33m_ge_0.25", "entropy_ge_floor", "mauve_ge_m2")]
    if conds["coverage_ge_0.80"] is False or conds["r_gpt2_le_0.10"] or (
            r_gpt2 is not None and r_gpt2 > RESCUE_KILL and not conds["entropy_ge_floor"]):
        reading = "killed"
    elif r_gpt2 is None:
        reading = "not available"
    elif not all(base) or conds["dup_le_m2"] is False:
        reading = "between"
    else:
        reading = "met" if conds["dup_le_m2"] else "incomplete"
    return {"reading": reading, "entropy_floor": ent_floor, "conditions": conds}


# ------------------------------------------------------------------------------------------------ loading
def shown(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/") if os.path.abspath(path).startswith(ROOT) else os.path.basename(path)


def load_arm(root, notebook, run):
    """Final-eval summary of one run, or None when its result.json is absent."""
    path = os.path.join(root, notebook, f"{run}__result.json")
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    fe = r["final_eval"]
    g, t = np.array(fe[f"{GPT2}_per_group"], dtype=float), np.array(fe[f"{TS33}_per_group"], dtype=float)
    assert abs(g.mean() - fe[GPT2]) < 1e-9 and abs(t.mean() - fe[TS33]) < 1e-9, f"{run}: reported Gen-PPL is not the group mean"
    return {"file": shown(path), "steps": r.get("steps"), "gpt2_groups": g, "ts33m_groups": t, "gpt2": float(g.mean()),
            "ts33m": float(t.mean()), "val_nll": fe["val_nll"], "entropy": fe["entropy"], "mauve": fe["mauve"],
            "dup_eval": fe.get("dup_adj1_per_1k"), "result": r}


def load_dup(stats_dir, stats_set, key):
    """adjacent_dup_per_1k of `key` in results/<set>_sample_stats.json (scripts/w1_sample_stats.py), or None."""
    path = os.path.join(stats_dir, f"{stats_set}_sample_stats.json")
    if not os.path.exists(path):
        return None
    s = json.load(open(path))["sets"].get(key)
    return None if s is None else s["adjacent_dup_per_1k"]


def _strip(arm):
    return None if arm is None else {k: v for k, v in arm.items() if k not in ("result", "gpt2_groups", "ts33m_groups")}


# ------------------------------------------------------------------------------------------------ the three analyses
def analyze_w2c(root, stats_dir):
    m2 = {s: load_arm(root, "w2c1_L_ts_m2", f"L_ts_m2_frozen_pw_s{s}") for s in SEEDS}
    m3 = {s: load_arm(root, "w2c1_L_ts_m3", f"L_ts_m3_uniform_wrong_s{s}") for s in SEEDS}
    per_seed, missing = {}, []
    for s in SEEDS:
        a, b = m2[s], m3[s]
        if a is None:
            missing.append(f"{shown(root)}/w2c1_L_ts_m2/L_ts_m2_frozen_pw_s{s}__result.json")
        if b is None:
            missing.append(f"{shown(root)}/w2c1_L_ts_m3/L_ts_m3_uniform_wrong_s{s}__result.json")
        if a is None or b is None:
            continue
        d, d_se = paired_diff(a["gpt2_groups"], b["gpt2_groups"])
        dts, dts_se = paired_diff(a["ts33m_groups"], b["ts33m_groups"])
        du2, du3 = load_dup(stats_dir, "w2c1", f"M2_s{s}"), load_dup(stats_dir, "w2c1", f"M3_s{s}")
        per_seed[s] = {"D_gpt2": d, "D_gpt2_se_paired": d_se, "D_ts33m": dts, "D_ts33m_se_paired": dts_se,
                       "dNLL": a["val_nll"] - b["val_nll"], "dMAUVE": a["mauve"] - b["mauve"],
                       "dDup": None if du2 is None or du3 is None else du2 - du3,
                       "G_m2": a["gpt2"], "G_m3": b["gpt2"], "m2": _strip(a), "m3": _strip(b)}
    out = {"seeds_present": sorted(per_seed), "per_seed": per_seed, "missing": missing}
    if len(per_seed) < len(SEEDS):
        out["available"] = False
        out["note"] = "readings need M2 and M3 at all of seeds 1-3"
        return out
    if any(per_seed[s]["dDup"] is None for s in SEEDS):
        out["available"] = False
        out["missing"].append(f"{shown(stats_dir)}/w2c1_sample_stats.json "
                              "(duplicates per 1k: scripts/w1_sample_stats.py --set w2c1)")
        out["note"] = "readings need the duplicates per 1k of every arm"
        return out
    ps = [per_seed[s] for s in SEEDS]
    s_p = pooled_seed_sd([p["G_m2"] for p in ps], [p["G_m3"] for p in ps])
    mean_d_ts = float(np.mean([p["D_ts33m"] for p in ps]))
    mean_dnll = float(np.mean([p["dNLL"] for p in ps]))
    dam = read_damage([p["D_gpt2"] for p in ps], [p["D_ts33m"] for p in ps], [p["dDup"] for p in ps], s_p, mean_d_ts,
                      float(np.mean([p["dMAUVE"] for p in ps])))
    diss = read_dissociation(mean_dnll)
    out.update({"available": True, "pooled_seed_sd": s_p, "mean_D_gpt2": dam["mean_D"], "mean_D_ts33m": mean_d_ts,
                "mean_dNLL": mean_dnll, "mean_dMAUVE": float(np.mean([p["dMAUVE"] for p in ps])),
                "damage": dam, "dissociation": diss,
                "criterion_c_candidate_claim": dam["reading"] == "replicates" and diss == "holds"})
    return out


def analyze_m30(root, stats_dir):
    m3 = load_arm(root, "w2m_L_ts_m3", "L_ts_m3_uniform_wrong_s0")
    c0 = load_arm(root, "w2p_L_ts_m3_c0", "L_ts_m3_uniform_wrong_c0_s0")
    a0 = load_arm(root, "w1_L_ts_ctrl", "L_ts_released16")
    a1 = load_arm(root, "w1_L_ts_ctrl", "L_ts_ctrl_s0")
    if m3 is None or c0 is None:
        return {"available": False, "missing": [f"{shown(root)}/w2p_L_ts_m3_c0/L_ts_m3_uniform_wrong_c0_s0__result.json"
                                                if c0 is None else f"{shown(root)}/w2m_L_ts_m3/L_ts_m3_uniform_wrong_s0__result.json"]}
    i, i_se = paired_diff(m3["gpt2_groups"], c0["gpt2_groups"])
    its, its_se = paired_diff(m3["ts33m_groups"], c0["ts33m_groups"])
    d3, d0 = load_dup(stats_dir, "w2p", "M3"), load_dup(stats_dir, "w2p", "M3-0")
    return {"available": True, "I_gpt2": i, "I_gpt2_se_paired": i_se, "reading": read_m30(i),
            "I_ts33m": its, "I_ts33m_se_paired": its_se, "G_m3": m3["gpt2"], "G_m3_0": c0["gpt2"],
            "mauve": {"m3": m3["mauve"], "m3_0": c0["mauve"]}, "entropy": {"m3": m3["entropy"], "m3_0": c0["entropy"]},
            "val_nll": {"m3": m3["val_nll"], "m3_0": c0["val_nll"]},
            "dup_per_1k": {"m3": d3, "m3_0": d0},
            "references_descriptive": {"A0_gpt2": None if a0 is None else a0["gpt2"], "W1_A1_gpt2": None if a1 is None else a1["gpt2"]}}


def _coverage(arm, override):
    if override is not None:
        return override
    r = arm["result"]
    for v in (r.get("l1r_coverage"), r.get("coverage"), (r.get("l1r") or {}).get("coverage")):
        if isinstance(v, (int, float)):
            return float(v)
    return None


def analyze_l1r(root, stats_dir, coverage_override=None):
    m2 = load_arm(root, "w2m_L_ts_m2", "L_ts_m2_frozen_pw_s0")
    m3 = load_arm(root, "w2m_L_ts_m3", "L_ts_m3_uniform_wrong_s0")
    m2r = load_arm(root, "l1r_L_ts_m2r", "L_ts_m2r_frozen_pw_l1reassign_s0")
    if m2 is None or m3 is None or m2r is None:
        return {"available": False,
                "missing": [f"{shown(root)}/{nb}/{run}__result.json" for nb, run, arm in
                            (("w2m_L_ts_m2", "L_ts_m2_frozen_pw_s0", m2), ("w2m_L_ts_m3", "L_ts_m3_uniform_wrong_s0", m3),
                             ("l1r_L_ts_m2r", "L_ts_m2r_frozen_pw_l1reassign_s0", m2r)) if arm is None]}
    r_g = rescue(m2["gpt2"], m2r["gpt2"], m3["gpt2"])
    r_t = rescue(m2["ts33m"], m2r["ts33m"], m3["ts33m"])
    cov = _coverage(m2r, coverage_override)
    d2, dr = load_dup(stats_dir, "l1r", "M2"), load_dup(stats_dir, "l1r", "M2R")
    read = read_l1r(r_g, r_t, m2r["entropy"], m3["entropy"], m2r["mauve"], m2["mauve"], dr, d2, cov)
    return {"available": True, "rescue_gpt2": r_g, "rescue_ts33m": r_t, "coverage": cov, **read,
            "G": {"m2": m2["gpt2"], "m3": m3["gpt2"], "m2r": m2r["gpt2"]},
            "G_ts33m": {"m2": m2["ts33m"], "m3": m3["ts33m"], "m2r": m2r["ts33m"]},
            "entropy": {"m2": m2["entropy"], "m3": m3["entropy"], "m2r": m2r["entropy"]},
            "mauve": {"m2": m2["mauve"], "m2r": m2r["mauve"]}, "dup_per_1k": {"m2": d2, "m2r": dr},
            "val_nll": {"m2": m2["val_nll"], "m3": m3["val_nll"], "m2r": m2r["val_nll"]},
            "note": None if cov is not None else "coverage not recorded in the result (a finished run passed the run-time "
                                                 "gate); pass --l1r_coverage to record it"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "colab"))
    ap.add_argument("--stats_dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "w2c_analysis.json"))
    ap.add_argument("--l1r_coverage", type=float, default=None)
    a = ap.parse_args()
    out = {"w2c_stage1": analyze_w2c(a.root, a.stats_dir), "m3_0": analyze_m30(a.root, a.stats_dir),
           "l1r": analyze_l1r(a.root, a.stats_dir, a.l1r_coverage)}
    for name, r in out.items():
        if not r["available"]:
            print(f"{name}: not available (missing: {', '.join(r.get('missing', []))})")
    w = out["w2c_stage1"]
    if w["available"]:
        print(f"W2C: D_s {[round(w['per_seed'][s]['D_gpt2'], 2) for s in SEEDS]}  mean D {w['mean_D_gpt2']:.2f}  s_p "
              f"{w['pooled_seed_sd']:.2f}  mean dNLL {w['mean_dNLL']:.2f}  -> damage {w['damage']['reading']}, "
              f"dissociation {w['dissociation']}")
    m = out["m3_0"]
    if m["available"]:
        print(f"M3-0: I = {m['I_gpt2']:.2f} +- {m['I_gpt2_se_paired']:.2f} -> {m['reading']}")
    l = out["l1r"]
    if l["available"]:
        print(f"L1R: rescue GPT-2 large {l['rescue_gpt2']}, TinyStories-33M {l['rescue_ts33m']} -> {l['reading']}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", shown(a.out))


if __name__ == "__main__":
    main()
