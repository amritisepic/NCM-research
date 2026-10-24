"""Wave-1 registered analysis (W1 + W1a + W1b, seed 0): recompute every screen quantity from the saved results.

  python scripts/analyze_w1.py [--root results/colab] [--out results/w1_analysis_s0.json]

Reads each arm's `*__result.json` (16 per-group Gen-PPL values, MAUVE per quantization seed, entropy, clean val NLL)
and the step-matched A1 curve points from A1's `metrics.jsonl`. Nothing here is fitted or tuned; the thresholds are
the registered ones (`docs/preregistration.md`, W1b). The screen's entropy / MAUVE conditions are evaluated under
both readings of W1b's text: against the comparator being beaten (R1) and against A1 as in W1 (R2).
Exploratory extras (paired differences over the 16 shared sample seeds) are labelled as such.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

ARMS = {   # arm -> (run directory, result prefix, steps)
    "A0": ("w1_L_ts_ctrl", "L_ts_released16", 0),
    "A1": ("w1_L_ts_ctrl", "L_ts_ctrl_s0", 1800),
    "A2": ("w1_L_ts_requant", "L_ts_requant_s0", 1800),
    "A3": ("w1_L_ts_onpol", "L_ts_onpol_s0", 1350),
    "A4": ("w1_L_ts_b1", "L_ts_b1_s0", 1350),
    "A5": ("w1_L_ts_onpol2", "L_ts_onpol2_s0", 1080),
    "A6": ("w1_L_ts_soft", "L_ts_soft_s0", 1800),
}
COMPARATORS = {"A2": ["A1"], "A3": ["A1"], "A4": ["A1", "A3"], "A5": ["A1", "A3"], "A6": ["A1"]}   # W1b
BAR, ENT_TOL, MAUVE_TOL = 0.35, 0.05, 0.01      # W1b screen
CONFOUND = 0.5                                  # W1b step-count confound: A1 at the matched step this much better
SANITY_NLL, SANITY_PPL, RECIPE_LIMIT = 870.0, 16.1, 1.0   # W1 sanity checks (R008, R010; A1 - A0 limit)
KEY = "genppl_gpt2-large"


def load_arm(root, arm):
    d, prefix, steps = ARMS[arm]
    r = json.load(open(os.path.join(root, d, f"{prefix}__result.json")))
    fe = r["final_eval"]
    g = np.array(fe[f"{KEY}_per_group"], dtype=float)
    m = np.array(fe["mauve_per_seed"], dtype=float)
    assert r["steps"] == steps, f"{arm}: {r['steps']} steps, registered {steps}"
    assert abs(g.mean() - fe[KEY]) < 1e-9, f"{arm}: reported Gen-PPL is not the group mean"
    return {"run": r["run"], "steps": r["steps"], "gpu": r.get("gpu"), "eval_precision": r.get("eval_precision"),
            "n_groups": len(g), "n_samples": fe["n"], "genppl": float(g.mean()), "genppl_sd": float(g.std(ddof=1)),
            "genppl_se": float(g.std(ddof=1) / math.sqrt(len(g))), "groups": g.tolist(),
            "genppl_ts33m": fe.get("genppl_TinyStories-33M"), "mauve": float(m.mean()), "mauve_sd": float(m.std(ddof=1)),
            "mauve_n": fe.get("mauve_n"), "entropy": fe["entropy"], "val_nll": fe["val_nll"],
            "val_nll_per_level": fe["val_nll_per_level"], "rep_2": fe["rep_2"], "distinct_2": fe["distinct_2"],
            "dup_adj1_per_1k": fe["dup_adj1_per_1k"], "dup_adj2_per_1k": fe["dup_adj2_per_1k"]}


def curve(root, arm):
    """Sample-curve points (256 samples, one fixed seed) of an arm's final process, keyed by step."""
    d, prefix, _ = ARMS[arm]
    pts = {}
    for line in open(os.path.join(root, d, f"{prefix}__metrics.jsonl")):
        rec = json.loads(line)
        if KEY in rec.get("eval", {}):
            pts[rec["step"]] = rec["eval"][KEY]    # a resumed process re-logs a step: keep the last value
    return pts


def compare(a, c, ref):
    """Screen of arm a against comparator c; entropy / MAUVE conditions against `ref` (c for R1, A1 for R2)."""
    d = c["genppl"] - a["genppl"]                  # > 0: the arm is better
    ent, mv = a["entropy"] - ref["entropy"], a["mauve"] - ref["mauve"]
    ok = {"genppl": d >= BAR, "entropy": abs(ent) <= ENT_TOL, "mauve": mv >= -MAUVE_TOL}
    return {"genppl_gain": d, "se_unpaired": math.hypot(a["genppl_se"], c["genppl_se"]), "entropy_gap": ent,
            "mauve_gap": mv, "conditions": ok, "passes": all(ok.values())}


def paired(a, c):
    """Exploratory: the 16 groups share sample seeds (42, 0-14), so group-wise differences are paired."""
    diff = np.array(c["groups"]) - np.array(a["groups"])
    return {"mean": float(diff.mean()), "se_paired": float(diff.std(ddof=1) / math.sqrt(len(diff))),
            "corr_groups": float(np.corrcoef(a["groups"], c["groups"])[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/colab")
    ap.add_argument("--out", default="results/w1_analysis_s0.json")
    a = ap.parse_args()
    arms = {k: load_arm(a.root, k) for k in ARMS}
    A0, A1 = arms["A0"], arms["A1"]
    out = {"arms": arms, "screen": {}, "exploratory_paired": {}}

    print(f"{'arm':4} {'steps':>5} {'Gen-PPL':>8} {'SD':>5} {'SE':>5} {'MAUVE':>6} {'(sd)':>6} {'val NLL':>8} "
          f"{'H':>6} {'TS-33M':>7}")
    for k, r in arms.items():
        print(f"{k:4} {r['steps']:5d} {r['genppl']:8.2f} {r['genppl_sd']:5.2f} {r['genppl_se']:5.2f} {r['mauve']:6.3f} "
              f"{r['mauve_sd']:6.3f} {r['val_nll']:8.1f} {r['entropy']:6.3f} {r['genppl_ts33m']:7.2f}")

    print("\nScreen (W1b): gain = comparator - arm Gen-PPL; pass needs gain >= 0.35, |entropy gap| <= 0.05, "
          "MAUVE gap >= -0.01")
    for k, comps in COMPARATORS.items():
        for c in comps:
            r1, r2 = compare(arms[k], arms[c], arms[c]), compare(arms[k], arms[c], A1)
            out["screen"][f"{k}_vs_{c}"] = {"R1_vs_comparator": r1, "R2_vs_A1": r2}
            out["exploratory_paired"][f"{k}_vs_{c}"] = paired(arms[k], arms[c])
            print(f"  {k} vs {c}: gain {r1['genppl_gain']:+7.2f} (unpaired SE {r1['se_unpaired']:.2f}) | "
                  f"R1: dH {r1['entropy_gap']:+.3f} dMAUVE {r1['mauve_gap']:+.3f} -> {'PASS' if r1['passes'] else 'fail'}"
                  f" | R2: dH {r2['entropy_gap']:+.3f} dMAUVE {r2['mauve_gap']:+.3f} -> "
                  f"{'PASS' if r2['passes'] else 'fail'}")

    pts = curve(a.root, "A1")
    conf = {}
    for s in (1080, 1350):
        better = pts[1800] - pts[s]                # > 0: A1 at the matched step is better than at 1800
        conf[str(s)] = {"A1_curve_at_step": pts[s], "A1_curve_at_1800": pts[1800], "improvement_vs_1800": better,
                        "fires": better >= CONFOUND}
    out["step_confound"] = conf
    print("\nStep-count confound (A1 curve, 256 samples, seed 10000): " +
          "; ".join(f"A1@{s} {v['A1_curve_at_step']:.3f} vs A1@1800 {v['A1_curve_at_1800']:.3f} "
                    f"({'fires' if v['fires'] else 'does not fire'})" for s, v in conf.items()))

    san = {"A0_val_nll": A0["val_nll"], "A0_val_nll_ref": SANITY_NLL, "A0_genppl": A0["genppl"],
           "A0_genppl_ref": SANITY_PPL, "A1_minus_A0": A1["genppl"] - A0["genppl"], "recipe_limit": RECIPE_LIMIT,
           "recipe_damages_model": A1["genppl"] - A0["genppl"] > RECIPE_LIMIT}
    out["sanity"] = san
    print(f"\nSanity: A0 val NLL {A0['val_nll']:.1f} (ref ~{SANITY_NLL:.0f}), A0 Gen-PPL {A0['genppl']:.2f} "
          f"(ref ~{SANITY_PPL}); A1 - A0 = {san['A1_minus_A0']:+.2f} (limit {RECIPE_LIMIT})")

    adv = {rd: sorted({k for k, comps in COMPARATORS.items() for c in comps
                       if out["screen"][f"{k}_vs_{c}"][rd]["passes"]}) for rd in ("R1_vs_comparator", "R2_vs_A1")}
    out["advances"] = adv
    out["conditional_controls"] = {"A1m": "A3" in adv["R1_vs_comparator"] or "A3" in adv["R2_vs_A1"],
                                   "A6s": "A6" in adv["R1_vs_comparator"] or "A6" in adv["R2_vs_A1"]}
    print(f"Advance to seeds 1-2: R1 {adv['R1_vs_comparator'] or 'none'}; R2 {adv['R2_vs_A1'] or 'none'}; "
          f"A1 always. Conditional controls triggered: {out['conditional_controls']}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
