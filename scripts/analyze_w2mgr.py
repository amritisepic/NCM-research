"""W2M-GR (docs/preregistration.md, "W2M-GR"): the registered readings of W2M-G's localization at W2C's seeds 1-3, computed
from the saved per-seed results of `scripts/w2mg_eval.py --arms m2,m3` (nothing is re-sampled).

  python scripts/analyze_w2mgr.py [--root results/colab] [--out results/w2mgr_analysis.json]

Inputs: <root>/w2mgr_eval/w2mgr_eval_s{1,2,3}__result.json (the notebook w2mgr_eval copies each run directory's result.json as
`<basename>__result.json`). Seed 0 (W2M-G's exploratory run, <root>/w2mg_eval/w2mg_eval__result.json) is reported beside, never
pooled. Every result is checked before it is read: complete, arms m2 and m3 only, the right run seed, the registered protocol
(counts, seeds, precision, an L4), exactly the 8 registered cells, no (iv), the stored shares equal to the shares recomputed from the
cells' per-group log losses, and the sanity block equal to its recomputation. A missing or rejected file is reported ("not
available", with the files / reasons) and no reading is made; nothing crashes.

Per seed (GPT-2 large log loss primary, TinyStories-33M beside): pi_c, sigma_c and the interaction at c = 2, 4, 6 (jackknife SEs as
stored), W2M-G's (ii) reading, (i)'s ratios and rule, (iii)'s paired CI and reading, the sanity check against W2C's final eval.
Across seeds: the mean and the seed SD (ddof 1) of every share. Readings (seeds 1-3 only): localization replicates / overturned /
mixed; (i) replicates; (iii) replicates. The reading functions are pure; the thresholds are the module-level constants below
(tests/test_w2mgr.py checks them at their boundaries and against scripts/w2mg_eval.py's protocol).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(ROOT, "results", "colab")
DEFAULT_OUT = os.path.join(ROOT, "results", "w2mgr_analysis.json")
SEEDS = (1, 2, 3)
CUTS = (2, 4, 6)
SCORERS = ("gpt2-large", "TinyStories-33M")                  # primary, beside
BANKS = ("released", "m2", "m3")                              # (i)'s generated banks
NB, NB_SEED0 = "w2mgr_eval", "w2mg_eval"

# W2M-GR localization (registered). Comparisons are exact except for EPS, which absorbs float noise in a seed mean that lands on a
# threshold (e.g. the mean of three 0.1s); 1e-12 is far below any resolvable difference between shares.
EPS = 1e-12
PI2_MAX = 0.10               # (a) pi_2 <= 0.10: the L0-L1 prefix carries little
PI6_MIN = 0.30               # (b) pi_6 >= 0.30: M2's prefix through L5 carries a substantial share
SIGMA6_MAX = 0.20            # (c) sigma_6 <= 0.20: M2's levels 6-11 alone carry little
INTERACTION_MIN = 0.20       # (d) interaction >= 0.20 at both cut 4 and cut 6
OVERTURN_PI2 = 0.25          # overturned: pi_2 >= 0.25 (L1 carries the damage) on the GPT-2 large seed means,
OVERTURN_SIGMA6 = 0.50       #   or sigma_6 >= 0.50 (the late levels carry it),
OVERTURN_INTERACTION = 0.05  #   or the interaction < 0.05 at both cuts 4 and 6 (additive, no amplification)
# W2M-G's own rules, re-stated (tests check them against ncmr.w2mg)
I_RATIO_MIN, I_LOWER_MIN = 3.0, 1.2          # (i) holds: every r_b >= 3 and every lower CI bound > 1.2
EARLY_PI4, EARLY_PI6, COND_SIGMA6 = 0.5, 0.75, 0.8
SANITY_SE = 3.0

# the protocol every seed's result must carry (G.Protocol()'s registered defaults; tests compare them)
REGISTERED = {"n_clean": 512, "group_n": 256, "bank_seeds": [900, 901], "replay_seeds": list(range(8)), "cuts": [2, 4, 6],
              "js_levels": [2, 3, 4, 5], "n_boot": 2000, "boot_seed": 0, "root_seed": 2027, "n_roots": 8,
              "precision": "fp32, TF32 off, eval mode",
              "scorers": {"gpt2-large": "gpt2-large", "TinyStories-33M": "roneneldan/TinyStories-33M"}}
GPU_REQUIRED = "L4"
CELL_KEYS = ("m2_m2", "m3_m3", *[f"{p}_c{c}" for c in CUTS for p in ("m2to3", "m3to2")])
CELL_GROUPS, CELL_N = 8, 2048
RUN_DIRS = {"m2": "L_ts_m2_frozen_pw", "m3": "L_ts_m3_uniform_wrong"}
W2C_NB = {"m2": "w2c1_L_ts_m2", "m3": "w2c1_L_ts_m3"}


# ------------------------------------------------------------------------------------------------ pure functions
def _ge(x, t):
    return bool(x >= t - EPS)


def _le(x, t):
    return bool(x <= t + EPS)


def _lt(x, t):
    return bool(x < t - EPS)


def compact(shares):
    """The five numbers the localization reading uses from one scorer's `ii.shares` ({"2": {...}, "4": {...}, "6": {...}})
    of one seed: pi_2, pi_4, pi_6, sigma_6, interaction at cuts 4 and 6."""
    return {"pi2": shares["2"]["pi"], "pi4": shares["4"]["pi"], "pi6": shares["6"]["pi"], "sigma6": shares["6"]["sigma"],
            "i4": shares["4"]["interaction"], "i6": shares["6"]["interaction"]}


def conditions(rec):
    """Localization conditions (a)-(d) on one share record (a seed, or a seed mean)."""
    c = {"a_pi2_le_0.10": _le(rec["pi2"], PI2_MAX), "b_pi6_ge_0.30": _ge(rec["pi6"], PI6_MIN),
         "c_sigma6_le_0.20": _le(rec["sigma6"], SIGMA6_MAX),
         "d_interaction_ge_0.20_at_cuts_4_and_6": _ge(rec["i4"], INTERACTION_MIN) and _ge(rec["i6"], INTERACTION_MIN)}
    c["all"] = all(c.values())
    return c


def overturn_conditions(rec):
    """The overturn conditions on a seed-mean record (GPT-2 large)."""
    c = {"pi2_ge_0.25_L1_carries_it": _ge(rec["pi2"], OVERTURN_PI2),
         "sigma6_ge_0.50_late_levels_carry_it": _ge(rec["sigma6"], OVERTURN_SIGMA6),
         "interaction_lt_0.05_at_cuts_4_and_6_additive": _lt(rec["i4"], OVERTURN_INTERACTION) and _lt(rec["i6"], OVERTURN_INTERACTION)}
    c["any"] = any(c.values())
    return c


def mean_record(recs):
    return {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}


def read_localization(gpt2_recs, ts_recs):
    """W2M-GR's localization reading from the per-seed share records (lists over seeds) under GPT-2 large and TinyStories-33M.
    replicates: (a)-(d) in every seed under GPT-2 large AND on the seed means under TinyStories-33M. overturned: on the GPT-2
    large seed means pi_2 >= 0.25, or sigma_6 >= 0.50, or the interaction < 0.05 at both cuts 4 and 6. Anything else: mixed.
    (The two cannot both hold; overturned is tested first so a claim is never made against an overturn condition.)"""
    per_seed = [conditions(r) for r in gpt2_recs]
    gm, tm = mean_record(gpt2_recs), mean_record(ts_recs)
    ts_cond, over = conditions(tm), overturn_conditions(gm)
    rep = all(c["all"] for c in per_seed) and ts_cond["all"]
    reading = "overturned" if over["any"] else "replicates" if rep else "mixed"
    return {"reading": reading, "gpt2_per_seed_conditions": per_seed, "gpt2_every_seed_a_to_d": all(c["all"] for c in per_seed),
            "ts33m_seed_mean": tm, "ts33m_seed_mean_conditions": ts_cond, "gpt2_seed_mean": gm, "overturn_conditions": over}


def read_i(ratios):
    """W2M-G's (i) rule for one seed: ratios {bank: {"r", "lo", "hi"}}: holds if every r >= 3 and every lower bound > 1.2;
    overturned if every upper bound <= 1.2; else mixed."""
    v = list(ratios.values())
    if all(x["r"] >= I_RATIO_MIN and x["lo"] > I_LOWER_MIN for x in v):
        return "holds"
    if all(x["hi"] <= I_LOWER_MIN for x in v):
        return "overturned"
    return "mixed"


def read_iii(lo, hi):
    """(iii) for one seed: the paired 95% CI of rho(M3) - rho(M2) above 0 holds, below 0 overturned, else neither."""
    return "holds" if lo > 0 else "overturned" if hi < 0 else "neither"


def read_ii(pi4, pi6, sigma6):
    """W2M-G's (ii) reading of one seed and scorer: early-state selection (pi_4 >= 0.5 and pi_6 >= 0.75), conditional rendering
    (sigma_6 >= 0.8), both (only with a negative interaction), else distributed."""
    early, cond = (pi4 >= EARLY_PI4 and pi6 >= EARLY_PI6), sigma6 >= COND_SIGMA6
    if early and cond:
        return "both"
    return "early-state selection" if early else "conditional rendering" if cond else "distributed"


def sanity(g22, se22, g33, se33, ref):
    """G(M2_s, M2_s) and G(M3_s, M3_s) (GPT-2 large) within 3 SE of the reference {"m2", "m3"}."""
    return {"G_m2m2": g22, "z_m2m2": (g22 - ref["m2"]) / se22, "G_m3m3": g33, "z_m3m3": (g33 - ref["m3"]) / se33, "ref": dict(ref),
            "within_3se": bool(abs(g22 - ref["m2"]) <= SANITY_SE * se22 and abs(g33 - ref["m3"]) <= SANITY_SE * se33)}


def replicates_all(flags):
    return bool(len(flags) > 0 and all(flags))


def spread(values):
    """Mean and sample SD (ddof 1) across seeds, with the values."""
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None, "per_seed": [float(v) for v in a]}


def point_shares(l22, l33, l23, l32):
    """pi, sigma, interaction of one cut from the cells' per-group log losses (point estimates, as w2mg.shares)."""
    m22, m33, m23, m32 = (float(np.mean(v)) for v in (l22, l33, l23, l32))
    delta = m22 - m33
    if delta == 0:
        return (math.nan, math.nan, math.nan)
    pi, sg = (m23 - m33) / delta, (m32 - m33) / delta
    return (pi, sg, 1.0 - pi - sg)


# ------------------------------------------------------------------------------------------------ loading and checking
def shown(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/") if os.path.abspath(path).startswith(ROOT) else os.path.basename(path)


def seed_path(root, seed):
    return os.path.join(root, NB, f"{NB}_s{seed}__result.json")


def seed0_path(root):
    return os.path.join(root, NB_SEED0, f"{NB_SEED0}__result.json")


def load_json(path):
    """(object, None) or (None, reason)."""
    if not os.path.exists(path):
        return None, "missing"
    try:
        with open(path) as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return None, f"unreadable ({type(e).__name__})"


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_result(r, seed):
    """The reasons a seed's result.json cannot be read as W2M-GR's run of that seed (empty: acceptable)."""
    bad = []
    if not isinstance(r, dict):
        return ["not a JSON object"]
    pr = r.get("protocol") if isinstance(r.get("protocol"), dict) else {}
    if r.get("complete") is not True:
        bad.append("not complete (complete != true)")
    if pr.get("arms") != ["m2", "m3"]:
        bad.append(f"arms {pr.get('arms')} != ['m2', 'm3']")
    if pr.get("run_seed") != seed:
        bad.append(f"run_seed {pr.get('run_seed')} != {seed}")
    for k, v in REGISTERED.items():
        if pr.get(k) != v:
            bad.append(f"protocol {k} = {pr.get(k)!r}, registered {v!r}")
    if pr.get("corrupted_bank") is not None:
        bad.append("a corrupted bank was built")
    runs = pr.get("runs") if isinstance(pr.get("runs"), dict) else {}
    want = {"released": "released", **{a: f"{d}_s{seed}/final.pt" for a, d in RUN_DIRS.items()}}
    if runs != want:
        bad.append(f"runs {runs} != {want}")
    if sorted(pr.get("ckpt_sha256") or {}) != ["m2", "m3"]:
        bad.append("ckpt_sha256 does not hold exactly m2 and m3")
    gpu = (pr.get("identity") or {}).get("gpu")
    if not (isinstance(gpu, str) and GPU_REQUIRED in gpu.replace("-", " ").split()):
        bad.append(f"gpu {gpu!r} is not an {GPU_REQUIRED}")
    cells = r.get("cells") if isinstance(r.get("cells"), dict) else {}
    if set(cells) != set(CELL_KEYS):
        bad.append(f"cells {sorted(cells)} != the 8 registered cells")
    for k in CELL_KEYS:
        c = cells.get(k)
        if isinstance(c, dict) and (c.get("n_groups") != CELL_GROUPS or c.get("n") != CELL_N):
            bad.append(f"cell {k}: n_groups {c.get('n_groups')}, n {c.get('n')} != {CELL_GROUPS}, {CELL_N}")
    if "iv" in r:
        bad.append("an (iv) section is present")
    for sec in ("ii", "i", "iii"):
        if not isinstance(r.get(sec), dict):
            bad.append(f"no ({sec}) section")
    if bad:
        return bad
    # sections are there: their contents
    try:
        sh = r["ii"]["shares"]
        for sc in SCORERS:
            for c in CUTS:
                for q in ("pi", "sigma", "interaction", "pi_se", "sigma_se", "interaction_se"):
                    if not _finite(sh[sc][str(c)][q]):
                        bad.append(f"ii.shares[{sc}][{c}].{q} is not a finite number")
        if not bad:
            for sc in SCORERS:
                ll = {k: cells[k]["genppl_" + sc]["log_loss_per_group"] for k in CELL_KEYS}
                for c in CUTS:
                    got = point_shares(ll["m2_m2"], ll["m3_m3"], ll[f"m2to3_c{c}"], ll[f"m3to2_c{c}"])
                    stored = tuple(sh[sc][str(c)][q] for q in ("pi", "sigma", "interaction"))
                    if not np.allclose(got, stored, atol=1e-9, rtol=0):
                        bad.append(f"ii.shares[{sc}][{c}] {stored} != the shares of the cells' log losses {got}")
        for b in BANKS:
            if not all(_finite(r["i"]["ratios"][b][q]) for q in ("r", "lo", "hi")):
                bad.append(f"i.ratios[{b}] is not finite")
        if not all(_finite(r["iii"]["m3_minus_m2"][q]) for q in ("mean", "lo", "hi")):
            bad.append("iii.m3_minus_m2 is not finite")
        sblk = r["ii"].get("sanity_vs_w2c_final_eval")
        if not isinstance(sblk, dict):
            bad.append("no sanity_vs_w2c_final_eval block (the reference must be W2C's final eval)")
        else:
            want_src = {a: f"results/colab/{W2C_NB[a]}/{RUN_DIRS[a]}_s{seed}__result.json" for a in RUN_DIRS}
            if sblk.get("source") != want_src:
                bad.append(f"sanity source {sblk.get('source')} != {want_src}")
            g = lambda k: cells[k]["genppl_gpt2-large"]
            again = sanity(g("m2_m2")["mean"], g("m2_m2")["se"], g("m3_m3")["mean"], g("m3_m3")["se"], sblk["ref"])
            if again["within_3se"] != sblk.get("within_3se") or not np.isclose(
                    [again["G_m2m2"], again["G_m3m3"]], [sblk.get("G_m2m2"), sblk.get("G_m3m3")], rtol=1e-9, atol=0).all():
                bad.append("the sanity block does not equal its recomputation from the cells")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        bad.append(f"malformed result ({type(e).__name__}: {e})")
    return bad


def summarise_seed(r):
    """The per-seed quantities and readings of one accepted result (or of seed 0's, descriptively)."""
    sh = r["ii"]["shares"]
    ratios = {b: {k: r["i"]["ratios"][b][k] for k in ("r", "lo", "hi")} for b in BANKS}
    d = r["iii"]["m3_minus_m2"]
    sblk = r["ii"].get("sanity_vs_w2c_final_eval") or r["ii"].get("sanity_vs_w2m")
    out = {"shares": sh,
           "ii_reading": {sc: read_ii(sh[sc]["4"]["pi"], sh[sc]["6"]["pi"], sh[sc]["6"]["sigma"]) for sc in SCORERS},
           "conditions_gpt2": conditions(compact(sh["gpt2-large"])),
           "i": {"ratios": ratios, "rule": read_i(ratios), "stored_reading": r["i"].get("reading")},
           "iii": {"m3_minus_m2": {k: d[k] for k in ("mean", "se", "lo", "hi") if k in d}, "reading": read_iii(d["lo"], d["hi"]),
                   "stored_reading": r["iii"].get("reading")},
           "sanity": sblk}
    return out


# ------------------------------------------------------------------------------------------------ the analysis
def analyze_w2mgr(root):
    per_seed, missing, rejected, results = {}, [], {}, {}
    for s in SEEDS:
        path = seed_path(root, s)
        r, why = load_json(path)
        if why == "missing":
            missing.append(shown(path))
            continue
        bad = [f"{shown(path)}: {why}"] if why else check_result(r, s)
        if bad:
            rejected[str(s)] = bad
            continue
        results[s] = r
        per_seed[str(s)] = {"file": shown(path), **summarise_seed(r)}
    out = {"seeds_present": sorted(int(k) for k in per_seed), "per_seed": per_seed, "missing": missing, "rejected": rejected}
    if len(per_seed) < len(SEEDS):
        out["available"] = False
        out["note"] = "readings need an accepted result for every one of seeds 1-3"
        return out
    ps = [per_seed[str(s)] for s in SEEDS]
    recs = {sc: [compact(p["shares"][sc]) for p in ps] for sc in SCORERS}
    loc = read_localization(recs["gpt2-large"], recs["TinyStories-33M"])
    stats = {sc: {str(c): {q: spread([p["shares"][sc][str(c)][q] for p in ps]) for q in ("pi", "sigma", "interaction")}
                  for c in CUTS} for sc in SCORERS}
    i_flags = [p["i"]["rule"] == "holds" for p in ps]
    iii_flags = [p["iii"]["reading"] == "holds" for p in ps]
    out.update({"available": True, "seed_stats": stats, "localization": loc, "i_replicates": replicates_all(i_flags),
                "iii_replicates": replicates_all(iii_flags),
                "ii_reading_per_seed": {str(s): per_seed[str(s)]["ii_reading"] for s in SEEDS},
                "sanity_within_3se_per_seed": {str(s): bool(per_seed[str(s)]["sanity"]["within_3se"]) for s in SEEDS},
                "localization_replicates": loc["reading"] == "replicates",
                "note": "'localization replicates' with W2C's 'damage replicates' and 'dissociation holds' is the criterion-(c) "
                        "candidate; the user is told before any further compute is spent on it"})
    return out


def analyze_seed0(root):
    """Seed 0 (W2M-G, exploratory) beside, never pooled: its stored shares and readings, with (a)-(d) shown descriptively."""
    path = seed0_path(root)
    r, why = load_json(path)
    if why:
        return {"available": False, "missing": [shown(path)] if why == "missing" else [], "problems": [] if why == "missing" else [why]}
    try:
        out = {"available": True, "file": shown(path), "complete": r.get("complete"), "pooled": False,
               "note": "W2M-G's exploratory seed-0 run, reported beside seeds 1-3 and never pooled with them", **summarise_seed(r)}
    except (KeyError, TypeError, ValueError) as e:
        return {"available": False, "missing": [], "problems": [f"malformed ({type(e).__name__}: {e})"]}
    return out


# ------------------------------------------------------------------------------------------------ driver
def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    out = {"w2mgr": analyze_w2mgr(a.root), "seed0_beside": analyze_seed0(a.root)}
    w = out["w2mgr"]
    if not w["available"]:
        print(f"W2M-GR: not available (missing: {', '.join(w['missing']) or 'none'}; rejected: {w['rejected'] or 'none'})")
    else:
        for sc in SCORERS:
            for c in CUTS:
                st = w["seed_stats"][sc][str(c)]
                print(f"{sc} cut {c}: " + "  ".join(f"{q} {st[q]['mean']:.3f} (sd {fmt(st[q]['sd'])})" for q in st))
        print(f"W2M-GR: localization {w['localization']['reading']}; (i) replicates {w['i_replicates']}; "
              f"(iii) replicates {w['iii_replicates']}; (ii) per seed {w['ii_reading_per_seed']}; "
              f"sanity within 3 SE {w['sanity_within_3se_per_seed']}")
    if not out["seed0_beside"]["available"]:
        print(f"seed 0 (beside): not available (missing: {', '.join(out['seed0_beside'].get('missing', [])) or 'none'})")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", shown(a.out))


if __name__ == "__main__":
    main()
