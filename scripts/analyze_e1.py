"""E1 + E1a (docs/preregistration.md): the registered readings of how much of NCM's code NLL is representation choice,
computed from the saved `results/colab/e1_equivalence/e1_{ts,wmt,sdk}__result.json` (nothing is re-sampled).

Per item x and cut k (registered cuts: k >= 8 on every item; k = 5..7 on the first 8 items, descriptive):
  N_k = sum(level_nll[k:]),  A_k = sum(level_nll[:k]),  rho_k = P(sampled finer codes decode to x^) = hits / samples,
  T_k = -log rho_k,  R_k = N_k - T_k,  U_k = A_k + T_k,  and rho_k >= exp(-N_k) (the canonical codes decode to x^).
E1a's primary uncertainty: the anytime-valid beta-mixture confidence sequence for rho (Bernoulli, prior Beta(1/2, 1/2);
valid at E1's data-dependent stop by Ville's inequality), intersected with [exp(-N_k), 1], at
delta = 0.05 / (items x registered cuts) per model. Everything is computed in log p (N_k reaches hundreds of nats).

  python scripts/analyze_e1.py [--in_dir results/colab/e1_equivalence] [--out results/e1_analysis.json]

Needs `src` on PYTHONPATH (it imports scripts/e1_equivalence.py for the registered point summaries, as the run did).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("e1_equivalence", os.path.join(ROOT, "scripts", "e1_equivalence.py"))
E1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E1)

FILES = {"tinystories": "e1_ts__result.json", "wmt14": "e1_wmt__result.json", "sudoku": "e1_sdk__result.json"}
REG_CUT_MIN = 8                      # registered cuts: k >= 8 on every item
ALPHA = 0.05                         # familywise level, split over (items x registered cuts) per model
PRIOR = (0.5, 0.5)
UNTOUCHED_FROM = 3                   # TinyStories: val windows 0-2 were seen by the R046 pilot
RELEASED = {"tinystories": "A0", "wmt14": "J0", "sudoku": "S0"}
BASE = {"tinystories": "A1", "wmt14": "J0"}
REGISTERED_P2 = [("A1", "A2"), ("A1", "A4"), ("J0", "J1")]            # (base a, model b): dT/dN of b - a
DESCRIPTIVE_P2 = [("A1", "A0"), ("J0", "J2F")]
QUALITY_ORDER = {"tinystories": ["A0", "A2", "A1", "A4"],             # Gen-PPL, lower is better
                 "wmt14": ["J0", "J2F", "J1"]}                        # val BLEU, higher is better
QUALITY_VALUE = {"tinystories": {"A0": 16.16, "A2": 16.29, "A1": 16.60, "A4": 36.8, "metric": "Gen-PPL (lower better)"},
                 "wmt14": {"J0": 23.55, "J2F": 22.90, "J1": 22.82, "metric": "val BLEU (higher better)"}}
NOT_READ_N = 5.0                     # mean N_8 below this many nats: reported, not read
REL_TOL = 1e-9


# ------------------------------------------------------------------------------------------------ confidence sequence
def log1mexp(x):
    """log(1 - exp(x)) for x <= 0 (-inf at x = 0)."""
    if x == 0.0:
        return -math.inf
    return math.log(-math.expm1(x)) if x > -math.log(2.0) else math.log1p(-math.exp(x))


def _betaln(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_mixture(S, n, log_p, a=PRIOR[0], b=PRIOR[1]):
    """log M(p) = betaln(a+S, b+n-S) - betaln(a, b) - S log p - (n-S) log(1-p), the beta-mixture martingale."""
    return _log_m(S, n, log_p, log1mexp(log_p) if log_p < 0 else -math.inf, a, b)


def _log_m(S, n, u, v, a, b):
    out = _betaln(a + S, b + n - S) - _betaln(a, b)
    if S > 0:
        out -= S * u
    if n - S > 0:
        out -= (n - S) * v
    return out


@functools.lru_cache(maxsize=None)
def cs_log_interval(S, n, delta, a=PRIOR[0], b=PRIOR[1]):
    """(log p_L, log p_U) of {p : log M(p) < log(1/delta)}: an interval containing S/n. S = 0 gives log p_L = -inf;
    S = n gives log p_U = 0. Roots by bisection in log p (lower end) and log(1-p) (upper end)."""
    assert 0 < delta < 1 and 0 <= S <= n
    if n == 0:
        return -math.inf, 0.0
    c = math.log(1.0 / delta)
    base = _betaln(a + S, b + n - S) - _betaln(a, b) - c
    if S > 0:
        u0 = math.log(S / n)
        v0 = log1mexp(u0)
    else:
        u0, v0 = -math.inf, 0.0

    def bisect(g, hi):                      # g(hi) < 0, g -> +inf downwards; returns the root
        step = 1.0
        lo = hi - step
        while g(lo) <= 0:
            step *= 2.0
            lo = hi - step
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    if S == 0:
        lo_u = -math.inf
    else:
        lo_u = bisect(lambda u: base - S * u - ((n - S) * log1mexp(u) if n > S else 0.0), u0)
    if S == n:
        hi_u = 0.0
    else:
        v = bisect(lambda v: base - ((S * log1mexp(v)) if S > 0 else 0.0) - (n - S) * v, v0)
        hi_u = log1mexp(v)
    return lo_u, hi_u


def intersect_floor(log_lo, log_hi, N):
    """Intersect [exp(log_lo), exp(log_hi)] with [exp(-N), 1] in log space. Empty (log_hi < -N): the point exp(-N)."""
    floor = -float(N)
    if log_hi < floor:
        return floor, floor, True
    return max(log_lo, floor), min(log_hi, 0.0), False


def delta_for(n_items, n_registered_cuts):
    return ALPHA / (n_items * n_registered_cuts)


# ------------------------------------------------------------------------------------------------ per-item intervals
def _N(r, k):
    return float(sum(r["level_nll"][k:]))


def _A(r, k):
    return float(sum(r["level_nll"][:k]))


def build_intervals(rows, cuts, delta):
    """{cut: dict of arrays over the items valid at that cut (present, samples > 0)}: idx, S, n, N, A, lp_lo, lp_hi
    (after the exp(-N) intersection). `empty` lists the items whose CS lay entirely below exp(-N)."""
    out = {}
    for k in cuts:
        idx = [i for i, r in enumerate(rows) if str(k) in r["mc"] and r["mc"][str(k)][1] > 0]
        if not idx:
            continue
        S = np.array([rows[i]["mc"][str(k)][0] for i in idx])
        n = np.array([rows[i]["mc"][str(k)][1] for i in idx])
        N = np.array([_N(rows[i], k) for i in idx])
        A = np.array([_A(rows[i], k) for i in idx])
        lo, hi, emp = np.empty(len(idx)), np.empty(len(idx)), np.zeros(len(idx), bool)
        for j in range(len(idx)):
            l, h = cs_log_interval(int(S[j]), int(n[j]), delta)
            lo[j], hi[j], emp[j] = intersect_floor(l, h, N[j])
        out[k] = {"idx": np.array(idx), "S": S, "n": n, "N": N, "A": A, "lp_lo": lo, "lp_hi": hi, "empty": emp}
    return out


def _se(v):
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def summarise_set(iv, mask, point):
    """E1a summary at one cut over the items selected by `mask` (a boolean over iv's arrays)."""
    S, N, A, lo, hi = (iv[x][mask] for x in ("S", "N", "A", "lp_lo", "lp_hi"))
    m = len(S)
    if m == 0:
        return {"items": 0}
    Nm = float(N.mean())
    R_lo, R_hi = float((N + lo).mean()), float((N + hi).mean())
    return {"items": m, "censored": int((S == 0).sum()), "hit_share": float((S > 0).mean()),
            "censored_share": float((S == 0).mean()), "N_mean": Nm, "N_se": _se(N), "A_mean": float(A.mean()),
            "T_lo_mean": float((-hi).mean()), "T_hi_mean": float((-lo).mean()),
            "R_lo_mean": R_lo, "R_hi_mean": R_hi,
            "U_lo_mean": float((A - hi).mean()), "U_hi_mean": float((A - lo).mean()),
            "R_share_lo": R_lo / Nm if Nm > 0 else None, "R_share_hi": R_hi / Nm if Nm > 0 else None,
            "T_width_le2_share": float(((hi - lo) <= 2.0).mean()), "empty_intersections": int(iv["empty"][mask].sum()),
            "point": point}


def item_sets(domain):
    return ["all", "untouched"] if domain == "tinystories" else ["all"]


def _mask(iv, set_name):
    return np.ones(len(iv["idx"]), bool) if set_name == "all" else iv["idx"] >= UNTOUCHED_FROM


def model_summaries(domain, rows, intervals, cuts, K):
    out = {}
    for k, iv in intervals.items():
        out[str(k)] = {}
        for s in item_sets(domain):
            mask = _mask(iv, s)
            sel = [int(i) for i in iv["idx"][mask]]
            point = E1.summarise([rows[i] for i in sel], [k], K).get(str(k)) if sel else None
            d = summarise_set(iv, mask, point)
            d["registered_cut"] = k >= REG_CUT_MIN
            out[str(k)][s] = d
    return out


# ------------------------------------------------------------------------------------------------ integrity
def _diff(a, b, path, bad):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                bad.append(f"{path}/{key}: key present in only one side")
            else:
                _diff(a[key], b[key], f"{path}/{key}", bad)
    elif a is None or b is None:
        if a is not b:
            bad.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        if a != b:
            bad.append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-12):
            bad.append(f"{path}: {a!r} vs {b!r}")
    else:
        bad.append(f"{path}: unexpected types {type(a).__name__} / {type(b).__name__}")


def check_saved(data):
    """Recompute the run's `summary` and `paired_vs_*` blocks from the saved rows with the run script's functions;
    return the list of mismatches against the saved blocks."""
    bad = []
    cuts = data["cuts"]
    for m, rows in data["rows"].items():
        K = len(rows[0]["level_nll"])
        _diff(E1.summarise(rows, cuts, K), data["summary"].get(m, {}), f"summary/{m}", bad)
    for key in [k for k in data if k.startswith("paired_vs_")]:
        base = key[len("paired_vs_"):]
        saved = data[key]
        want = {m: {str(k): E1.paired(data["rows"][base], data["rows"][m], k) for k in cuts if k >= REG_CUT_MIN}
                for m in data["rows"] if m != base}
        _diff(want, saved, key, bad)
    if not any(k.startswith("paired_vs_") for k in data) and len(data["rows"]) > 1:
        bad.append("several models but no paired_vs_* block")
    return bad


def monotonicity(model, intervals):
    """Per item over consecutive cuts present for it: R_k must not increase with k, U_k must not decrease.
    Violations beyond the intervals: R_lo(k+1) > R_hi(k) or U_hi(k+1) < U_lo(k). Point violations (both cuts with
    hits) are counted descriptively."""
    beyond, pts = [], {"pairs": 0, "R_rise": 0, "U_fall": 0}
    pos = {k: {int(i): j for j, i in enumerate(iv["idx"])} for k, iv in intervals.items()}
    for k in sorted(intervals):
        if k + 1 not in intervals:
            continue
        a, b = intervals[k], intervals[k + 1]
        for i, ja in pos[k].items():
            jb = pos[k + 1].get(i)
            if jb is None:
                continue
            R_hi_a, R_lo_b = a["N"][ja] + a["lp_hi"][ja], b["N"][jb] + b["lp_lo"][jb]
            U_lo_a, U_hi_b = a["A"][ja] - a["lp_hi"][ja], b["A"][jb] - b["lp_lo"][jb]
            if R_lo_b > R_hi_a:
                beyond.append([model, i, k, "R"])
            if U_hi_b < U_lo_a:
                beyond.append([model, i, k, "U"])
            if a["S"][ja] > 0 and b["S"][jb] > 0:
                Ta, Tb = -math.log(a["S"][ja] / a["n"][ja]), -math.log(b["S"][jb] / b["n"][jb])
                pts["pairs"] += 1
                pts["R_rise"] += int((b["N"][jb] - Tb) > (a["N"][ja] - Ta) + 1e-9)
                pts["U_fall"] += int((b["A"][jb] + Tb) < (a["A"][ja] + Ta) - 1e-9)
    return beyond, pts


# ------------------------------------------------------------------------------------------------ predictions
def p1_e1_rule(share_lo, share_hi):
    """E1 on bounds: holds if the lower bound of mean R / mean N >= 0.8; overturned if the upper bound < 0.5."""
    if share_lo is not None and share_lo >= 0.8:
        return "holds"
    if share_hi is not None and share_hi < 0.5:
        return "overturned"
    return "neither"


def p1_e1a_rule(share_lo, share_hi, censored_share, width_share):
    """E1a added rule: inconclusive if > 20% of items are censored; else holds if the lower bound > 0.90 and >= 80% of
    items have a T width <= 2 nats; overturned if the upper bound < 0.80; else neither."""
    if censored_share > 0.20:
        return "inconclusive"
    if share_lo is not None and share_lo > 0.90 and width_share >= 0.80:
        return "holds"
    if share_hi is not None and share_hi < 0.80:
        return "overturned"
    return "neither"


def prediction1(domain, model, summaries):
    cell = summaries.get(model, {}).get(str(REG_CUT_MIN))
    if not cell:
        return None
    out = {"model": model, "cut": REG_CUT_MIN, "sets": {}}
    for s, d in cell.items():
        if d["items"] == 0:
            continue
        reg_a = (s == "untouched") if domain == "tinystories" else True
        if d["N_mean"] < NOT_READ_N:
            e1 = e1a = "not_read"
        else:
            e1 = p1_e1_rule(d["R_share_lo"], d["R_share_hi"])
            e1a = p1_e1a_rule(d["R_share_lo"], d["R_share_hi"], d["censored_share"], d["T_width_le2_share"])
        out["sets"][s] = {"items": d["items"], "N_mean": d["N_mean"], "censored": d["censored"],
                          "censored_share": d["censored_share"], "R_share_lo": d["R_share_lo"],
                          "R_share_hi": d["R_share_hi"], "T_width_le2_share": d["T_width_le2_share"],
                          "e1_rule": {"verdict": e1, "registered": True},
                          "e1a_rule": {"verdict": e1a, "registered": reg_a}}
    return out


def p2_compare(rows_a, rows_b, iv_a, iv_b, k=REG_CUT_MIN):
    """Model b minus base a at cut k over items valid in both: dN, the dT interval (endpoints propagated) and the
    holds / carries_half / neither reading."""
    ia = {int(i): j for j, i in enumerate(iv_a[k]["idx"])}
    ib = {int(i): j for j, i in enumerate(iv_b[k]["idx"])}
    common = sorted(set(ia) & set(ib))
    ja, jb = np.array([ia[i] for i in common]), np.array([ib[i] for i in common])
    dN = iv_b[k]["N"][jb] - iv_a[k]["N"][ja]
    Tlo_a, Thi_a = -iv_a[k]["lp_hi"][ja], -iv_a[k]["lp_lo"][ja]
    Tlo_b, Thi_b = -iv_b[k]["lp_hi"][jb], -iv_b[k]["lp_lo"][jb]
    dT_lo, dT_hi = float((Tlo_b - Thi_a).mean()), float((Thi_b - Tlo_a).mean())
    pt = E1.paired(rows_a, rows_b, k)
    out = {"items": len(common), "dN_mean": float(dN.mean()), "dN_se": _se(dN), "dT_lo": dT_lo, "dT_hi": dT_hi,
           "dT_point_mean": pt.get("dT_mean"), "dT_point_se": pt.get("dT_se"),
           "dT_point_share_of_dN": pt.get("dT_share_of_dN")}
    out["reading"] = p2_reading(out["dN_mean"], dT_lo, dT_hi)
    return out


def p2_reading(dN, dT_lo, dT_hi):
    """holds: the whole dT interval within +/- 0.2 |dN|. carries_half: the whole interval on dN's side at >= 0.5 |dN|."""
    lim = 0.2 * abs(dN)
    if dT_lo >= -lim and dT_hi <= lim:
        return "holds"
    if (dN < 0 and dT_hi <= 0.5 * dN) or (dN > 0 and dT_lo >= 0.5 * dN):
        return "carries_half"
    return "neither"


def p2_overall(readings):
    """readings: {"A2-A1": ..., "A4-A1": ..., "J1-J0": ...} (the three registered comparisons). Holds if all three
    hold; overturned if >= 2 of 3 carry half; else neither (missing comparisons never count towards holds)."""
    vals = list(readings.values())
    if len(vals) == 3 and all(v == "holds" for v in vals):
        return "holds"
    if sum(v == "carries_half" for v in vals) >= 2:
        return "overturned"
    return "neither"


def prediction3(domain, summaries):
    models = [m for m in summaries if summaries[m].get(str(REG_CUT_MIN), {}).get("all", {}).get("items")]
    rows = {}
    for m in models:
        d = summaries[m][str(REG_CUT_MIN)]["all"]
        pt = d["point"] or {}
        if d["censored"] == 0 and "U_mean" in pt:
            U, basis = pt["U_mean"], "point (every item hits)"
        else:
            U, basis = 0.5 * (d["U_lo_mean"] + d["U_hi_mean"]), "interval midpoint"
        rows[m] = {"U": U, "U_basis": basis, "U_lo": d["U_lo_mean"], "U_hi": d["U_hi_mean"], "N": d["N_mean"],
                   "T_lo": d["T_lo_mean"], "T_hi": d["T_hi_mean"], "T_mid": 0.5 * (d["T_lo_mean"] + d["T_hi_mean"])}
    order = lambda key: sorted(rows, key=lambda m: rows[m][key])
    out = {"models": rows, "order_by_U": order("U"), "order_by_N": order("N"), "order_by_T_lo": order("T_lo"),
           "order_by_T_hi": order("T_hi"), "order_by_T_mid": order("T_mid")}
    q = QUALITY_ORDER.get(domain)
    if q:
        qo = [m for m in q if m in rows]
        out.update(quality_metric=QUALITY_VALUE[domain]["metric"], quality_order=qo,
                   U_order_equals_quality=out["order_by_U"] == qo, N_order_equals_quality=out["order_by_N"] == qo)
    return out


# ------------------------------------------------------------------------------------------------ driver
def analyse_file(data):
    domain = data["domain"]
    cuts = data["cuts"]
    reg_cuts = [k for k in cuts if k >= REG_CUT_MIN]
    res = {"domain": domain, "cuts": cuts, "registered_cuts": reg_cuts, "models": list(data["rows"]), "per_model": {}}
    intervals_by_model, summaries = {}, {}
    for m, rows in data["rows"].items():
        K = len(rows[0]["level_nll"])
        delta = delta_for(len(rows), len(reg_cuts))
        iv = build_intervals(rows, cuts, delta)
        intervals_by_model[m] = iv
        summaries[m] = model_summaries(domain, rows, iv, cuts, K)
        beyond, pts = monotonicity(m, iv)
        res["per_model"][m] = {
            "n_items": len(rows), "delta": delta, "recon_exact_false": sum(not r["recon_exact"] for r in rows),
            "empty_intersection": [[m, int(i), k] for k, v in iv.items() for i in v["idx"][v["empty"]]],
            "monotonicity_beyond_intervals": beyond, "monotonicity_point": pts,
            "intervals": {str(k): {"idx": v["idx"].tolist(), "S": v["S"].tolist(), "n": v["n"].tolist(),
                                   "log_p_lo": np.round(v["lp_lo"], 10).tolist(),
                                   "log_p_hi": np.round(v["lp_hi"], 10).tolist()} for k, v in iv.items()},
            "summaries": summaries[m]}
    p1 = prediction1(domain, RELEASED[domain], summaries)
    if p1:
        res["prediction1"] = p1
    comps = {}
    base = BASE.get(domain)
    if base in data["rows"]:
        for a, b in REGISTERED_P2 + DESCRIPTIVE_P2:
            if a == base and b in data["rows"] and REG_CUT_MIN in intervals_by_model[a] and REG_CUT_MIN in intervals_by_model[b]:
                c = p2_compare(data["rows"][a], data["rows"][b], intervals_by_model[a], intervals_by_model[b])
                c["registered"] = (a, b) in REGISTERED_P2
                comps[f"{b}-{a}"] = c
    res["prediction2_comparisons"] = comps
    res["prediction3"] = prediction3(domain, summaries)
    return res


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def analyse(in_dir):
    out = {"protocol": {"alpha": ALPHA, "prior": list(PRIOR), "registered_cut_min": REG_CUT_MIN,
                        "untouched_from_item": UNTOUCHED_FROM, "not_read_if_mean_N8_below": NOT_READ_N,
                        "delta": "ALPHA / (items in the model's rows x registered cuts in the file), per model"},
           "inputs": {}, "domains": {}, "integrity": {}}
    for dom, fn in FILES.items():
        path = os.path.join(in_dir, fn)
        if not os.path.exists(path):
            continue
        out["inputs"][fn] = sha256(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        bad = check_saved(data)
        out["integrity"][fn] = {"saved_summary_and_paired_match_recomputation": not bad, "mismatches": bad}
        assert not bad, f"{fn}: saved summary / paired blocks differ from the recomputation:\n  " + "\n  ".join(bad[:20])
        out["domains"][dom] = analyse_file(data)
    regs = {}
    for dom in out["domains"].values():
        for name, c in dom["prediction2_comparisons"].items():
            if c["registered"]:
                regs[name] = c["reading"]
    out["prediction2_overall"] = {"registered_readings": regs, "verdict": p2_overall(regs)}
    return out


# ------------------------------------------------------------------------------------------------ output
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _dump(o, ind=0):
    """JSON with dicts one key per line and lists of scalars on one line (the per-item intervals stay compact)."""
    pad = " " * ind
    if isinstance(o, dict):
        if not o:
            return "{}"
        body = ",\n".join(f'{pad} {json.dumps(k)}: {_dump(v, ind + 1)}' for k, v in o.items())
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(o, list) and any(isinstance(v, (dict, list)) for v in o):
        return "[\n" + ",\n".join(pad + " " + _dump(v, ind + 1) for v in o) + "\n" + pad + "]"
    return json.dumps(o, allow_nan=False)


def print_table(res):
    print("E1 / E1a cut-8 readings (registered rule marked *)")
    for dom, d in res["domains"].items():
        p1 = d.get("prediction1")
        if p1:
            for s, v in p1["sets"].items():
                print(f"  P1 {dom:11s} {p1['model']} [{s:9s}] items={v['items']:3d} cens={v['censored']:3d} "
                      f"N8={v['N_mean']:7.2f} Rshare=[{_f(v['R_share_lo'])},{_f(v['R_share_hi'])}] "
                      f"Twidth<=2:{v['T_width_le2_share']:.2f}  E1{'*' if v['e1_rule']['registered'] else ''}="
                      f"{v['e1_rule']['verdict']}  E1a{'*' if v['e1a_rule']['registered'] else ''}={v['e1a_rule']['verdict']}")
        for name, c in d["prediction2_comparisons"].items():
            print(f"  P2 {name:7s}{'*' if c['registered'] else ' '} dN={c['dN_mean']:8.2f}+-{c['dN_se']:.2f} "
                  f"dT in [{c['dT_lo']:.2f},{c['dT_hi']:.2f}] point={_f(c['dT_point_mean'])}  -> {c['reading']}")
        p3 = d["prediction3"]
        if "quality_order" in p3:
            print(f"  P3 {dom}: U order {p3['order_by_U']} N order {p3['order_by_N']} quality {p3['quality_order']} "
                  f"(U==quality {p3['U_order_equals_quality']}, N==quality {p3['N_order_equals_quality']})")
        for m, pm in d["per_model"].items():
            mp = pm["monotonicity_point"]
            print(f"  {dom} {m}: delta={pm['delta']:.2e} recon_exact_false={pm['recon_exact_false']} "
                  f"empty_intersections={len(pm['empty_intersection'])} monotone_beyond_intervals="
                  f"{len(pm['monotonicity_beyond_intervals'])} point R_rise/U_fall={mp['R_rise']}/{mp['U_fall']} of {mp['pairs']}")
    p2 = res["prediction2_overall"]
    print(f"  P2 overall: {p2['verdict']} {p2['registered_readings']}")
    print("  integrity:", {k: v["saved_summary_and_paired_match_recomputation"] for k, v in res["integrity"].items()})


def _f(x):
    return "None" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=os.path.join("results", "colab", "e1_equivalence"))
    ap.add_argument("--out", default=os.path.join("results", "e1_analysis.json"))
    a = ap.parse_args()
    res = _clean(analyse(a.in_dir))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(_dump(res) + "\n")
    print_table(res)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
