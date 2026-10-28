"""Audit 11: independent, standard-library recomputation on hash-pinned results.

The interval inversion uses log odds, independently of analyze_e1's two log-space
bisections. No production analysis or sampling function is imported. This file
also runs directly with standard-library Python, without pytest or model assets.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


ROOT = Path(__file__).resolve().parents[1]
BASE = "results/colab/e1_equivalence/"
CONT = "results/colab/e1b_continue/e1b__result.json"
PINS = {
    BASE + "e1_ts__result.json": "5b134622a64121d9e835a8bae483c505768efbc63e68688a7e7ac5781942c5db",
    BASE + "e1_wmt__result.json": "c66d29ed71fb17458413fb1ef683c54cb16413ca3b2d1bb8a131f14d2f6e478e",
    BASE + "e1_sdk__result.json": "e936925fcb6935230f253ecc4220e12b33428dda3234db03ec8f1190b3df64e5",
    CONT: "358ff27d7ccafa81bbe4d83467b8fcb9904d313aee9c2d1cc2d840c5d6792100",
    "results/e1_analysis.json": "7fb5c7b3b8a58cc87f2f9d762d3f6d235b2eee0de00cbbe0101b4db04416ac38",
    "results/e1b_analysis.json": "1debc7e80e7a8775d7cc669706b3f89515674eb6f11d6796147c0e44ae7e769c",
}
DOMAINS = (("tinystories", "ts", "A0", "A1", 64, 4),
           ("wmt14", "wmt", "J0", "J0", 128, 3),
           ("sudoku", "sdk", "S0", None, 128, 4))


@lru_cache(None)
def read_pinned(path):
    raw = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == PINS[path], path
    return json.loads(raw)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif a is None or isinstance(a, (bool, str)):
        assert a == b, (a, b)
    else:
        close(a, b)


def se(values):
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


@lru_cache(None)
def independent_cs(hits, samples, delta):
    """Invert the mixture in t=log(p/(1-p)); brackets exceed every saved N."""
    if samples == 0:
        return -math.inf, 0.0
    s, n = hits, samples
    c = (math.lgamma(s + 0.5) + math.lgamma(n - s + 0.5)
         - math.lgamma(n + 1) - 2 * math.lgamma(0.5) + math.log(delta))

    def equation(t):
        return c + s * softplus(-t) + (n - s) * softplus(t)

    center = math.log(s / (n - s)) if 0 < s < n else 0.0

    def root(low, high, left):
        assert (equation(low) > 0) == left
        assert (equation(high) > 0) != left
        for _ in range(160):
            mid = (low + high) / 2
            if (equation(mid) > 0) == left:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    lo = -softplus(-root(-1e6, 1e6 if s == n else center, True)) if s else -math.inf
    hi = -softplus(-root(-1e6 if s == 0 else center, 1e6, False)) if s < n else 0.0
    return lo, hi


def item_interval(row, cut, delta):
    s, n = row["mc"][str(cut)]
    N, A = sum(row["level_nll"][cut:]), sum(row["level_nll"][:cut])
    lo, hi = independent_cs(s, n, delta)
    empty = hi < -N
    lo, hi = (-N, -N) if empty else (max(lo, -N), hi)
    return dict(S=s, n=n, N=N, A=A, lo=lo, hi=hi, empty=empty)


def summaries(rows, cut, delta, first=0):
    selected = [r for r in rows[first:] if str(cut) in r["mc"] and r["mc"][str(cut)][1] > 0]
    q = [item_interval(r, cut, delta) for r in selected]
    N, A = [v["N"] for v in q], [v["A"] for v in q]
    nmean = mean(N)
    rlo, rhi = [v["N"] + v["lo"] for v in q], [v["N"] + v["hi"] for v in q]
    censored = sum(v["S"] == 0 for v in q)
    return dict(items=len(q), censored=censored, hit_share=1-censored/len(q), censored_share=censored/len(q),
                N_mean=nmean, N_se=se(N), A_mean=mean(A), T_lo_mean=mean(-v["hi"] for v in q),
                T_hi_mean=mean(-v["lo"] for v in q), R_lo_mean=mean(rlo), R_hi_mean=mean(rhi),
                U_lo_mean=mean(v["A"]-v["hi"] for v in q), U_hi_mean=mean(v["A"]-v["lo"] for v in q),
                R_share_lo=mean(rlo)/nmean, R_share_hi=mean(rhi)/nmean,
                T_width_le2_share=mean(v["hi"]-v["lo"] <= 2 for v in q),
                empty_intersections=sum(v["empty"] for v in q))


def point_summary(rows, cut):
    rows = [r for r in rows if str(cut) in r["mc"] and r["mc"][str(cut)][1] > 0]
    N, A = [sum(r["level_nll"][cut:]) for r in rows], [sum(r["level_nll"][:cut]) for r in rows]
    sn = [r["mc"][str(cut)] for r in rows]
    T = [-math.log(s/n) if s else math.inf for s, n in sn]
    hit_share = mean(s > 0 for s, _ in sn)
    rlo = [N[i]-T[i] if s else 0.0 for i, (s, _) in enumerate(sn)]
    rhi = [N[i]-(T[i] if s else min(math.log(n/3), N[i])) for i, (s, n) in enumerate(sn)]
    out = dict(items=len(rows), hit_share=hit_share, samples_mean=mean(n for _, n in sn), N_mean=mean(N),
               N_se=se(N), A_mean=mean(A), A_se=se(A), T_median=median(T) if hit_share > .5 else None,
               R_lo_mean=mean(rlo), R_hi_mean=mean(rhi), R_lo_share_of_N=mean(rlo)/mean(N),
               R_hi_share_of_N=mean(rhi)/mean(N))
    if hit_share == 1:
        R, U = [n-t for n, t in zip(N, T)], [a+t for a, t in zip(A, T)]
        out.update(T_mean=mean(T), T_se=se(T), R_mean=mean(R), R_se=se(R), U_mean=mean(U),
                   U_se=se(U), R_share_of_N=mean(R)/mean(N))
    return out


def verdicts(d):
    lo, hi, cens, narrow = (d[k] for k in ("R_share_lo", "R_share_hi", "censored_share", "T_width_le2_share"))
    if d["N_mean"] < 5:
        return "not_read", "not_read"
    e1 = "holds" if lo >= .8 else "overturned" if hi < .5 else "neither"
    e1a = ("inconclusive" if cens > .2 else "holds" if lo > .9 and narrow >= .8
           else "overturned" if hi < .8 else "neither")
    return e1, e1a


def test_independent_all_saved_e1_intervals_summaries_and_point_blocks():
    saved = read_pinned("results/e1_analysis.json")
    for domain, short, _, base, n, cuts in DOMAINS:
        data = read_pinned(BASE + "e1_" + short + "__result.json")
        delta = .05/(n*cuts)
        for model, rows in data["rows"].items():
            recorded = saved["domains"][domain]["per_model"][model]
            close(delta, recorded["delta"])
            for k in data["cuts"]:
                same(point_summary(rows, k), data["summary"][model][str(k)])
                for name, first in (("all", 0), ("untouched", 3)) if short == "ts" else (("all", 0),):
                    actual = summaries(rows, k, delta, first)
                    expected = recorded["summaries"][str(k)][name]
                    same(actual, {key: expected[key] for key in actual})
                iv = recorded["intervals"][str(k)]
                for i, lo, hi in zip(iv["idx"], iv["log_p_lo"], iv["log_p_hi"]):
                    q = item_interval(rows[i], k, delta)
                    close(lo, q["lo"])
                    close(hi, q["hi"])
            if base is not None and model != base:
                for k, expected in data["paired_vs_"+base][model].items():
                    a, b = data["rows"][base], rows
                    dn = [sum(y["level_nll"][int(k):])-sum(x["level_nll"][int(k):]) for x, y in zip(a, b)]
                    actual = dict(dN_mean=mean(dn), dN_se=se(dn))
                    if all(x["mc"][k][0] and y["mc"][k][0] for x, y in zip(a, b)):
                        dt = [-math.log(y["mc"][k][0]/y["mc"][k][1])+math.log(x["mc"][k][0]/x["mc"][k][1]) for x, y in zip(a, b)]
                        actual.update(dT_mean=mean(dt), dT_se=se(dt), dT_share_of_dN=abs(mean(dt))/abs(mean(dn)))
                    same(actual, expected)


def test_independent_e1b_replacement_readings_and_selection():
    cont, saved = read_pinned(CONT), read_pinned("results/e1b_analysis.json")
    assert len(cont["items"]) == 72 and not cont["not_reached"] and cont["complete"]
    assert all(r["replay_match"] and r["replay_hits"] == r["e1"][0] for r in cont["items"])
    for domain, short, model, _, n, cuts in DOMAINS:
        data = read_pinned(BASE + "e1_" + short + "__result.json")
        delta = .05/(n*cuts)
        source, rows = data["rows"][model], deepcopy(data["rows"][model])
        censored, wide = [], []
        for i, row in enumerate(rows):
            q = item_interval(row, 8, delta)
            if not q["S"]:
                censored.append(i)
            elif q["hi"]-q["lo"] > 2:
                wide.append(i)
        mine = [r for r in cont["items"] if r["model"] == model]
        assert sorted(r["item"] for r in mine) == sorted(censored+wide)
        for r in mine:
            assert source[r["item"]]["mc"]["8"] == r["e1"]
            rows[r["item"]]["mc"]["8"] = r["final"]
        for name, first in (("all", 0), ("untouched", 3)) if short == "ts" else (("all", 0),):
            for phase, selected in (("e1", source), ("e1b", rows)):
                actual = summaries(selected, 8, delta, first)
                expected = saved["models"][model][phase]["sets"][name]
                for key in ("items", "censored", "N_mean", "R_share_lo", "R_share_hi", "censored_share", "T_width_le2_share"):
                    close(actual[key], expected[key])
                assert verdicts(actual) == tuple(expected[k]["verdict"] for k in ("e1_rule", "e1a_rule"))
        assert len(mine) == dict(A0=14, J0=58, S0=0)[model]
        assert sum(r["stop"] == "width" for r in mine) == dict(A0=7, J0=39, S0=0)[model]
        assert sum(r["stop"] == "cap" for r in mine) == dict(A0=7, J0=19, S0=0)[model]


def test_independent_exploratory_numbers():
    ts = read_pinned(BASE+"e1_ts__result.json")
    sdk = read_pinned(BASE+"e1_sdk__result.json")
    wmt = read_pinned(BASE+"e1_wmt__result.json")
    for rows in ts["rows"].values():
        assert all(row["mc"]["5"] == [0, 2048] for row in rows[:8])
        assert all(summaries(rows, k, .05/256)["R_share_lo"] >= .9 for k in range(9, 12))
    close(min(summaries(sdk["rows"]["S0"], k, .05/512)["R_share_lo"] for k in range(5, 12)), .9377845145757284)
    for rows, expected in ((ts["rows"]["A0"], .12088775831630817), (wmt["rows"]["J0"], .21272650692565653)):
        censored = [r for r in rows if not r["mc"]["8"][0]]
        close(sum(sum(r["level_nll"][8:]) for r in censored)/sum(sum(r["level_nll"][8:]) for r in rows), expected)
    lo, hi = independent_cs(20, 448, .05/256)
    close(math.exp(lo), .012060669487439545)
    close(math.exp(hi), .1084107910918084)
    close(hi-lo, 2.1959779300553763)
    close(-independent_cs(0, 2048, .05/256)[1], 5.0685596644659885)


def test_independent_p2_p3_and_integrity_numbers():
    saved = read_pinned("results/e1_analysis.json")
    for domain, short, _, base, n, cuts in DOMAINS:
        data = read_pinned(BASE+"e1_"+short+"__result.json")
        recorded = saved["domains"][domain]
        delta = .05/(n*cuts)
        U, N = {}, {}
        for model, rows in data["rows"].items():
            d = summaries(rows, 8, delta)
            N[model] = d["N_mean"]
            U[model] = (point_summary(rows, 8)["U_mean"] if not d["censored"]
                        else (d["U_lo_mean"]+d["U_hi_mean"])/2)
            p3 = recorded["prediction3"]["models"][model]
            for actual, key in ((U[model], "U"), (N[model], "N"), (d["T_lo_mean"], "T_lo"),
                                (d["T_hi_mean"], "T_hi"), ((d["T_lo_mean"]+d["T_hi_mean"])/2, "T_mid")):
                close(actual, p3[key])
            pairs = rises = falls = beyond_r = beyond_u = empty = 0
            for row in rows:
                for k in data["cuts"]:
                    if str(k) not in row["mc"] or not row["mc"][str(k)][1]:
                        continue
                    a = item_interval(row, k, delta)
                    empty += a["empty"]
                    if str(k+1) not in row["mc"] or not row["mc"][str(k+1)][1]:
                        continue
                    b = item_interval(row, k+1, delta)
                    beyond_r += b["N"]+b["lo"] > a["N"]+a["hi"]
                    beyond_u += b["A"]-b["lo"] < a["A"]-a["hi"]
                    if a["S"] and b["S"]:
                        pairs += 1
                        ta, tb = -math.log(a["S"]/a["n"]), -math.log(b["S"]/b["n"])
                        rises += b["N"]-tb > a["N"]-ta+1e-9
                        falls += b["A"]+tb < a["A"]+ta-1e-9
            expected = recorded["per_model"][model]
            assert expected["monotonicity_point"] == dict(pairs=pairs, R_rise=rises, U_fall=falls)
            assert not (empty or beyond_r or beyond_u or rises or falls)
            assert sum(not r["recon_exact"] for r in rows) == expected["recon_exact_false"]
            if model == base or base is None:
                continue
            a, b = [item_interval(r, 8, delta) for r in data["rows"][base]], [item_interval(r, 8, delta) for r in rows]
            dn = [y["N"]-x["N"] for x, y in zip(a, b)]
            lo, hi = mean(x["lo"]-y["hi"] for x, y in zip(a, b)), mean(x["hi"]-y["lo"] for x, y in zip(a, b))
            expected = recorded["prediction2_comparisons"][model+"-"+base]
            for actual, key in ((mean(dn), "dN_mean"), (se(dn), "dN_se"), (lo, "dT_lo"), (hi, "dT_hi")):
                close(actual, expected[key])
            assert expected["reading"] == "neither"
        assert sorted(U, key=U.get) == recorded["prediction3"]["order_by_U"]
        assert sorted(N, key=N.get) == recorded["prediction3"]["order_by_N"]


def test_current_analysis_against_independent_roots_for_every_saved_count():
    """Run under pytest/conftest; the standalone arithmetic above needs no torch."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("audit11_current_analysis", ROOT/"scripts/analyze_e1.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    cases = set()
    delta_by_model = {}
    for _, short, _, _, n, cuts in DOMAINS:
        data = read_pinned(BASE+"e1_"+short+"__result.json")
        delta = .05/(n*cuts)
        for model, rows in data["rows"].items():
            delta_by_model[model] = delta
            for row in rows:
                cases.update((s, count, delta) for s, count in row["mc"].values())
    for row in read_pinned(CONT)["items"]:
        cases.add((*row["final"], delta_by_model[row["model"]]))
    for delta in (.05/256, .05/384, .05/512):
        cases.update(((0, 0, delta), (0, 32768, delta), (32768, 32768, delta)))
    for s, n, delta in cases:
        actual, expected = analysis.cs_log_interval(s, n, delta), independent_cs(s, n, delta)
        for x, y in zip(actual, expected):
            assert x == y or abs(x-y) <= 1e-8, (s, n, delta, x, y)
        for lp in (*actual, -1.0, -1000.0):
            # Skip undefined 0*infinity terms explicitly at the boundaries.
            log_m = (math.lgamma(s+.5)+math.lgamma(n-s+.5)-math.lgamma(n+1)
                     - 2*math.lgamma(.5))
            if s:
                log_m -= s*lp
            if n-s:
                log_m -= (n-s)*math.log(-math.expm1(lp))
            current = analysis.log_mixture(s, n, lp)
            assert current == log_m or abs(current-log_m) <= 1e-8, (s, n, delta, lp, current, log_m)


if __name__ == "__main__":
    test_independent_all_saved_e1_intervals_summaries_and_point_blocks()
    test_independent_e1b_replacement_readings_and_selection()
    test_independent_exploratory_numbers()
    test_independent_p2_p3_and_integrity_numbers()
    for domain, short, model, _, n, cuts in DOMAINS:
        rows = read_pinned(BASE+"e1_"+short+"__result.json")["rows"][model]
        print(domain, json.dumps(summaries(rows, 8, .05/(n*cuts)), sort_keys=True))
