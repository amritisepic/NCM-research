"""scripts/analyze_e1.py (E1 + E1a, docs/preregistration.md): confidence sequence, intersection, Bonferroni delta,
integrity, monotonicity and the prediction 1 / 2 / 3 readings, on synthetic inputs and on the saved E1 results."""
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_e1", ROOT / "scripts" / "analyze_e1.py")
AE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AE)
REAL = ROOT / "results" / "colab" / "e1_equivalence"


# ------------------------------------------------------------------------------------------------ AC1
def _logM(S, n, lp):
    return AE.log_mixture(S, n, lp)


@pytest.mark.parametrize("S,n", [(1, 64), (5, 64), (20, 448), (32, 64), (63, 64), (3, 2048), (19, 2048), (1, 1), (2, 3)])
@pytest.mark.parametrize("delta", [0.1, 1.95e-4, 1e-5])
def test_cs_endpoints_solve_the_mixture_equation(S, n, delta):
    lo, hi = AE.cs_log_interval(S, n, delta)
    c = math.log(1 / delta)
    phat = math.log(S / n)
    assert lo < phat < hi or (S == n and hi == 0.0)
    assert abs(_logM(S, n, lo) - c) <= 1e-9
    if S < n:
        assert abs(_logM(S, n, hi) - c) <= 1e-9
    else:
        assert hi == 0.0
    # just inside: in the set (log M < c); just outside: out
    probes = [(lo + 1e-6, True), (lo - 1e-6, False), (hi - 1e-6, True)]
    if S < n and hi + 1e-6 < 0:
        probes.append((hi + 1e-6, False))
    for lp, inside in probes:
        assert (_logM(S, n, lp) < c) == inside, (lp, inside)


def test_cs_closed_forms_for_all_misses_and_all_hits():
    a = b = 0.5
    bl = lambda x, y: math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)
    for n in (1, 7, 64, 2048):
        for delta in (0.1, 1e-4):
            lo, hi = AE.cs_log_interval(0, n, delta)
            assert lo == -math.inf
            pU = 1 - math.exp((math.log(delta) + bl(a, b + n) - bl(a, b)) / n)
            assert math.isclose(math.exp(hi), pU, rel_tol=1e-9)
            lo, hi = AE.cs_log_interval(n, n, delta)
            assert hi == 0.0
            pL = math.exp((math.log(delta) + bl(a + n, b) - bl(a, b)) / n)
            assert math.isclose(math.exp(lo), pL, rel_tol=1e-9)


# ------------------------------------------------------------------------------------------------ AC2
@pytest.mark.parametrize("rho", [0.005, 0.05, 0.5])
def test_cs_is_valid_at_e1s_data_dependent_stop(rho):
    """Rounds of 64, stop at >= 20 hits, cap 2048 samples; delta = 0.1: miscoverage at the stop <= delta + 3 SE, and
    (time-uniformity, Ville) the true rate is also inside the sequence at every round of the path."""
    delta, reps = 0.1, 1500
    rng = np.random.default_rng(12345 + int(rho * 1000))
    c = math.log(1 / delta)
    miss = ever = 0
    for _ in range(reps):
        S = n = 0
        crossed = False
        while n < 2048 and S < 20:
            S += int(rng.binomial(64, rho))
            n += 64
            crossed |= AE.log_mixture(S, n, math.log(rho)) >= c
        lo, hi = AE.cs_log_interval(S, n, delta)
        miss += not (lo <= math.log(rho) <= hi)
        ever += crossed
    tol = delta + 3 * math.sqrt(delta * (1 - delta) / reps)
    assert miss / reps <= tol, miss / reps
    assert ever / reps <= tol, ever / reps


@pytest.mark.parametrize("p", [0.02, 0.3, 0.9])
def test_mixture_is_a_martingale_at_the_true_rate(p):
    """E_p[M_n(p)] = 1 exactly: sum_S Binom(S; n, p) M(S, n; p) = 1 (independent of the root finding)."""
    for n in (1, 5, 40):
        tot = sum(math.comb(n, S) * p ** S * (1 - p) ** (n - S) * math.exp(AE.log_mixture(S, n, math.log(p)))
                  for S in range(n + 1))
        assert tot == pytest.approx(1.0, abs=1e-9), (n, tot)


# ------------------------------------------------------------------------------------------------ AC3
def test_intersection_in_log_space_without_underflow():
    assert math.exp(-1000.0) == 0.0                          # the naive route underflows
    lo, hi, emp = AE.intersect_floor(-math.inf, -2.0, 1000.0)
    assert (lo, hi, emp) == (-1000.0, -2.0, False)
    lo, hi, emp = AE.intersect_floor(-1500.0, -2.0, 1000.0)
    assert lo == -1000.0
    lo, hi, emp = AE.intersect_floor(-5.0, -2.0, 1000.0)
    assert (lo, hi) == (-5.0, -2.0)


def test_empty_intersection_uses_the_point_and_is_recorded():
    lo, hi = AE.cs_log_interval(0, 64, 0.01)                 # p_U ~ 0.1
    N = 0.01                                                 # exp(-N) ~ 0.99 > p_U
    assert hi < -N
    assert AE.intersect_floor(lo, hi, N) == (-N, -N, True)
    rows = [_row([0.0] * 8 + [0.01, 0, 0, 0], {"8": [0, 64]})]
    iv = AE.build_intervals(rows, [8], 0.01)[8]
    assert iv["empty"].tolist() == [True] and iv["lp_lo"][0] == iv["lp_hi"][0] == -0.01
    # the registered-style record lists it
    data = _data("tinystories", {"A0": rows})
    res = AE.analyse_file(data)
    assert res["per_model"]["A0"]["empty_intersection"] == [["A0", 0, 8]]


def test_intervals_have_R_nonnegative_and_T_at_most_N_at_huge_N():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        N = [0.0, 1.0, 50.0, 400.0, 1000.0][i % 5]
        n = int(rng.choice([64, 128, 448, 2048]))
        S = int(rng.integers(0, min(n, 30) + 1)) if i % 3 else n
        rows.append(_row([5.0] * 8 + [N / 4] * 4, {"8": [S, n]}))
    iv = AE.build_intervals(rows, [8], 0.001)[8]
    N, lo, hi = iv["N"], iv["lp_lo"], iv["lp_hi"]
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert (lo <= hi).all() and (hi <= 0).all()
    assert (N + lo >= -1e-12).all() and (N + hi >= -1e-12).all()          # R >= 0
    assert (-hi <= N + 1e-12).all() and (-lo <= N + 1e-12).all()          # T <= N
    assert (iv["N"] > 999).any()


# ------------------------------------------------------------------------------------------------ helpers
def _row(level_nll, mc, exact=True):
    return {"level_nll": list(level_nll), "recon_exact": exact, "mc": dict(mc)}


def _data(domain, models, K=12, cuts=None):
    cuts = cuts or list(range(5, K))
    return {"domain": domain, "n": len(next(iter(models.values()))), "cuts": cuts, "rows": models}


def _rows(n_items, K, mc_fn, nll=3.0):
    return [_row([nll] * K, {str(k): mc_fn(i, k) for k in range(8, K)}) for i in range(n_items)]


# ------------------------------------------------------------------------------------------------ AC4
@pytest.mark.parametrize("K,reg", [(11, 3), (12, 4)])
def test_bonferroni_delta_is_alpha_over_items_times_registered_cuts(K, reg):
    rows = _rows(10, K, lambda i, k: [30, 64])
    rows[3]["mc"] = {str(k): [0, 0] for k in range(8, K)}                # x^-invalid items still count
    res = AE.analyse_file(_data("wmt14", {"J0": rows}, K=K, cuts=list(range(5, K))))
    assert res["registered_cuts"] == list(range(8, K)) and len(res["registered_cuts"]) == reg
    assert res["per_model"]["J0"]["delta"] == pytest.approx(0.05 / (10 * reg))
    assert AE.delta_for(64, 4) == pytest.approx(0.05 / 256)
    # descriptive cuts (5-7) use the same per-interval delta
    rows2 = [dict(r, mc={**r["mc"], "6": [0, 64]}) for r in rows]
    res2 = AE.analyse_file(_data("wmt14", {"J0": rows2}, K=K))
    lo, hi = AE.cs_log_interval(0, 64, 0.05 / (10 * reg))
    assert res2["per_model"]["J0"]["intervals"]["6"]["log_p_hi"][0] == pytest.approx(hi, abs=1e-9)
    assert res2["per_model"]["J0"]["summaries"]["6"]["all"]["registered_cut"] is False


# ------------------------------------------------------------------------------------------------ AC5
def _load(name):
    with open(REAL / name, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_result():
    return AE.analyse(str(REAL))


@pytest.mark.parametrize("fn", ["e1_ts__result.json", "e1_wmt__result.json", "e1_sdk__result.json"])
def test_recomputed_summaries_equal_the_saved_ones_on_the_real_files(fn):
    assert AE.check_saved(_load(fn)) == []


def _perturb_summary(d):
    m = next(iter(d["summary"]))
    d["summary"][m]["8"]["N_mean"] *= 1.0001


def _perturb_paired(d):
    key = [k for k in d if k.startswith("paired_vs_")][0]
    m = next(iter(d[key]))
    d[key][m]["8"]["dN_mean"] += 0.01


def _drop_key(d):
    m = next(iter(d["summary"]))
    del d["summary"][m]["8"]["A_mean"]


def _none_for_value(d):
    m = next(iter(d["summary"]))
    d["summary"][m]["8"]["R_lo_share_of_N"] = None


@pytest.mark.parametrize("mut", [_perturb_summary, _perturb_paired, _drop_key, _none_for_value])
def test_a_perturbed_saved_block_fails_the_integrity_check(mut):
    d = _load("e1_ts__result.json")
    mut(d)
    assert AE.check_saved(d)


def test_analyse_asserts_on_a_perturbed_input_directory(tmp_path):
    for fn in ("e1_ts__result.json", "e1_wmt__result.json", "e1_sdk__result.json"):
        d = _load(fn)
        if fn.startswith("e1_wmt"):
            _perturb_summary(d)
        (tmp_path / fn).write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(AssertionError, match="e1_wmt"):
        AE.analyse(str(tmp_path))


# ------------------------------------------------------------------------------------------------ AC6
def test_monotonicity_lists_violations_beyond_the_intervals_only():
    nll = [1.0] * 8 + [0.0, 5.0, 5.0, 5.0]               # N_8 == N_9, A_8 == A_9: only T can move R and U
    bad = _row(nll, {"8": [1, 2048], "9": [64, 64]})      # T_8 ~ 7.6 nats, T_9 ~ 0: R rises by ~7 nats
    ok = _row(nll, {"8": [12, 64], "9": [10, 64]})        # overlapping intervals: within the noise
    iv = AE.build_intervals([ok, bad], [8, 9], 0.001)
    beyond, pts = AE.monotonicity("M", iv)
    assert ["M", 1, 8, "R"] in beyond and ["M", 1, 8, "U"] in beyond
    assert not any(b[1] == 0 for b in beyond)
    assert pts["pairs"] == 2 and pts["R_rise"] >= 1 and pts["U_fall"] >= 1


def test_monotonicity_ignores_non_consecutive_cuts_and_invalid_items():
    nll = [1.0] * 8 + [0.0, 5.0, 5.0, 5.0]
    gap = _row(nll, {"8": [1, 2048], "9": [0, 0], "10": [64, 64]})    # cut 9 invalid: 8 -> 10 not consecutive
    iv = AE.build_intervals([gap], [8, 9, 10], 0.001)
    beyond, pts = AE.monotonicity("M", iv)
    assert beyond == [] and pts["pairs"] == 0


# ------------------------------------------------------------------------------------------------ AC7
def _cell(lo, hi, cens=0.0, width=1.0, N=100.0, items=50):
    return {"items": items, "N_mean": N, "censored": int(cens * items), "censored_share": cens, "R_share_lo": lo,
            "R_share_hi": hi, "T_width_le2_share": width, "point": None}


@pytest.mark.parametrize("lo,hi,want", [(0.8, 0.95, "holds"), (0.79999, 0.99, "neither"), (0.3, 0.4999, "overturned"),
                                       (0.3, 0.5, "neither")])
def test_p1_e1_rule(lo, hi, want):
    assert AE.p1_e1_rule(lo, hi) == want


@pytest.mark.parametrize("lo,hi,cens,width,want", [
    (0.95, 0.99, 0.0, 0.8, "holds"), (0.95, 0.99, 0.0, 0.79, "neither"), (0.90, 0.99, 0.0, 1.0, "neither"),
    (0.5, 0.7999, 0.0, 1.0, "overturned"), (0.5, 0.80, 0.0, 1.0, "neither"),
    (0.95, 0.99, 0.2, 1.0, "holds"), (0.95, 0.99, 0.2001, 1.0, "inconclusive"), (0.1, 0.2, 0.3, 0.0, "inconclusive")])
def test_p1_e1a_rule(lo, hi, cens, width, want):
    assert AE.p1_e1a_rule(lo, hi, cens, width) == want


def test_prediction1_registers_the_right_sets_and_does_not_read_small_N():
    cells = {"A0": {"8": {"all": _cell(0.85, 0.99), "untouched": _cell(0.95, 0.99, width=0.9)}}}
    p = AE.prediction1("tinystories", "A0", cells)
    assert p["sets"]["all"]["e1_rule"] == {"verdict": "holds", "registered": True}
    assert p["sets"]["all"]["e1a_rule"] == {"verdict": "neither", "registered": False}
    assert p["sets"]["untouched"]["e1a_rule"] == {"verdict": "holds", "registered": True}
    w = AE.prediction1("wmt14", "J0", {"J0": {"8": {"all": _cell(0.3, 0.4, cens=0.3)}}})
    assert w["sets"]["all"]["e1_rule"]["verdict"] == "overturned"
    assert w["sets"]["all"]["e1a_rule"] == {"verdict": "inconclusive", "registered": True}
    small = AE.prediction1("sudoku", "S0", {"S0": {"8": {"all": _cell(0.99, 1.0, N=4.99)}}})
    assert small["sets"]["all"]["e1_rule"]["verdict"] == small["sets"]["all"]["e1a_rule"]["verdict"] == "not_read"
    edge = AE.prediction1("sudoku", "S0", {"S0": {"8": {"all": _cell(0.99, 1.0, N=5.0)}}})
    assert edge["sets"]["all"]["e1_rule"]["verdict"] == "holds"


def test_untouched_subset_excludes_items_0_to_2():
    rows = [_row([2.0] * 12, {"8": [0, 2048] if i < 3 else [60, 64]}) for i in range(10)]
    res = AE.analyse_file(_data("tinystories", {"A0": rows}))
    s = res["per_model"]["A0"]["summaries"]["8"]
    assert s["all"]["items"] == 10 and s["all"]["censored"] == 3
    assert s["untouched"]["items"] == 7 and s["untouched"]["censored"] == 0
    assert s["untouched"]["point"]["items"] == 7 and "T_mean" in s["untouched"]["point"]     # run's summarise on the subset
    assert "untouched" not in AE.analyse_file(_data("wmt14", {"J0": rows}))["per_model"]["J0"]["summaries"]["8"]


@pytest.mark.parametrize("dN,lo,hi,want", [
    (-100, -20, 20, "holds"), (-100, -20.01, 20, "neither"), (-100, -50, -50, "carries_half"),
    (-100, -60, -50, "carries_half"), (-100, -60, -49.9, "neither"), (100, 50, 70, "carries_half"),
    (100, 49.9, 70, "neither"), (100, -20, 20, "holds"), (-100, -10, 30, "neither"), (0.0, 0.0, 0.0, "holds"),
    (0.0, 0.1, 0.2, "neither")])
def test_p2_reading(dN, lo, hi, want):
    assert AE.p2_reading(dN, lo, hi) == want


@pytest.mark.parametrize("r,want", [
    (["holds"] * 3, "holds"), (["holds", "holds", "neither"], "neither"),
    (["carries_half", "carries_half", "holds"], "overturned"), (["carries_half", "holds", "neither"], "neither"),
    (["carries_half"] * 3, "overturned"), (["holds"] * 2, "neither")])
def test_p2_overall(r, want):
    assert AE.p2_overall(dict(zip(["A2-A1", "A4-A1", "J1-J0"], r))) == want


def test_p2_compare_propagates_interval_endpoints_and_sign():
    K = 12
    a = [_row([3.0] * K, {"8": [64, 64]}) for _ in range(6)]
    b = [_row([2.0] * K, {"8": [64, 64]}) for _ in range(6)]
    b[5]["mc"]["8"] = [0, 0]                                         # invalid in b: dropped from the pair
    iv = {m: AE.build_intervals(r, [8], 0.01) for m, r in (("a", a), ("b", b))}
    c = AE.p2_compare(a, b, iv["a"], iv["b"])
    lo, hi = AE.cs_log_interval(64, 64, 0.01)
    w = -lo                                                          # T_hi - T_lo = -log p_L
    assert c["items"] == 5 and c["dN_mean"] == pytest.approx(-4 * 1.0)
    assert c["dT_lo"] == pytest.approx(-w) and c["dT_hi"] == pytest.approx(w)
    assert w < 0.2 * 4 and c["reading"] == "holds"                    # interval +/- w inside +/- 0.2 |dN| = 0.8
    assert AE.p2_reading(c["dN_mean"], c["dT_lo"], c["dT_hi"]) == c["reading"]
    assert c["dT_point_mean"] == pytest.approx(0.0)


def test_prediction3_orders_and_quality_comparison():
    def cell(U_lo, U_hi, N, T_lo, T_hi, cens, U_pt=None):
        return {"8": {"all": {"items": 5, "censored": cens, "U_lo_mean": U_lo, "U_hi_mean": U_hi, "N_mean": N,
                              "T_lo_mean": T_lo, "T_hi_mean": T_hi, "point": {"U_mean": U_pt} if U_pt else None}}}
    s = {"J0": cell(10, 14, 130, 1, 5, 2), "J2F": cell(11, 13, 120, 2, 4, 0, U_pt=12.5), "J1": cell(12, 20, 100, 3, 9, 1)}
    p = AE.prediction3("wmt14", s)
    assert p["models"]["J0"]["U"] == 12.0 and p["models"]["J0"]["U_basis"] == "interval midpoint"
    assert p["models"]["J2F"]["U"] == 12.5 and p["models"]["J2F"]["U_basis"].startswith("point")
    assert p["order_by_U"] == ["J0", "J2F", "J1"] and p["order_by_N"] == ["J1", "J2F", "J0"]
    assert p["quality_order"] == ["J0", "J2F", "J1"]
    assert p["U_order_equals_quality"] is True and p["N_order_equals_quality"] is False
    assert p["order_by_T_lo"] == ["J0", "J2F", "J1"]


# ------------------------------------------------------------------------------------------------ AC8
def test_script_runs_on_the_real_files_and_writes_the_json(tmp_path, monkeypatch, capsys, real_result):
    out = tmp_path / "e1.json"
    monkeypatch.setattr(sys, "argv", ["analyze_e1.py", "--in_dir", str(REAL), "--out", str(out)])
    AE.main()
    res = json.loads(out.read_text(encoding="utf-8"))
    assert set(res["domains"]) == {"tinystories", "wmt14", "sudoku"}
    assert set(res["inputs"]) == {"e1_ts__result.json", "e1_wmt__result.json", "e1_sdk__result.json"}
    assert res["inputs"]["e1_ts__result.json"] == AE.sha256(str(REAL / "e1_ts__result.json"))
    assert res["domains"]["tinystories"]["per_model"]["A0"]["delta"] == pytest.approx(0.05 / (64 * 4))
    assert res["domains"]["wmt14"]["per_model"]["J0"]["delta"] == pytest.approx(0.05 / (128 * 3))
    assert res["domains"]["sudoku"]["per_model"]["S0"]["delta"] == pytest.approx(0.05 / (128 * 4))
    assert set(res["domains"]["tinystories"]["prediction1"]["sets"]) == {"all", "untouched"}
    assert res["domains"]["tinystories"]["prediction1"]["sets"]["untouched"]["items"] == 61
    assert set(res["domains"]["tinystories"]["prediction2_comparisons"]) == {"A0-A1", "A2-A1", "A4-A1"}
    assert set(res["domains"]["wmt14"]["prediction2_comparisons"]) == {"J1-J0", "J2F-J0"}
    assert res["prediction2_overall"]["verdict"] in {"holds", "overturned", "neither"}
    assert all(v["saved_summary_and_paired_match_recomputation"] for v in res["integrity"].values())
    for dom in res["domains"].values():
        for m in dom["per_model"].values():
            assert m["monotonicity_beyond_intervals"] == [] or isinstance(m["monotonicity_beyond_intervals"], list)
    assert "P1" in capsys.readouterr().out
    # the in-memory result and the file agree on every verdict
    assert res["domains"]["sudoku"]["prediction1"] == AE._clean(real_result["domains"]["sudoku"]["prediction1"])
