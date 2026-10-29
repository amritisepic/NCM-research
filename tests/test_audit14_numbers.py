"""Audit 14: independent, standard-library arithmetic on pinned saved records.

No analysis/sampling function from the repository is used here. JS-window and
root-reliance-window arrays were not committed: their stored bootstrap bounds
are transcribed, not independently bootstrapped. Shares and jackknife errors
are reconstructed from log(per-group Gen-PPL), rather than stored shares.
"""
import hashlib
import json
import math
import operator
import re
import statistics as st

import pytest

from conftest import ROOT
from test_audit12_readings import pinned_result as w2c_result


SCORERS = ("gpt2-large", "TinyStories-33M")
CUTS = (2, 4, 6)
RESULT_PINS = {
    0: "ef78b288a4d702582e858a2cf05753a45f636bd7e7351d70fc44bbf6c8fce393",
    1: "46515b911aa803bf1df07dd2ff6fdee0fbe86f9cefebe8ed23850381b644c32c",
    2: "31555fa891fd9eec5ca39aa7799dc561ae92ac841e948c5852b3c9f5c2776ce4",
    3: "daf61e475b3a129f501f6dd7eafd4b3ac757dda58c2afd7262274a23d450de93",
}
PROVENANCE_PINS = {
    1: "def25097a566fae10047afe169d4da8081be2808ec02a07910ceb1603ead5b62",
    2: "4693249257f2fdf08d5eef9749b797e7bcf55c044f6c17de50e2a90fb3b3c53e",
    3: "64a498f7ce926317aee60a3f0e6912edd9ed2532ea4bb36a0312e1d03ac7e7e9",
}
META_PIN = "1019b0dff7d7e4fb8831ed3eac7b13cba17968329946e3196f8659fce8f1dae7"
EXTRA_PINS = {
    "results/colab/w1_L_ts_ctrl/L_ts_released16__result.json":
        "3c4be8cdbf49d97efb599f647d407442d6a3183837623964418fc39e1e3ef516",
    "results/colab/w2p_L_ts_m3_c0/L_ts_m3_uniform_wrong_c0_s0__result.json":
        "9e7bb2ec0609fcffc7798067dea8358814114dc769382c898e1a013f265e5102",
    "results/colab/w2m_L_ts_m3/L_ts_m3_uniform_wrong_s0__result.json":
        "afd280811fca660f1df3abbe9910b92c91c73869008410bebe2c2149f2e3baae",
}


def pinned_bytes(relative, digest):
    raw = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == digest, relative
    return raw


def result_path(seed):
    if seed == 0:
        return "results/colab/w2mg_eval/w2mg_eval__result.json"
    return f"results/colab/w2mgr_eval/w2mgr_eval_s{seed}__result.json"


def result(seed):
    return json.loads(pinned_bytes(result_path(seed), RESULT_PINS[seed]))


def estimates(vectors):
    a, b, x, y = map(st.mean, vectors)
    delta = a - b
    pi, sigma = (x - b) / delta, (y - b) / delta
    return {"delta": delta, "pi": pi, "sigma": sigma, "interaction": 1 - pi - sigma}


def shares(record, scorer, cut):
    keys = ("m2_m2", "m3_m3", f"m2to3_c{cut}", f"m3to2_c{cut}")
    vectors = [[math.log(p) for p in record["cells"][k]["genppl_" + scorer]["per_group"]] for k in keys]
    assert all(len(v) == 8 for v in vectors)
    point = estimates(vectors)
    leave_one_out = [estimates([v[:i] + v[i + 1:] for v in vectors]) for i in range(8)]
    for key in ("pi", "sigma", "interaction"):
        center = st.mean(v[key] for v in leave_one_out)
        point[key + "_se"] = math.sqrt(7 / 8 * sum((v[key] - center) ** 2 for v in leave_one_out))
    return point


def own_read_ii(cuts):
    early = cuts["4"]["pi"] >= .5 and cuts["6"]["pi"] >= .75
    conditional = cuts["6"]["sigma"] >= .8
    return "both" if early and conditional else "early-state selection" if early else "conditional rendering" if conditional else "distributed"


def reference_comparison(record, seed):
    out = {}
    for arm in ("m2", "m3"):
        replay = record["cells"][arm + "_" + arm]["genppl_gpt2-large"]["per_group"]
        final = w2c_result(arm, seed)["final_eval"]["genppl_gpt2-large_per_group"]
        gm, fm = st.mean(replay), st.mean(final)
        gs, fs = st.stdev(replay) / math.sqrt(8), st.stdev(final) / 4
        out[arm] = {"replay_mean": gm, "replay_se": gs, "w2c_final_mean": fm, "w2c_final_se": fs,
                    "difference": gm - fm, "z_replay_se_only": (gm - fm) / gs,
                    "z_quadrature_independence_sensitivity": (gm - fm) / math.hypot(gs, fs)}
    finals = [w2c_result(a, seed)["final_eval"]["genppl_gpt2-large_per_group"] for a in ("m2", "m3")]
    out["w2c_D_genppl"] = st.mean(finals[0]) - st.mean(finals[1])
    out["w2c_delta_mean_log_loss"] = st.mean(map(math.log, finals[0])) - st.mean(map(math.log, finals[1]))
    out["replay_D_genppl"] = out["m2"]["replay_mean"] - out["m3"]["replay_mean"]
    out["replay_delta_mean_log_loss"] = shares(record, "gpt2-large", 2)["delta"]
    return out


def build_numbers():
    rows = {}
    for seed in range(4):
        r = result(seed)
        sh = {sc: {str(c): shares(r, sc, c) for c in CUTS} for sc in SCORERS}
        js = {bank: st.mean(v["per_level"]) for bank, v in r["i"]["J"].items()}
        ratios = {bank: {"r_recomputed_from_saved_J_means": js[bank] / js["clean"],
                         "lo_stored": r["i"]["ratios"][bank]["lo"],
                         "hi_stored": r["i"]["ratios"][bank]["hi"]} for bank in ("released", "m2", "m3")}
        rho = r["iii"]["rho"]
        third = {"rho_stored": rho, "m3_minus_m2_recomputed_from_saved_means": rho["m3"]["mean"] - rho["m2"]["mean"],
                 "se_stored": r["iii"]["m3_minus_m2"]["se"], "lo_stored": r["iii"]["m3_minus_m2"]["lo"],
                 "hi_stored": r["iii"]["m3_minus_m2"]["hi"]}
        sanity_key = "sanity_vs_w2c_final_eval" if seed else "sanity_vs_w2m"
        sanity = {}
        for arm in ("m2", "m3"):
            values = r["cells"][arm + "_" + arm]["genppl_gpt2-large"]["per_group"]
            ref = r["ii"][sanity_key]["ref"][arm]
            sanity[arm] = (st.mean(values) - ref) / (st.stdev(values) / math.sqrt(8))
        rows[str(seed)] = {"exploratory_never_pooled": seed == 0, "shares_recomputed": sh,
                           "ii_reading_recomputed": {sc: own_read_ii(sh[sc]) for sc in SCORERS},
                           "i": {"J_recomputed_from_saved_level_means": js, "ratios": ratios,
                                 "reading_from_point_ratios_and_stored_bounds": "holds" if all(v["r_recomputed_from_saved_J_means"] >= 3 and v["lo_stored"] > 1.2 for v in ratios.values()) else "other"},
                           "iii": third, "sanity_z_recomputed": sanity}
        if seed:
            rows[str(seed)]["w2c_comparison_recomputed"] = reference_comparison(r, seed)
    aggregate = {}
    for sc in SCORERS:
        aggregate[sc] = {}
        for cut in CUTS:
            aggregate[sc][str(cut)] = {}
            for measure in ("delta", "pi", "sigma", "interaction"):
                values = [rows[str(seed)]["shares_recomputed"][sc][str(cut)][measure] for seed in (1, 2, 3)]
                aggregate[sc][str(cut)][measure] = {"mean": st.mean(values), "seed_sd_ddof1": st.stdev(values)}
    margins = {"maximum_pi2": max(rows[str(s)]["shares_recomputed"]["gpt2-large"]["2"]["pi"] for s in (1, 2, 3)),
               "minimum_pi6": min(rows[str(s)]["shares_recomputed"]["gpt2-large"]["6"]["pi"] for s in (1, 2, 3)),
               "maximum_sigma6": max(rows[str(s)]["shares_recomputed"]["gpt2-large"]["6"]["sigma"] for s in (1, 2, 3)),
               "minimum_interaction_c4_c6": min(rows[str(s)]["shares_recomputed"]["gpt2-large"][str(c)]["interaction"] for s in (1, 2, 3) for c in (4, 6)),
               "minimum_i_lower_bound_stored": min(rows[str(s)]["i"]["ratios"][b]["lo_stored"] for s in (1, 2, 3) for b in ("released", "m2", "m3"))}
    minutes = {}
    meta = json.loads(pinned_bytes("results/colab/w2mgr_eval/_meta.json", META_PIN))
    for seed in (1, 2, 3):
        raw = pinned_bytes(f"results/colab/w2mgr_eval/w2mgr_eval_s{seed}__provenance.jsonl", PROVENANCE_PINS[seed])
        events = [json.loads(line) for line in raw.splitlines()]
        assert events == meta["sessions"][f"w2mgr_eval_s{seed}"]
        assert [e["event"] for e in events] == ["start", "end"]
        assert events[0]["had_ckpt"] is False and events[1]["exit"] == 0
        minutes[str(seed)] = events[1]["minutes"]
    released, no_clean, m3zero = [json.loads(pinned_bytes(p, pin))["final_eval"] for p, pin in EXTRA_PINS.items()]
    released_g = st.mean(released["genppl_gpt2-large_per_group"])
    m3zero_g = st.mean(m3zero["genppl_gpt2-large_per_group"])
    no_clean_g = st.mean(no_clean["genppl_gpt2-large_per_group"])
    drift = {"released_genppl": released_g, "m3_seed0_genppl": m3zero_g, "m3_noclean_seed0_genppl": no_clean_g,
             "m3_seed0_minus_released": m3zero_g - released_g,
             "m3_seeds1to3_mean_minus_released": st.mean(w2c_result("m3", s)["final_eval"]["genppl_gpt2-large"] for s in (1, 2, 3)) - released_g,
             "m3_seed0_minus_noclean": m3zero_g - no_clean_g,
             "m3_noclean_minus_released": no_clean_g - released_g}
    return {"audited_commit": "3b601dcd4d39ad9b69fdc9c509165542ea151260", "method": "independent json/math/statistics arithmetic; no repository analysis functions",
            "ci_limit": "(i) and (iii) bootstrap bounds and rho marginal SEs are saved measurements, copied and labelled stored; per-window arrays are not committed, so bootstrap recomputation is unavailable",
            "quadrature_limit": "Both-SE z is a sensitivity calculation assuming independent estimates; overlapping seeds and different GPU/sampler implementations prevent certifying that independence from these files",
            "per_seed": rows, "aggregate_seeds1to3_only": aggregate, "registered_margin_extrema": margins,
            "provenance_minutes": minutes, "total_L4_minutes": sum(minutes.values()), "estimated_CU_at_1_71_per_hour": sum(minutes.values()) * 1.71 / 60,
            "released_control_drift": drift}


@pytest.mark.parametrize("seed", range(4))
def test_all_shares_and_jackknife_errors_from_group_genppl(seed):
    r = result(seed)
    for sc in SCORERS:
        for c in CUTS:
            expected = shares(r, sc, c)
            stored = r["ii"]["shares"][sc][str(c)]
            assert stored == pytest.approx(expected, abs=1e-12)
            for key in ("m2_m2", "m3_m3", f"m2to3_c{c}", f"m3to2_c{c}"):
                metric = r["cells"][key]["genppl_" + sc]
                assert metric["log_loss_per_group"] == pytest.approx(list(map(math.log, metric["per_group"])), abs=1e-12)
        assert own_read_ii({str(c): shares(r, sc, c) for c in CUTS}) == r["ii"]["reading"][sc] == "distributed"


def test_r065_table_and_seed_sd_round_correctly():
    numbers = build_numbers()
    expected = {
        0: ["0.009", "0.220", "0.511", "0.061", "0.469", "0.428"],
        1: ["0.011", "0.226", "0.540", "0.042", "0.445", "0.418"],
        2: ["0.025", "0.232", "0.524", "0.051", "0.445", "0.425"],
        3: ["0.011", "0.211", "0.517", "0.067", "0.441", "0.416"],
    }
    for seed, row in expected.items():
        sh = numbers["per_seed"][str(seed)]["shares_recomputed"]["gpt2-large"]
        values = [sh["2"]["pi"], sh["4"]["pi"], sh["6"]["pi"], sh["6"]["sigma"], sh["4"]["interaction"], sh["6"]["interaction"]]
        assert [f"{v:.3f}" for v in values] == row
    ag = numbers["aggregate_seeds1to3_only"]["gpt2-large"]
    for cut, measure, mean, sd in [(2, "pi", ".016", ".008"), (4, "pi", ".223", ".011"), (6, "pi", ".527", ".012"), (6, "sigma", ".053", ".013")]:
        v = ag[str(cut)][measure]
        assert f"{v['mean']:.3f}" == "0" + mean
        assert f"{v['seed_sd_ddof1']:.3f}" == "0" + sd
    assert f"{ag['4']['interaction']['mean']:.3f}" == "0.444"
    assert f"{ag['6']['interaction']['mean']:.3f}" == "0.420"
    ts_expected = {0: ["0.012", "0.487", "0.055"], 1: ["0.005", "0.495", "0.024"],
                   2: ["0.018", "0.495", "0.045"], 3: ["0.012", "0.479", "0.056"]}
    for seed, row in ts_expected.items():
        sh = numbers["per_seed"][str(seed)]["shares_recomputed"]["TinyStories-33M"]
        assert [f"{v:.3f}" for v in (sh["2"]["pi"], sh["6"]["pi"], sh["6"]["sigma"])] == row
    ts = numbers["aggregate_seeds1to3_only"]["TinyStories-33M"]
    assert [f"{v:.3f}" for v in (ts["2"]["pi"]["mean"], ts["6"]["pi"]["mean"], ts["6"]["sigma"]["mean"])] == ["0.012", "0.490", "0.042"]


def test_i_point_ratios_iii_mean_differences_and_stored_ci_readings():
    for seed in range(4):
        r = result(seed)
        j = {k: st.mean(v["per_level"]) for k, v in r["i"]["J"].items()}
        for bank in ("released", "m2", "m3"):
            ratio = r["i"]["ratios"][bank]
            assert j[bank] == pytest.approx(r["i"]["J"][bank]["J"], abs=1e-14)
            assert j[bank] / j["clean"] == pytest.approx(ratio["r"], abs=1e-12)
            assert ratio["r"] >= 3 and ratio["lo"] > 1.2
        assert r["i"]["reading"] == "holds"
        rho = r["iii"]["rho"]
        contrast = r["iii"]["m3_minus_m2"]
        assert rho["m3"]["mean"] - rho["m2"]["mean"] == pytest.approx(contrast["mean"], abs=1e-12)
        assert contrast["lo"] > 0 and r["iii"]["reading"] == "holds"


def test_r065_i_ranges_and_iii_printed_values():
    numbers = build_numbers()
    rows = [numbers["per_seed"][str(s)] for s in (1, 2, 3)]
    early = [r["i"]["ratios"][b]["r_recomputed_from_saved_J_means"] for r in rows for b in ("released", "m2")]
    control = [r["i"]["ratios"]["m3"]["r_recomputed_from_saved_J_means"] for r in rows]
    clean = [r["i"]["J_recomputed_from_saved_level_means"]["clean"] for r in rows]
    assert [f"{v:.1f}" for v in (min(early), max(early), min(control), max(control))] == ["6.2", "7.0", "5.4", "5.5"]
    assert [f"{v:.4f}" for v in (min(clean), max(clean))] == ["0.0120", "0.0126"]
    assert f"{numbers['registered_margin_extrema']['minimum_i_lower_bound_stored']:.2f}" == "5.16"
    expected = [["1.38", "1.27", "1.50"], ["1.40", "1.29", "1.53"], ["1.38", "1.27", "1.50"]]
    for row, printed in zip(rows, expected):
        third = row["iii"]
        assert [f"{third[k]:.2f}" for k in ("m3_minus_m2_recomputed_from_saved_means", "lo_stored", "hi_stored")] == printed
        assert f"{third['rho_stored']['released']['mean']:.2f}" == "3.04"
    for arm, endpoints in (("m2", ["1.99", "2.05"]), ("m3", ["3.38", "3.43"])):
        values = [r["iii"]["rho_stored"][arm]["mean"] for r in rows]
        assert [f"{v:.2f}" for v in (min(values), max(values))] == endpoints


def test_sanity_matches_independent_group_means_and_standard_errors():
    expected = {0: ["-0.96", "-1.06"], 1: ["-1.66", "-0.50"], 2: ["-1.45", "+0.26"], 3: ["-0.42", "-0.86"]}
    for seed in range(4):
        row = build_numbers()["per_seed"][str(seed)]
        assert [f"{row['sanity_z_recomputed'][a]:+.2f}" for a in ("m2", "m3")] == expected[seed]
        assert all(abs(v) <= 3 for v in row["sanity_z_recomputed"].values())


def test_provenance_and_released_control_drift():
    numbers = build_numbers()
    assert numbers["provenance_minutes"] == {"1": 18.88, "2": 18.50, "3": 18.11}
    assert numbers["total_L4_minutes"] == pytest.approx(55.49)
    drift = numbers["released_control_drift"]
    assert drift["m3_seed0_minus_released"] == pytest.approx(5.521368675253963)
    assert drift["m3_seeds1to3_mean_minus_released"] == pytest.approx(5.51244010125675)
    assert drift["m3_seed0_minus_noclean"] == pytest.approx(5.189515509258093)


def test_r065_literal_pi2_upper_bound_covers_observed_value():
    text = (ROOT / "log.md").read_text(encoding="utf-8").split("## R065", 1)[1]
    # The log is append-only: an appended correction supersedes the first
    # margin statement without rewriting the historical R065 entry.
    claims = re.findall(r"with margins: pi_2\s*(<=|<)\s*([0-9.]+)", text)
    if claims:
        relation, value = claims[-1]
        bound = float(value)
        actual = max(shares(result(s), "gpt2-large", 2)["pi"] for s in (1, 2, 3))
        assert {"<=": operator.le, "<": operator.lt}[relation](actual, bound), (actual, bound)


def test_r065_literal_interaction_lower_bound_covers_observed_value():
    text = (ROOT / "log.md").read_text(encoding="utf-8").split("## R065", 1)[1]
    claims = re.findall(r"interactions\s*(>=|>)\s*([0-9.]+)\s+vs", text)
    if claims:
        relation, value = claims[-1]
        bound = float(value)
        actual = min(shares(result(s), "gpt2-large", c)["interaction"] for s in (1, 2, 3) for c in (4, 6))
        assert {">=": operator.ge, ">": operator.gt}[relation](actual, bound), (actual, bound)
