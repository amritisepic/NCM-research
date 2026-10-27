"""Recompute R039 from its result JSONs, without copying results onto the audit branch.

R039 and its inputs were committed after the audited revision. Prefer the checked-out
inputs, otherwise read their pinned Git objects. No network or file writes.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "daf2b57ee5bd9296ccd200fb71c03cb32517ef80"
DIGESTS = (
    "035924e6d15b6bdd13b23d5cb7ac7af72bb0c607af5c4ea89f1d6d62347ef289",
    "57606703e410efeecccbeb8e268d657de00214ffd3e42b882369fe640c9ab567",
    "aaed35bd85c01e0bfd22096bb5399338fb1fb294bcae6ef84ef0296cf6651f0b",
)


def read_json(path):
    local = ROOT / path
    if local.is_file():
        # a Windows checkout with core.autocrlf rewrites LF as CRLF; the pinned digests are of the committed blobs
        return local.read_bytes().replace(b"\r\n", b"\n")
    r = subprocess.run(["git", "show", SOURCE + ":" + path], cwd=ROOT, capture_output=True)
    if r.returncode:
        pytest.skip("R039 inputs absent at audited commit; fetch the pinned R039 source commit")
    return r.stdout


@pytest.fixture(scope="module")
def results():
    out = []
    for i in range(3):
        raw = read_json(f"results/colab/w1d_requant_diag/w1d_A{i}__result.json")
        assert hashlib.sha256(raw).hexdigest() == DIGESTS[i]
        out.append(json.loads(raw))
    return out


def paired(a, b):
    d = [y - x for x, y in zip(a, b)]
    return mean(d), stdev(d) / math.sqrt(len(d))


@pytest.mark.parametrize("section,key,expected", [
    ("frontier", "0.8", (0.21601283061648324, 0.07731182250294748)),
    ("frontier", "0.9", (0.048882722675894774, 0.10000725151827562)),
    ("frontier", "1.0", (-0.33219567968365404, 0.1222504856217286)),
    ("oracle", "-1", (-0.13613188321441694, 0.09442339989733227)),
    ("oracle", "3", (0.4162592064886401, 0.1645843617820553)),
    ("oracle", "5", (0.10770731488753915, 0.028532562622885248)),
    ("oracle", "7", (0.022735701790405294, 0.007749122738811824)),
])
def test_r039_paired_mean_and_sample_standard_error(results, section, key, expected):
    vals = [r[section][key]["genppl_gpt2-large_per_group"] for r in results]
    assert len(vals[1]) == len(vals[2]) == (8 if section == "frontier" else 4)
    got = paired(vals[1], vals[2])
    print(section, key, "A2-A1 mean/SE", got, "A1-A0 mean/SE", paired(vals[0], vals[1]))
    assert got == pytest.approx(expected, abs=1e-12)


def test_r039_frontier_average_preserves_seed_pairing(results):
    d = [[c - b for b, c in zip(results[1]["frontier"][t]["genppl_gpt2-large_per_group"],
                                results[2]["frontier"][t]["genppl_gpt2-large_per_group"])]
         for t in ("0.8", "0.9", "1.0")]
    per_seed = [mean(v) for v in zip(*d)]
    assert (mean(per_seed), stdev(per_seed) / math.sqrt(8)) == pytest.approx(
        (-0.022433375463758683, 0.07562595295472375), abs=1e-12)


def test_r039_aggregates_integrity_gate_and_rounding_disagreements(results):
    wave1 = [("w1_L_ts_ctrl", "L_ts_released16"), ("w1_L_ts_ctrl", "L_ts_ctrl_s0"),
             ("w1_L_ts_requant", "L_ts_requant_s0")]
    for r, (directory, name) in zip(results, wave1):
        prior = json.loads((ROOT / "results/colab" / directory / (name + "__result.json")).read_text())
        seed_scores = dict(zip(sorted(prior["config"]["sample_eval"]["final_seeds"]),
                               prior["final_eval"]["genppl_gpt2-large_per_group"]))
        assert r["frontier"]["1.0"]["genppl_gpt2-large_per_group"] == [seed_scores[s] for s in sorted(r["args"]["seeds"])]
        for section in ("frontier", "oracle"):
            for metrics in r[section].values():
                for name in ("gpt2-large", "TinyStories-33M"):
                    key = "genppl_" + name
                    if key in metrics:
                        assert mean(metrics[key + "_per_group"]) == pytest.approx(metrics[key], abs=1e-12)
                        assert stdev(metrics[key + "_per_group"]) == pytest.approx(metrics[key + "_sd"], abs=1e-12)
                if "mauve" in metrics:
                    assert mean(metrics["mauve_per_seed"]) == pytest.approx(metrics["mauve"], abs=1e-12)
        gate = r["frontier"]["1.0"]["reproduction"]
        assert gate["ok"] and gate["valid_groups"] and not gate["missing_seeds"]
        assert max(abs(a - b) for a, b in gate["pairs"].values()) == gate["max_abs_diff"] == 0.0
    assert round(results[0]["oracle"]["5"]["genppl_gpt2-large"], 2) == 9.51  # log: 9.52
    _, se = paired(results[1]["oracle"]["3"]["genppl_gpt2-large_per_group"],
                   results[2]["oracle"]["3"]["genppl_gpt2-large_per_group"])
    assert round(se, 2) == 0.16  # log: 0.17 (also repeated in its exploratory paragraph)
