"""Audit 14: actual registered files satisfy the fixed admission protocol.

CPU only; uses the unchanged conftest.py and LF-normalized SHA-256 pins.
These positive checks are not fresh GPU measurements or evidence that the
W2C writer independently recorded final-checkpoint digests (it did not).
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit14_protocol_analyzer", ROOT / "scripts/analyze_w2mgr.py"
)
AN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AN)

PINS = {
    0: "ef78b288a4d702582e858a2cf05753a45f636bd7e7351d70fc44bbf6c8fce393",
    1: "46515b911aa803bf1df07dd2ff6fdee0fbe86f9cefebe8ed23850381b644c32c",
    2: "31555fa891fd9eec5ca39aa7799dc561ae92ac841e948c5852b3c9f5c2776ce4",
    3: "daf61e475b3a129f501f6dd7eafd4b3ac757dda58c2afd7262274a23d450de93",
}
MANIFEST_PIN = "a2ab2328575403d0c2e56f1b5ad13eadaa4580d3101bd5607ff1a248da42964e"


def pinned_json(relative, digest):
    raw = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == digest, relative
    return json.loads(raw)


def result(seed):
    relative = (
        "results/colab/w2mg_eval/w2mg_eval__result.json" if seed == 0 else
        f"results/colab/w2mgr_eval/w2mgr_eval_s{seed}__result.json"
    )
    return pinned_json(relative, PINS[seed])


def test_committed_analysis_is_exactly_reproduced_without_rewriting_it():
    root = str(ROOT / "results" / "colab")
    output = {
        "w2mgr": AN.analyze_w2mgr(root),
        "seed0_beside": AN.analyze_seed0(root),
    }
    saved = (ROOT / "results/w2mgr_analysis.json").read_bytes().replace(b"\r\n", b"\n")
    assert output == json.loads(saved)
    assert json.dumps(output, indent=1).encode("utf-8") == saved


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_committed_replay_is_accepted_for_its_actual_seed(seed):
    record = result(seed)
    assert AN.check_result(record, seed) == []
    assert AN.check_result(record, seed % 3 + 1)
    protocol = record["protocol"]
    assert protocol["bank_seeds"] == [900, 901]
    assert protocol["replay_seeds"] == list(range(8))
    assert protocol["identity"]["gpu"] == "NVIDIA L4"
    for cell in record["cells"].values():
        assert cell["spec"]["seeds"] == list(range(8))
        assert cell["n_groups"] == 8 and cell["n"] == 2048
        for scorer in AN.SCORERS:
            assert len(cell[f"genppl_{scorer}"]["per_group"]) == 8


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_released_assets_and_validation_hash_match_seed0_and_w2c(seed):
    identity = result(seed)["protocol"]["identity"]
    seed0_identity = result(0)["protocol"]["identity"]
    for key in ("released_sha256", "val_sha256"):
        assert identity[key] == seed0_identity[key]
    for arm, run in AN.RUN_DIRS.items():
        manifest = pinned_json(
            f"results/colab/{AN.W2C_NB[arm]}/{run}_s{seed}__run_manifest.json",
            MANIFEST_PIN,
        )
        assert identity["released_sha256"] == manifest["base_sha256"]
        assert identity["val_sha256"] == manifest["data_sha256"]["valid_bpe_8192.pickle"]


@pytest.mark.parametrize("departure", ["test_bank", "wrong_cut", "wrong_seeds", "four_groups"])
def test_a13_1_fix_rejects_departures_from_the_actual_seed2_result(departure):
    record = copy.deepcopy(result(2))
    assert AN.check_result(record, 2) == []
    cell = record["cells"]["m2to3_c2"]
    if departure == "test_bank":
        record["protocol"]["clean_bank"] = 'examples("test", 512)'
    elif departure == "wrong_cut":
        cell["spec"]["cut"] = 4
    elif departure == "wrong_seeds":
        cell["spec"]["seeds"] = list(range(8, 16))
    else:
        # Keep the local statistics mutually consistent; the actual group
        # count must expose this even though the advertised count stays 8.
        for scorer in AN.SCORERS:
            metric = cell[f"genppl_{scorer}"]
            values = metric["per_group"][:4]
            sd = statistics.stdev(values)
            metric.update(
                per_group=values, log_loss_per_group=[math.log(x) for x in values],
                mean=statistics.mean(values), sd=sd, se=sd / 2,
            )
    reasons = AN.check_result(record, 2)
    assert reasons
    expected = {
        "test_bank": "protocol clean_bank", "wrong_cut": "cell m2to3_c2: spec",
        "wrong_seeds": "cell m2to3_c2: spec", "four_groups": "needs 8 finite group",
    }
    assert any(expected[departure] in reason for reason in reasons)
