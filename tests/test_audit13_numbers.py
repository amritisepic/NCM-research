"""Audit 13: W2C sanity inputs, pinned before any W2M-GR measurement.

The shared CPU setup is tests/conftest.py. No model download is needed.
These are positive arithmetic/source checks; defect probes are separate.
"""
import hashlib
import importlib.util
import json
import statistics
import sys

import pytest

from conftest import ROOT


# (arm, seed): (result digest, GPT-2 large Gen-PPL recomputed from 16 groups).
SOURCES = {
    ("m2", 1): ("54bf7cfd0538faeb2cc32713ded4e2c662572e5e2c97803cacd1e6bde2181a59", 39.790341371937075),
    ("m2", 2): ("e34cbb08b55442d3c5635c058333827012e9d12f22b4927280abd37303a52061", 41.30791192128558),
    ("m2", 3): ("0e2d1191dbeab52dee34aa8a5978a5e74dda7b7987d1827bdfb6491f3be69582", 40.56038277086411),
    ("m3", 1): ("e0e2d7c2eb8bdda56afb1709f9cefbaa727030b8adc266fbe9d6f1aa2a66c772", 21.425413127057226),
    ("m3", 2): ("5d12a8807c977be276e2b254c59b24fe7aea6cd349a8b3f89ba6df1aa6e89725", 21.8019284403825),
    ("m3", 3): ("259ceec06496c0927b2811aabd3aa324f129197d7ed77004318b1f9a7714c94b", 21.803127274556942),
}
DIRS = {"m2": "L_ts_m2_frozen_pw", "m3": "L_ts_m3_uniform_wrong"}


def source_path(arm, seed):
    return f"results/colab/w2c1_L_ts_{arm}/{DIRS[arm]}_s{seed}__result.json"


def pinned(arm, seed):
    rel = source_path(arm, seed)
    raw = (ROOT / rel).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCES[arm, seed][0], rel
    return json.loads(raw)


@pytest.mark.parametrize("arm,seed", sorted(SOURCES))
def test_w2c_sanity_reference_equals_its_sixteen_group_mean(arm, seed):
    r = pinned(arm, seed)
    assert r["run"] == f"{DIRS[arm]}_s{seed}"
    assert r["w2m_arm"] == arm
    assert r["config"]["train"]["seed"] == seed
    assert r["steps"] == 600
    groups = r["final_eval"]["genppl_gpt2-large_per_group"]
    assert len(groups) == 16
    expected = SOURCES[arm, seed][1]
    assert statistics.mean(groups) == pytest.approx(expected, rel=0, abs=1e-12)
    assert r["final_eval"]["genppl_gpt2-large"] == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.fixture(scope="module")
def eval_script():
    spec = importlib.util.spec_from_file_location("audit13_reference_eval", ROOT / "scripts/w2mg_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_real_reference_reader_selects_only_the_requested_w2c_seed(eval_script, seed):
    ref, src = eval_script.w2c_reference(seed)
    assert set(ref) == set(src) == {"m2", "m3"}
    for arm in ("m2", "m3"):
        r = pinned(arm, seed)
        assert src[arm] == source_path(arm, seed)
        assert ref[arm] == r["final_eval"]["genppl_gpt2-large"]


@pytest.mark.parametrize("seed,rounded", [(1, 18.36), (2, 19.51), (3, 18.76)])
def test_r062_gpt2_contrast_rounding(seed, rounded):
    groups = [pinned(arm, seed)["final_eval"]["genppl_gpt2-large_per_group"] for arm in ("m2", "m3")]
    difference = statistics.mean(a - b for a, b in zip(*groups))
    assert round(difference, 2) == rounded
