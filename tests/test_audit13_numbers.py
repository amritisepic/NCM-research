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
    ("m2", 1): ("d079dc520e2cbc1b93530c3701aafc7e90b99f3b03ec78ad50a9b3423109c989", 39.790341371937075),
    ("m2", 2): ("a837c0dd61bdee8413d693700cf844b8f85f9b711a9b7367690051eb3b85829e", 41.30791192128558),
    ("m2", 3): ("33b9c036e6ad57f51469ef5760f840c4d9d03cefcdf8a8672707c600c0fd2a9e", 40.56038277086411),
    ("m3", 1): ("bd6898e17fc1c3a63363a2ba2bfd97201f296b47da4640b971525a69c644d959", 21.425413127057226),
    ("m3", 2): ("f7761c8391762747f18df6848c7a18c081b45d0ad1043409175830d9fbcca15d", 21.8019284403825),
    ("m3", 3): ("98cf9e5d95c24e07cf88a4869b57e3f888ed6371510d7b73ffb02a0dda54f66c", 21.803127274556942),
}
DIRS = {"m2": "L_ts_m2_frozen_pw", "m3": "L_ts_m3_uniform_wrong"}


def source_path(arm, seed):
    return f"results/colab/w2c1_L_ts_{arm}/{DIRS[arm]}_s{seed}__result.json"


def pinned(arm, seed):
    rel = source_path(arm, seed)
    raw = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")  # git's stored bytes on any checkout
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
