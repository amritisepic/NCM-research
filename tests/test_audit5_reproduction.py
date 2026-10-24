"""Audit 5: exercise the live W1d reproduction helper without GPU or LM downloads.

The defect concerns score/seed cardinality, so model weights and the hash-pinned
released-model fixture add no coverage here. Normal pytest loads conftest.py.
Only assertion failures are expected: import/setup errors must remain errors.
"""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reproduction():
    spec = importlib.util.spec_from_file_location(
        "audit5_diag_w1_frontier", ROOT / "scripts/diag_w1_frontier.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reproduction


@pytest.fixture
def expected_groups():
    return {
        "config": {"sample_eval": {"final_seeds": [42, 0, 1, 2]}},
        "final_eval": {"genppl_gpt2-large_per_group": [10.0, 11.0, 12.0, 13.0]},
    }


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="A5-1: zip drops requested groups")
@pytest.mark.parametrize("seeds", [[0, 1], [0, 999]])
def test_reproduction_rejects_missing_group_score(reproduction, expected_groups, seeds):
    # Seed 1 has a reference and seed 999 does not; neither has a current score.
    # Both must fail, rather than silently checking seed 0 alone.
    assert not reproduction([10.0], seeds, expected_groups)["ok"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="A5-1: duplicate seeds collapse")
def test_reproduction_rejects_duplicate_seed(reproduction, expected_groups):
    # evaluate_samples produces one score per distinct group. This is the shape
    # that --seeds 0 0 supplies to reproduction after grouping identical draws.
    assert not reproduction([10.0], [0, 0], expected_groups)["ok"]


def test_reproduction_allows_reordering_unique_seeds(reproduction, expected_groups):
    # Scores arrive in sorted group order, regardless of sampling order.
    assert reproduction([10.0, 11.0, 13.0], [42, 0, 1], expected_groups)["ok"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 10.02])
def test_reproduction_rejects_invalid_or_changed_score(reproduction, expected_groups, bad):
    assert not reproduction([bad, 11.0], [1, 0], expected_groups)["ok"]
