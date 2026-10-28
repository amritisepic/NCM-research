"""Audit 13 numerical boundary characterization; these are not new findings.

The localization EPS tests describe the implementation's deliberate 1e-12
tolerance. The remaining checks protect the registered strict inequalities.
All probes are CPU-only and use the repository's unchanged conftest.py.
"""
import math

import pytest

from test_audit13_analysis import AN


def record(**changes):
    rec = {
        "pi2": 0.0, "pi4": 0.3, "pi6": 0.5,
        "sigma6": 0.1, "i4": 0.3, "i6": 0.4,
    }
    return {**rec, **changes}


@pytest.mark.parametrize(
    "field,threshold,outward",
    [
        ("pi2", 0.10, 1),
        ("pi6", 0.30, -1),
        ("sigma6", 0.20, 1),
        ("i4", 0.20, -1),
        ("i6", 0.20, -1),
    ],
)
def test_localization_tolerance_accepts_half_eps_but_not_two_eps_outside(
    field, threshold, outward
):
    assert AN.EPS == 1e-12
    half_outside = threshold + outward * AN.EPS / 2
    twice_outside = threshold + outward * 2 * AN.EPS
    assert AN.conditions(record(**{field: threshold}))["all"] is True
    assert AN.conditions(record(**{field: half_outside}))["all"] is True
    assert AN.conditions(record(**{field: twice_outside}))["all"] is False


@pytest.mark.parametrize(
    "distance,expected", [(0.0, False), (0.5, False), (2.0, True)]
)
def test_strict_overturn_interaction_includes_the_eps_deadband(distance, expected):
    value = 0.05 - distance * AN.EPS
    outcome = AN.overturn_conditions(record(i4=value, i6=value))
    assert outcome["interaction_lt_0.05_at_cuts_4_and_6_additive"] is expected
    assert outcome["any"] is expected


@pytest.mark.parametrize("low_cut", ["i4", "i6"])
def test_interaction_overturn_still_requires_both_cuts_below_the_deadband(low_cut):
    outcome = AN.overturn_conditions(record(**{low_cut: 0.05 - 2 * AN.EPS}))
    assert outcome["interaction_lt_0.05_at_cuts_4_and_6_additive"] is False
    assert outcome["any"] is False


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("lo", 1.2, "mixed"),
        ("lo", math.nextafter(1.2, math.inf), "holds"),
        ("r", 3.0, "holds"),
        ("r", math.nextafter(3.0, -math.inf), "mixed"),
    ],
)
def test_i_rule_distinguishes_adjacent_floats_at_registered_boundaries(field, value, expected):
    ratios = {bank: {"r": 4.0, "lo": 2.0, "hi": 6.0} for bank in AN.BANKS}
    ratios["m3"][field] = value
    assert AN.read_i(ratios) == expected


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (0.0, 1.0, "neither"),
        (math.nextafter(0.0, math.inf), 1.0, "holds"),
        (-1.0, 0.0, "neither"),
        (-1.0, math.nextafter(0.0, -math.inf), "overturned"),
    ],
)
def test_iii_rule_is_strict_at_zero_without_a_localization_tolerance(lo, hi, expected):
    assert AN.read_iii(lo, hi) == expected
