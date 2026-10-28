"""Audit 11: the E1b analysis must validate records against its pinned plan.

Only saved JSON inputs are needed; conftest supplies the CPU environment.
The expected failures demonstrate A11-1 at the audited commit.
"""
import copy
import hashlib
import importlib.util
import json

import pytest

from conftest import ROOT


PINS = {
    "results/colab/e1_equivalence/e1_ts__result.json":
        "5b134622a64121d9e835a8bae483c505768efbc63e68688a7e7ac5781942c5db",
    "results/colab/e1_equivalence/e1_wmt__result.json":
        "c66d29ed71fb17458413fb1ef683c54cb16413ca3b2d1bb8a131f14d2f6e478e",
    "results/colab/e1_equivalence/e1_sdk__result.json":
        "e936925fcb6935230f253ecc4220e12b33428dda3234db03ec8f1190b3df64e5",
    "results/colab/e1b_continue/e1b__result.json":
        "358ff27d7ccafa81bbe4d83467b8fcb9904d313aee9c2d1cc2d840c5d6792100",
}


@pytest.fixture(scope="module")
def saved():
    data = {}
    for path, digest in PINS.items():
        raw = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == digest, path
        data[path.rsplit("/", 1)[-1]] = json.loads(raw)
    return data


@pytest.fixture(scope="module")
def analyzer():
    spec = importlib.util.spec_from_file_location(
        "audit11_analyze_e1b", ROOT / "scripts/analyze_e1b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input(saved):
    result = copy.deepcopy(saved["e1b__result.json"])
    # Preserve the production byte-level pin on checkouts that translate CRLF.
    for model, filename in (("A0", "e1_ts__result.json"), ("J0", "e1_wmt__result.json")):
        result["selection"][model]["e1_sha256"] = hashlib.sha256(
            (ROOT / "results/colab/e1_equivalence" / filename).read_bytes()).hexdigest()
    return result


def _read(analyzer, result):
    return analyzer.analyse(result, str(ROOT / "results/colab/e1_equivalence"))


@pytest.mark.parametrize("mutation", ["outside_selection", "wrong_domain", "missing_selection", "negative_index"])
def test_refuses_records_outside_the_pinned_plan(analyzer, saved, mutation):
    result = _input(saved)
    if mutation == "outside_selection":
        extra = copy.deepcopy(result["items"][0])
        extra["item"] = 0  # J0 item 0 is neither censored nor wide.
        extra["e1"] = saved["e1_wmt__result.json"]["rows"]["J0"][0]["mc"]["8"]
        extra["final"] = [extra["e1"][0] + 1, extra["e1"][1] + 512]
        result["items"].append(extra)
    elif mutation == "wrong_domain":
        result["items"][0]["domain"] = "sudoku"
    elif mutation == "missing_selection":
        del result["selection"]["J0"]  # Currently skips the source digest check too.
    else:
        record = next(r for r in result["items"] if r["model"] == "J0" and r["item"] == 127)
        record["item"] = -1  # Python aliases this to item 127, while the report prints -1.
    with pytest.raises(ValueError):
        _read(analyzer, result)


def test_substitution_rejects_duplicate_item_identity(analyzer, saved):
    result = _input(saved)
    first = copy.deepcopy(result["items"][0])
    second = copy.deepcopy(first)
    second["e1"] = list(first["final"])
    rows = saved["e1_wmt__result.json"]["rows"]["J0"]
    with pytest.raises(ValueError):
        analyzer.substitute(rows, [first, second], "J0")


def _omit_a_still_censored_item(result):
    record = next(r for r in result["items"] if r["model"] == "J0" and r["final"][0] == 0)
    result["items"].remove(record)
    return record["model"], record["item"]


def test_missing_selected_record_must_be_listed_as_not_reached(analyzer, saved):
    result = _input(saved)
    _omit_a_still_censored_item(result)
    assert result["complete"] and result["not_reached"] == []
    with pytest.raises(ValueError):
        _read(analyzer, result)


def test_a_disclosed_partial_continuation_may_meet_the_registered_bar(analyzer, saved):
    """The registration permits this; complete=False alone must not veto the bar."""
    result = _input(saved)
    model, item = _omit_a_still_censored_item(result)
    result["complete"] = False
    result["not_reached"] = [[model, item]]
    out = _read(analyzer, result)
    assert out["verdict"]["e1_replication_bar_met"] is True
    assert out["verdict"]["run_complete"] is False
    assert out["verdict"]["items_not_reached"] == {model: [item]}


def test_refuses_counts_declared_to_use_an_unregistered_temperature(analyzer, saved):
    result = _input(saved)
    result["protocol"]["temperature"] = 0.5
    with pytest.raises(ValueError):
        _read(analyzer, result)
