"""Audit 10: E1b must keep its registered cap and inputs across sessions."""
import hashlib
import importlib.util
import json

import pytest

from conftest import ROOT

SPEC = importlib.util.spec_from_file_location("audit10_e1b", ROOT / "scripts/e1b_continue.py")
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class Clock:
    def __init__(self):
        self.seconds = 0.0

    def __call__(self):
        return self.seconds


def _plan():
    plan = [{"domain": "wmt14", "model": "J0", "item": i, "e1": [0, 2048], "kind": "censored"}
            for i in range(3)]
    selection = {"J0": {"domain": "wmt14", "delta": 0.05 / 384, "n_items": 128,
                         "censored": [0, 1, 2], "wide": [], "e1_file": "e1_wmt__result.json"}}
    return plan, selection


def _record(entry):
    return {"replay_hits": entry["e1"][0], "replay_n": entry["e1"][1], "replay_match": True,
            "final": list(entry["e1"]), "stop": "cap"}


def test_interrupted_session_time_is_charged_before_another_item_can_start(tmp_path):
    plan, selection = _plan()
    first_clock = Clock()

    def first_runner(domain):
        def one(entry):
            if entry["item"] == 0:
                first_clock.seconds += 149.0 * 60.0
                return _record(entry)
            first_clock.seconds += 5.0 * 60.0
            raise KeyboardInterrupt
        return one

    with pytest.raises(KeyboardInterrupt):
        B.run(plan, selection, first_runner, str(tmp_path), max_minutes=150.0, clock=first_clock)

    resumed = []

    def second_runner(domain):
        def one(entry):
            resumed.append(entry["item"])
            return _record(entry)
        return one

    result = B.run(plan, selection, second_runner, str(tmp_path), max_minutes=150.0, clock=Clock())
    assert resumed == [], "the previous session already consumed 154 minutes of the 150-minute cumulative cap"
    assert result["minutes"] >= 154.0
    assert result["not_reached"] == [["J0", 1], ["J0", 2]]


@pytest.mark.parametrize("field", ["torch", "gpu"])
def test_resume_refuses_a_changed_runtime_before_drawing_another_item(tmp_path, field):
    plan, selection = _plan()
    before = {"torch": "runtime-a", "gpu": "device-a"}
    B.run(plan, selection, lambda domain: _record, str(tmp_path), max_items=1, clock=Clock(), extra=before)
    after = dict(before)
    after[field] += "-changed"
    resumed = []

    def runner(domain):
        def one(entry):
            resumed.append(entry["item"])
            return _record(entry)
        return one

    refused = False
    try:
        B.run(plan, selection, runner, str(tmp_path), max_items=1, clock=Clock(), extra=after)
    except SystemExit:
        refused = True
    assert refused, "an E1b run must pin its own runtime even when a replay mismatch versus E1 is permitted"
    assert resumed == []


def test_resume_refuses_changed_e1_floor_inputs_even_when_selection_is_unchanged(tmp_path):
    source = ROOT / "results/colab/e1_equivalence/e1_wmt__result.json"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "c66d29ed71fb17458413fb1ef683c54cb16413ca3b2d1bb8a131f14d2f6e478e":
        pytest.fail("saved E1 input digest differs from the audited commit")
    e1_dir = tmp_path / "e1"
    e1_dir.mkdir()
    local_source = e1_dir / source.name
    local_source.write_bytes(raw)
    run_dir = tmp_path / "continuation"
    plan, selection = B.build_plan(str(e1_dir), only=("J0",))
    B.run(plan, selection, lambda domain: _record, str(run_dir), max_items=1, clock=Clock())

    changed = json.loads(raw)
    # The replay and the confidence-sequence floor both refer to these source rows.
    # This changes N_8 without changing any hit counts or the selected item set.
    changed["rows"]["J0"][plan[0]["item"]]["level_nll"][8] += 1.0
    local_source.write_text(json.dumps(changed), encoding="utf-8")
    plan2, selection2 = B.build_plan(str(e1_dir), only=("J0",))
    assert [(e["model"], e["item"]) for e in plan2] == [(e["model"], e["item"]) for e in plan]

    refused = False
    try:
        B.run(plan2, selection2, lambda domain: _record, str(run_dir), max_items=1, clock=Clock())
    except SystemExit:
        refused = True
    assert refused, "the unchanged selection must not allow the saved E1 inputs to be replaced during continuation"
