"""Notebook provenance (PR 3): the run cell records which code ran and refuses to resume under changed or unknown code.

The setup helpers and the run cell are executed from `notebooks/make_notebooks.py` itself; only `subprocess` (the
training command and `git diff`) and `torch` (GPU queries) are stubs.
"""
import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CMD = "python scripts/train_gen.py --config configs/scale/L_ts_ctrl.yaml --run_dir {R}/L_ts_ctrl_s{S} --seed {S}"


def cells():
    spec = importlib.util.spec_from_file_location("make_notebooks", ROOT / "notebooks/make_notebooks.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.SETUP[m.SETUP.index("utc = lambda"):], m.RUN


class FakeSubprocess:
    def __init__(self, diff_rc=0):
        self.diff_rc, self.commands = diff_rc, []

    def run(self, cmd, shell=False, **kw):
        if isinstance(cmd, list) and cmd[:2] == ["git", "diff"]:
            return SimpleNamespace(returncode=self.diff_rc)
        self.commands.append(cmd)
        return SimpleNamespace(returncode=0)


def run_cell(tmp_path, sha, sub):
    helpers, run = cells()
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    ns = {"os": os, "json": json, "time": time, "subprocess": sub, "torch": torch,
          "CMDS": [CMD], "SEEDS": [0], "R": str(tmp_path), "D": str(tmp_path), "GIT_SHA": sha}
    exec(helpers, ns)
    exec(run, ns)
    return ns


def records(rd):
    return [json.loads(line) for line in open(rd / "provenance.jsonl")]


def test_fresh_run_records_start_and_end(tmp_path):
    sub = FakeSubprocess()
    run_cell(tmp_path, "a" * 40, sub)
    assert len(sub.commands) == 1
    rec = records(tmp_path / "L_ts_ctrl_s0")
    assert [r["event"] for r in rec] == ["start", "end"]
    assert {r["git_sha"] for r in rec} == {"a" * 40}
    assert rec[0]["had_ckpt"] is False and rec[1]["exit"] == 0


def test_refuses_checkpoint_without_provenance(tmp_path):
    """Codex PR 3 comment 4365630299: a checkpoint from before provenance existed must not resume silently."""
    rd = tmp_path / "L_ts_ctrl_s0"
    rd.mkdir()
    (rd / "ckpt.pt").write_bytes(b"old")
    sub = FakeSubprocess()
    with pytest.raises(SystemExit, match="before provenance was recorded"):
        run_cell(tmp_path, "b" * 40, sub)
    assert sub.commands == []
    assert not (rd / "provenance.jsonl").exists()


def test_refuses_resume_when_code_changed(tmp_path):
    rd = tmp_path / "L_ts_ctrl_s0"
    rd.mkdir()
    (rd / "provenance.jsonl").write_text(json.dumps({"event": "start", "git_sha": "a" * 40}) + "\n")
    (rd / "ckpt.pt").write_bytes(b"old")
    sub = FakeSubprocess(diff_rc=1)
    with pytest.raises(SystemExit, match="has changed"):
        run_cell(tmp_path, "b" * 40, sub)
    assert sub.commands == []


def test_resumes_when_code_paths_unchanged(tmp_path):
    rd = tmp_path / "L_ts_ctrl_s0"
    rd.mkdir()
    (rd / "provenance.jsonl").write_text(json.dumps({"event": "start", "git_sha": "a" * 40}) + "\n")
    (rd / "ckpt.pt").write_bytes(b"old")
    sub = FakeSubprocess(diff_rc=0)
    run_cell(tmp_path, "b" * 40, sub)
    assert len(sub.commands) == 1
    assert records(rd)[1]["had_ckpt"] is True
