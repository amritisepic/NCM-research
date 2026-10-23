"""Tests for the Codex review gate: the push/merge hook and codex_pr.py's gate logic (no network)."""
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "review_gate.sh"
RUN = "claude/ncm-research"
BASH = shutil.which("bash")


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _commit(repo, path, text="x\n"):
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _git(repo, "add", path)
    _git(repo, "commit", "-qm", f"add {path}")


@pytest.fixture
def repo(tmp_path):
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(tmp_path, "init", "-q", "-b", RUN, str(work))
    for k, v in (("user.name", "t"), ("user.email", "t@example.invalid")):
        _git(work, "config", k, v)
    _commit(work, "README.md")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "origin", RUN)
    return work


def hook(command, cwd):
    data = json.dumps({"tool_input": {"command": command}, "cwd": str(cwd)})
    return subprocess.run([BASH, str(HOOK)], input=data, capture_output=True, text=True,
                          env={**os.environ, "CLAUDE_PROJECT_DIR": str(cwd)}, timeout=60)


needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")


@needs_bash
def test_hook_blocks_bare_push_of_code_on_run_branch(repo):
    _commit(repo, "src/x.py")
    r = hook("git push", repo)
    assert r.returncode == 2 and "src/x.py" in r.stderr


@needs_bash
def test_hook_allows_docs_push_to_run_branch(repo):
    _commit(repo, "docs/note.md")
    assert hook(f"git push origin {RUN}", repo).returncode == 0


@needs_bash
def test_hook_allows_code_push_to_wip_branch(repo):
    _git(repo, "checkout", "-qb", "claude/wip-x")
    _commit(repo, "scripts/y.py")
    assert hook("git push -u origin claude/wip-x", repo).returncode == 0


@needs_bash
def test_hook_blocks_refspec_push_onto_run_branch(repo):
    _git(repo, "checkout", "-qb", "claude/wip-x")
    _commit(repo, "configs/a.yaml")
    assert hook(f"git push origin HEAD:{RUN}", repo).returncode == 2
    assert hook(f"git push origin claude/wip-x:refs/heads/{RUN}", repo).returncode == 2


@needs_bash
def test_hook_follows_cd_and_dash_c(repo, tmp_path):
    _commit(repo, "notebooks/n.ipynb")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert hook(f"cd {repo.as_posix()} && git push", elsewhere).returncode == 2
    assert hook(f"git -C {repo.as_posix()} push origin {RUN}", elsewhere).returncode == 2


@needs_bash
def test_hook_blocks_direct_merges(repo):
    assert hook("gh pr merge 3 --merge", repo).returncode == 2
    assert hook("gh api -X PUT repos/o/r/pulls/3/merge", repo).returncode == 2
    assert hook("python scripts/review/codex_pr.py merge 3", repo).returncode == 0


# --- codex_pr.py gate logic with a fake GitHub ---------------------------------------------------

def _load_codex_pr():
    spec = importlib.util.spec_from_file_location("codex_pr", ROOT / "scripts" / "review" / "codex_pr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEAD, T_HEAD, T_AFTER = "a" * 40, "2026-10-23T10:00:00Z", "2026-10-23T11:00:00Z"
BOT_USER = {"login": "chatgpt-codex-connector[bot]"}


def fake_github(mod, monkeypatch, reviews=(), rcomments=(), icomments=(), triage=None, files=("src/a.py",)):
    def gh(path, method="GET", data=None, paginate=False):
        if path.endswith("/pulls/7"):
            return {"head": {"sha": HEAD}, "base": {"ref": RUN}}
        if "/commits/" in path:
            return {"commit": {"committer": {"date": T_HEAD}}}
        if "/files" in path:
            return [{"filename": f} for f in files]
        if "/reviews" in path:
            return list(reviews)
        if "/pulls/7/comments" in path:
            return list(rcomments)
        if "/issues/7/comments" in path:
            return list(icomments)
        raise AssertionError(path)

    def run(cmd, **kw):
        return SimpleNamespace(returncode=1 if triage is None else 0, stdout=triage or "")

    monkeypatch.setattr(mod, "gh", gh)
    monkeypatch.setattr(mod.subprocess, "run", run)


def icomment(body):
    return {"id": 1, "user": BOT_USER, "created_at": T_AFTER, "body": body}


def test_gate_rejects_usage_limit_notice_as_review(monkeypatch):
    mod = _load_codex_pr()
    fake_github(mod, monkeypatch, icomments=[icomment("You have reached your Codex usage limits for code reviews.")])
    assert not mod.gate(7, verbose=False)


def test_gate_rejects_unrecognised_codex_comment(monkeypatch):
    mod = _load_codex_pr()
    fake_github(mod, monkeypatch, icomments=[icomment("To use Codex here, create an environment for this repo.")])
    assert not mod.gate(7, verbose=False)


def test_gate_accepts_clean_codex_comment(monkeypatch):
    mod = _load_codex_pr()
    fake_github(mod, monkeypatch, icomments=[icomment("Codex Review: Didn't find any major issues. Nice work!")])
    assert mod.gate(7, verbose=False) == HEAD


def test_gate_needs_verdict_next_to_each_p1_id(monkeypatch):
    mod = _load_codex_pr()
    review = {"id": 11, "user": BOT_USER, "commit_id": HEAD, "submitted_at": T_AFTER, "body": "Codex Review"}
    p1 = {"id": 1234567890, "user": BOT_USER, "commit_id": HEAD, "path": "src/a.py", "line": 3,
          "body": "![P1 Badge](https://img.shields.io/badge/P1-orange) Off-by-one in x[0][1]"}
    fake_github(mod, monkeypatch, reviews=[review], rcomments=[p1], triage="| 1234567890 | off-by-one | todo |")
    assert not mod.gate(7, verbose=False)
    fake_github(mod, monkeypatch, reviews=[review], rcomments=[p1],
                triage="| 1234567890 | off-by-one | **accept**: reproduced | fixed in abc |")
    assert mod.gate(7, verbose=False) == HEAD


def test_gate_skips_review_for_ungated_paths(monkeypatch):
    mod = _load_codex_pr()
    fake_github(mod, monkeypatch, files=("docs/x.md", "scripts/review/codex_pr.py"))
    assert mod.gate(7, verbose=False) == HEAD


def test_paginate_survives_brackets_in_bodies(monkeypatch):
    mod = _load_codex_pr()
    pages = {1: [{"body": "x[0][1]"}] * 100, 2: [{"body": "]["}] * 5}

    def run(cmd, **kw):
        page = int(cmd[-1].rsplit("page=", 1)[1])
        return SimpleNamespace(returncode=0, stdout=json.dumps(pages[page]), stderr="")

    monkeypatch.setattr(mod.subprocess, "run", run)
    assert len(mod.gh("repos/o/r/issues/1/comments?per_page=100", paginate=True)) == 105
