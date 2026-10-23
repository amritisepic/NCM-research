"""Execute pinned audit subjects in memory; never replace protected source files."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import types

ROOT = Path(__file__).resolve().parents[1]
# The audit ran these tests against b397656 (strict xfails = demonstrated defects). After the R028 fixes they run
# against the working tree as regression tests; set AUDIT2_REF to a commit to re-run them against that object.
AUDIT_REF = os.environ.get("AUDIT2_REF", "")


def source(path):
    if not AUDIT_REF:
        return (ROOT / path).read_text(encoding="utf-8")
    return subprocess.check_output(
        ["git", "show", f"{AUDIT_REF}:{path}"], cwd=ROOT,
    ).decode("utf-8")


def definitions(path, names, namespace):
    tree = ast.parse(source(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"git/{AUDIT_REF or 'worktree'}/{path}", "exec"), namespace)
    return namespace


def jacobi_module():
    module = types.ModuleType("audit2_jacobi")
    exec(compile(source("src/ncmr/jacobi_train.py"), "audit2_jacobi", "exec"), module.__dict__)
    return module
