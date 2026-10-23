"""Execute pinned audit subjects in memory; never replace protected source files."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import types

ROOT = Path(__file__).resolve().parents[1]
AUDIT_REF = "b3976566a998f2362b82c665780f7573f2606b0c"


def source(path):
    return subprocess.check_output(
        ["git", "show", f"{AUDIT_REF}:{path}"], cwd=ROOT,
    ).decode("utf-8")


def definitions(path, names, namespace):
    tree = ast.parse(source(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"git/{AUDIT_REF}/{path}", "exec"), namespace)
    return namespace


def jacobi_module():
    module = types.ModuleType("audit2_jacobi")
    exec(compile(source("src/ncmr/jacobi_train.py"), "audit2_jacobi", "exec"), module.__dict__)
    return module
