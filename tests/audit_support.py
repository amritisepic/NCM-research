"""Read-only access to the immutable analysis implementation when absent locally."""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
AUDIT_COMMIT = "1c7185c0e0de23d364c8bad33b2606c852bbd00d"


def load_jacobi():
    path = ROOT / "src/ncmr/jacobi_train.py"
    if path.is_file():
        return importlib.import_module("ncmr.jacobi_train")
    # Do not copy the other ref over protected source files. Execute the exact Git
    # blob under its real module name so the requested function itself is tested.
    name = "ncmr.jacobi_train"
    if name not in sys.modules:
        source = subprocess.check_output(
            ["git", "show", f"{AUDIT_COMMIT}:src/ncmr/jacobi_train.py"], cwd=ROOT,
        ).decode("utf-8")
        module = types.ModuleType(name)
        module.__file__ = f"git/{AUDIT_COMMIT}/src/ncmr/jacobi_train.py"
        module.__package__ = "ncmr"
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        sys.modules[name] = module
    return sys.modules[name]
