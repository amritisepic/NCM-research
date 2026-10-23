"""CPU-only audit fixtures; no automatic downloads or training."""
from pathlib import Path
import hashlib
import json
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch.set_num_threads(4)


@pytest.fixture(scope="session")
def released_sudoku():
    from ncmr.ncm_ref_loader import load_ncm, safe_pickle_load

    # Prefer already installed assets; the audit downloads live only under tests.
    candidates = [
        (ROOT / "artifacts/sudoku/hard", ROOT / "data/sudoku/hard/puzzles.pickle"),
        (ROOT / "tests/_assets/sudoku-hard", ROOT / "tests/_assets/sudoku-hard/puzzles.pickle"),
    ]
    for ckpt, data in candidates:
        if data.is_file() and all((ckpt / f).is_file() for f in ("config.yaml", "tokenizer.pt", "generator.pt")):
            specs = json.loads((ROOT / "tests/audit_assets.json").read_text(encoding="utf-8"))
            for name, spec in specs.items():
                path = data if name == "puzzles.pickle" else ckpt / name
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                assert actual == spec["sha256"], "released asset digest mismatch: " + name
            bundle = load_ncm(str(ckpt), device="cpu")
            puzzles = safe_pickle_load(str(data))
            assert len(puzzles) >= 20
            # Always the first twenty released hard puzzles, in released order.
            x = torch.tensor([list(p)[:bundle.net.prefix_len] for p in puzzles[:20]], dtype=torch.long)
            return bundle, x
    pytest.fail("Released Sudoku-hard assets missing; see docs/reviews/codex_audit.md for setup.")
