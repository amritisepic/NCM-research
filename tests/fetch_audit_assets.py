"""Explicit opt-in download of hash-pinned CPU test assets, only under tests/."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assets = json.loads((HERE / "audit_assets.json").read_text(encoding="utf-8"))
    dest = HERE / "_assets/sudoku-hard"
    dest.mkdir(parents=True, exist_ok=True)
    for name, spec in assets.items():
        path = dest / name
        if path.is_file() and digest(path) == spec["sha256"]:
            print(name, "verified")
            continue
        tmp = path.with_suffix(path.suffix + ".download")
        with urlopen(spec["url"], timeout=30) as response, tmp.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if digest(tmp) != spec["sha256"]:
            raise ValueError("asset digest mismatch: " + name)
        tmp.replace(path)
        print(name, "downloaded and verified")


if __name__ == "__main__":
    main()
