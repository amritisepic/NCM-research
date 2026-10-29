"""CPU pre-launch PD/PDa coverage gate: first twenty real seed-0 batches, W2M masks and released M2 proposals.

Run before launching PD notebooks. The report contains only asset digests and repo-relative config references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ncmr.corruption import CorruptionCfg
from ncmr.data import EOS, TrainStream
from ncmr.ncm_ref_loader import load_ncm
from ncmr.w2m import MechArm


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_coverage(report):
    if report.get("steps") != 20 or report.get("seed") != 0 or report.get("device") != "cpu":
        raise ValueError("PD requires the first twenty seed-0 steps checked on CPU")
    value = report["coverage"]["overall"]
    if value < 0.80:
        raise ValueError(f"PD coverage {value:.6f} is below 0.80: kill inconclusive, do not launch")
    return value


def chunk_frozen_forward(mech, forward_batch):
    """Bound CPU attention memory; keep each original 64-row root/fine sampling draw unchanged.

    Only the deterministic fp32 eval forward is partitioned. Logits are concatenated before the existing
    canonical exclusion, softmax and single draw_fine call for the complete registered micro-batch.
    """
    original = mech._logits

    def logits(model, fwd, canonical):
        return torch.cat([original(model, fwd, canonical[start:start + forward_batch])
                          for start in range(0, len(canonical), forward_batch)])

    mech._logits = logits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/pd_coverage.json")
    ap.add_argument("--data_dir", default=str(ROOT / "data/tinystories"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--forward_batch", type=int, default=8, help="CPU-only deterministic frozen-forward rows")
    ap.add_argument("--verify", action="store_true", help="check an existing CPU report without model work")
    a = ap.parse_args()
    if a.forward_batch <= 0:
        ap.error("forward_batch must be positive")
    cfg = OmegaConf.load(ROOT / "configs/scale/L_ts_pd_d.yaml")
    art, data = ROOT / "artifacts/tinystories", Path(a.data_dir)
    assets = {f"base/{name}": art / name for name in ("config.yaml", "generator.pt", "tokenizer.pt")}
    assets.update({f"data/{name}": data / name for name in ("train_bpe_8192.bin", "train_bpe_8192_offsets.npy")})
    missing = [name for name, path in assets.items() if not path.is_file()]
    if missing:
        sys.exit("PD CPU coverage not run: released assets absent: " + ", ".join(missing))
    identity = {"asset_sha256": {name: digest(path) for name, path in assets.items()},
                "config": OmegaConf.to_container(cfg, resolve=True), "micro_batch": 64,
                "cpu_forward_batch": a.forward_batch}
    if a.verify:
        report = json.loads(Path(a.out).read_text(encoding="utf-8"))
        if report["identity"] != identity:
            sys.exit("PD coverage report differs from current released assets/config")
        check_coverage(report)
        print("PD CPU coverage gate verified", flush=True)
        return
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    bundle = load_ncm(str(art), device="cpu")
    tok, net = bundle.vqvae.eval(), bundle.net.eval()
    prior = bundle.ckpt_extra["inference"]["level0_unigram"].float()
    prefix = lambda b: tok.encoder.wte(torch.full((b, 1), EOS, dtype=torch.long))
    m = MechArm("pd", tok, net, prior, prefix, "cpu", 0, CorruptionCfg(mode="mech").rates(len(tok.vq.levels)),
                frozen=MechArm.frozen_copy(net))
    chunk_frozen_forward(m, a.forward_batch)
    stream, rng, started = TrainStream(str(data)), np.random.default_rng(0), time.monotonic()
    for step in range(20):
        x = torch.from_numpy(stream.batch(rng, 256))
        m.prepare_batch(x, m.draw_plan(256), 4)  # notebook's required --micro_batch64
        print(json.dumps({"step": step + 1, "coverage": m.realized()["pd_coverage"]}), flush=True)
    report = {"registration": "PD/PDa", "identity": identity, "steps": 20, "seed": 0, "device": "cpu",
              "coverage": m.realized()["pd_coverage"], "wall_s": time.monotonic() - started}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(temporary, out)
    check_coverage(report)


if __name__ == "__main__":
    main()
