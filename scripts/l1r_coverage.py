"""L1R coverage estimate (docs/preregistration.md, "L1R"): how much of the seed-0 stream's root-corrupted rows the balanced
L1 reassignment can cover, before any GPU run. Uses the real training windows (`TrainStream`, numpy stream seeded as the
run's), their canonical roots from the released tokenizer (CUDA if available, else CPU), the real mask stream (a CPU
`MechArm` of the same seed) and SIMULATED supplied roots: M2's level-0 draw (unigram without the canonical code) from the
arm's own CPU replacement generator (the GPU run's level >= 1 draws shift its stream, so the exact roots differ; the
run-time replay and gate are the binding check). Runs the same selection as the run (src/ncmr/l1r.py).

  python scripts/l1r_coverage.py [--steps 600] [--batch 256] [--seed 0] [--out results/l1r_coverage_cpu.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr import l1r as L1R  # noqa: E402
from ncmr import w2m as W2M  # noqa: E402
from ncmr.corruption import CorruptionCfg  # noqa: E402
from ncmr.data import TrainStream  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402
from ncmr.train import fp32_eval, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--coverage_min", type=float, default=0.80)
ap.add_argument("--data_dir", default="")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "l1r_coverage_cpu.json"))
a = ap.parse_args()

dev = pick_device()
BUNDLE = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"), device=dev)
tok = BUNDLE.vqvae.eval()
for p in tok.parameters():
    p.requires_grad_(False)
L0 = BUNDLE.ckpt_extra["inference"]["level0_unigram"].float().cpu()
K = len(tok.vq.levels)
# a CPU arm of the run's seed: its mask and replacement generators are the run's (the replacement values are simulated)
arm = W2M.MechArm("m2r", tok, None, L0, None, "cpu", a.seed, CorruptionCfg(mode="mech").rates(K), clean_frac=0.25, dry=True)
stream = TrainStream(a.data_dir) if a.data_dir else TrainStream()
rng = np.random.default_rng(a.seed)


@torch.no_grad()
def canon_fn(xb):
    with fp32_eval(tok, dev):
        return tok.vq.z_to_idx(tok._encode_pre_quant(xb.to(dev))).cpu()


t0 = time.time()
rec = L1R.replay(lambda: torch.from_numpy(stream.batch(rng, a.batch)), arm, a.steps, 1, out_vocab=None, canon_fn=canon_fn)
plan = L1R.build_plan(rec, a.seed)
res = {"what": "CPU estimate of the L1R coverage on the seed-0 data stream (real windows and masks, simulated replacement roots)",
       "steps": a.steps, "batch_size": a.batch, "seed": a.seed, "canonical_encode_device": dev,
       "coverage_min": a.coverage_min, "passes_gate": plan.coverage >= a.coverage_min, "wall_s": round(time.time() - t0, 1),
       **plan.summary()}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
