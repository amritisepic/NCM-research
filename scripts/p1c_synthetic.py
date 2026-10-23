"""P1c step 1: population TC calibration of the existing two-stream probes on CPU.

One modeled level has four binary codes; all coarser conditioning is constant.
Enumerating its 16 possible strings gives exact training AND evaluation population
expectations (no train/test estimation noise or finite-sample entropy correction).
The existing TwoStreamLevelModel, upstream transformer and codebook-tied heads are
used unchanged. This does not implement P1c step 2's proposed side channel.

python -B scripts/p1c_synthetic.py --out docs/reviews/p1c_synthetic.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from ncmr.models import TwoStreamLevelModel, build_ncm_generator, build_tokenizer, n_params


def population(case, length=4):
    """Exact stationary binary distributions, with natural-log entropy formulas."""
    codes = torch.tensor(list(itertools.product((0, 1), repeat=length)), dtype=torch.long)
    if case == "independent":
        prob = torch.full((len(codes),), 2.0 ** -length, dtype=torch.float64)
        joint_entropy = length * math.log(2)
    elif case == "duplicated":
        prob = (codes == codes[:, :1]).all(1).double() / 2
        joint_entropy = math.log(2)
    else:
        flip = {"markov_0.10": 0.10, "markov_0.25": 0.25}[case]
        changes = (codes[:, 1:] != codes[:, :-1]).sum(1)
        prob = 0.5 * flip ** changes.double() * (1 - flip) ** (length - 1 - changes.double())
        h = -flip * math.log(flip) - (1 - flip) * math.log1p(-flip)
        joint_entropy = math.log(2) + (length - 1) * h
    return codes, prob, dict(fact_nll=length * math.log(2), ar_nll=joint_entropy,
                             tc=length * math.log(2) - joint_entropy)


def make_pair(seed, length=4):
    torch.manual_seed(seed)
    tok = build_tokenizer(embed_dim=8, codebook=2, levels=[1, length], in_vocab=4,
                          context=length, n_heads=2, dropout=0, fine_drop=0)
    tok.eval().requires_grad_(False)
    base = build_ncm_generator(tok, n_layer=1, n_head=2, n_embd=32, dropout=0, prefix_len=1)
    # Same body/head initialization; the actual AR probe retains its normal prev_emb init.
    fact = TwoStreamLevelModel(copy.deepcopy(base), "fact")
    ar = TwoStreamLevelModel(copy.deepcopy(base), "ar")
    # Match every common tensor, including the unused fact prev_emb, for provenance.
    ar.load_state_dict({k: v for k, v in fact.state_dict().items() if k != "mask"}, strict=False)
    return fact, ar


def population_nll(model, z, codes, weights):
    logits = model(z, codes)
    by_position = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes.reshape(-1), reduction="none")
    by_string = by_position.reshape(codes.shape).sum(1)
    return (by_string.double() * weights).sum()


def fit(model, codes, weights, steps, lr):
    z = torch.zeros(len(codes), 1 + codes.shape[1], 8)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    model.train()
    with torch.no_grad():
        initial = population_nll(model, z, codes, weights).item()
    curve = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = population_nll(model, z, codes, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for group in opt.param_groups:
            group["lr"] = lr * (0.1 + 0.9 * (1 + math.cos(math.pi * step / steps)) / 2)
        opt.step()
        if step + 1 in {steps // 2, steps}:
            with torch.no_grad():
                curve.append({"step": step + 1, "population_nll": population_nll(model, z, codes, weights).item()})
    model.eval()
    with torch.no_grad():
        final = population_nll(model, z, codes, weights).item()
    return dict(initial_nll=initial, final_nll=final, curve=curve, parameters=n_params(model))


def run(steps=800, seeds=(0, 1, 2), lr=0.01, tolerance=0.05):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cases = []
    for case in ("independent", "duplicated", "markov_0.10", "markov_0.25"):
        codes, weights, truth = population(case)
        rows = []
        for seed in seeds:
            fact, ar = make_pair(seed)
            f = fit(fact, codes, weights, steps, lr)
            a = fit(ar, codes, weights, steps, lr)
            estimate = f["final_nll"] - a["final_nll"]
            row = dict(seed=seed, fact=f, ar=a, estimated_tc=estimate, error=estimate - truth["tc"])
            rows.append(row)
            print(json.dumps(dict(case=case, seed=seed, estimated_tc=estimate, true_tc=truth["tc"])), flush=True)
        values = torch.tensor([r["estimated_tc"] for r in rows], dtype=torch.float64)
        cases.append(dict(case=case, true=truth, runs=rows, estimated_tc_mean=values.mean().item(),
                          estimated_tc_sd=values.std(correction=1).item() if len(rows) > 1 else None,
                          max_absolute_error=max(abs(r["error"]) for r in rows),
                          passed=all(abs(r["error"]) <= tolerance for r in rows)))
    models_source = (ROOT / "src/ncmr/models.py").read_bytes()
    return dict(protocol="P1c step 1; existing fact/ar masks; no step-2 side channel", device="cpu",
                dtype="float32 (loss expectation accumulated in float64)", torch=torch.__version__,
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                models_sha256=hashlib.sha256(models_source).hexdigest(),
                configuration=dict(levels=[1, 4], modeled_levels=[4], codebook=2, vq_dim=8, n_embd=32,
                                   n_layer=1, n_head=2, dropout=0, steps=steps, lr=lr, weight_decay=0,
                                   seeds=list(seeds), threads=2, tolerance_nats=tolerance,
                                   initialization="identical common weights; AR prev_emb normal; distinct masks"),
                units="nats per four-code modeled level, conditional on constant coarser input",
                evaluation="exact population expectation over all 16 strings; same population objective in training",
                formulas=dict(independent="TC=0", duplicated="TC=(L-1)*ln(2)",
                              markov="stationary fair first bit, P(flip)=q; TC=(L-1)*(ln(2)-h(q))"),
                cases=cases, passed=all(c["passed"] for c in cases), wall_s=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/reviews/p1c_synthetic.json")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    result = run(args.steps, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
