"""Adversarial on-policy diagnostics; no production changes or automatic training.

Default pytest checks the experimental matching claim. Opt-in inference-only
experiment: python -B tests/test_adv_onpolicy_mechanism.py --run --weights DIR.
DIR contains the two optional saved Sudoku experiment state dictionaries.
Only the requested Markdown report is written, with the complete numeric record.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import released_sudoku
from ncmr.jacobi_train import parallel_logits
from ncmr.sudoku_data import SudokuTrain

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "1a496e2"
RESULT_REF = "416cb82"
REPORT = ROOT / "docs/reviews/codex_adversarial_onpolicy.md"
MARKER = "<!-- ADV_ONPOLICY_NUMBERS -->"


def git_text(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT).decode("utf-8")


def subject_loss():
    path = ROOT / "scripts/onpolicy_repro.py"
    source = path.read_text(encoding="utf-8") if path.exists() else git_text(SUBJECT, "scripts/onpolicy_repro.py")
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "loss_fn"]
    assert len(nodes) == 1
    return ast.Module(body=nodes, type_ignores=[])


def rng_after_subject_policy(arm):
    """Exercise the real loss function through its sampling branch, with small fakes."""
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(0.1)

        def build_l0_query_input(self, batch, dtype):
            return torch.zeros(batch, 1, 1, dtype=dtype)

        def forward(self, x):
            return self.drop(torch.ones(x.shape[0], 4, 16))

    net = Net().train()
    c = torch.zeros(2, 4, dtype=torch.long)
    seen = []

    def hierarchy(*args, **kwargs):
        seen.append(torch.get_rng_state().clone())
        return {"inputs": torch.zeros(2, 3, 1), "targets": c,
                "canonical": c, "corrupted_frac_per_level": [0.5]}

    ns = dict(torch=torch, F=F, a=SimpleNamespace(arm=arm), net=net, frozen=None,
              tok=SimpleNamespace(encoder=SimpleNamespace(wte=lambda p: torch.zeros(2, 1, 1)),
                                  vq=SimpleNamespace(vqvae_vocab_sizes=[16])),
              canon=lambda s: c, parallel_logits=lambda *args: torch.zeros(2, 4, 16),
              rep_gen=torch.Generator().manual_seed(1), gen=torch.Generator().manual_seed(0),
              corr=None, build_hierarchy=hierarchy,
              per_level=lambda t: [float(t.float().mean())])
    exec(compile(subject_loss(), "scripts/onpolicy_repro.py", "exec"), ns)
    torch.manual_seed(123)
    ns["loss_fn"](c, c)
    return seen[0]


# fixed in PR 6 (was a strict xfail: ADV-1: policy sampling changes training dropout RNG)
def test_sudoku_wrong_arms_share_dropout_rng():
    assert torch.equal(rng_after_subject_policy("uniform_wrong"),
                       rng_after_subject_policy("plausible_wrong"))


def test_exclusion_changes_location_distribution():
    # Forced-wrong sampling weights these locations equally. Natural policy
    # errors weight the second location 80 times more than the first.
    p_correct = torch.tensor([0.99, 0.2], dtype=torch.float64)
    natural_location = (1 - p_correct) / (1 - p_correct).sum()
    assert natural_location[1] / natural_location[0] == pytest.approx(80)
    assert not torch.allclose(natural_location, torch.tensor([0.5, 0.5], dtype=torch.float64))


def original_numbers():
    record = {}
    names = ["uniform_wrong_t1_s0", "plausible_wrong_t1_s0",
             "plausible_wrong_requant_t1_s0", "onpol_sample_s0"]
    baselines = []
    for name in names:
        d = json.loads(git_text(RESULT_REF, f"results/onpolicy_repro/{name}.json"))
        rows = []
        for ev in d["evals"]:
            rows.append({k: ev[k] for k in ("step", "nll", "solved_pct", "solved_t1_pct") if k in ev})
            rows[-1]["entropy_nats_per_puzzle"] = sum(a * b for a, b in
                zip(d["levels"], ev.get("entropy_per_code_per_level", [])))
        baselines.append({k: v for k, v in d["evals"][0].items() if k not in ("wall_s", "solved_t1", "solved_t1_pct")})
        record[name] = {"args": d["args"], "evals": rows}
    record["shared_step0_greedy_nll_roundtrip_entropy"] = all(x == baselines[0] for x in baselines)
    w1 = json.loads(git_text(RESULT_REF, "results/w1_analysis_s0.json"))["arms"]
    record["w1"] = {a: {k: w1[a][k] for k in ("genppl", "genppl_se", "val_nll", "mauve")}
                    for a in w1}
    record["A4_fraction_A3_excess_PPL_removed"] = ((w1["A3"]["genppl"] - w1["A4"]["genppl"]) /
        (w1["A3"]["genppl"] - w1["A1"]["genppl"]))
    return record


def mean_se(values):
    x = torch.as_tensor(values, dtype=torch.float64)
    return {"mean": float(x.mean()), "se_across_puzzles": float(x.std(correction=1) / len(x)**0.5)}


def run(weights=None, n=16, cpu_seconds=500):
    """Directly test coarse reliance using fixed counterfactuals for all models.

    The predeclared falsifier of the Sudoku *discount* interpretation is stronger
    plausible-arm reliance than uniform-arm reliance, at source level 6, in
    both downstream target-NLL advantage and scale-normalized logit response.
    A null/mixed result is inconclusive. This cannot falsify the text mechanism.
    """
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0, cpu0 = time.perf_counter(), time.process_time()
    bundle, _ = released_sudoku.__wrapped__()
    tok, net = bundle.vqvae.eval(), bundle.net.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    for p in net.parameters():
        p.requires_grad_(False)
    levels = [int(x) for x in tok.vq.levels]
    bounds = [0, *np.cumsum(levels).tolist()]
    data_path = ROOT / "assets/sudoku_train/hard.npz"
    data = SudokuTrain("hard", str(data_path))
    prompts, sols = (torch.from_numpy(x) for x in data.val(n))
    seeds, sources = [0, 1, 2], [6]

    def check_budget():
        if time.process_time() - cpu0 > cpu_seconds:
            raise RuntimeError("CPU experiment budget exceeded")

    @torch.no_grad()
    def logits(codes):
        out = []
        for i in range(0, n, 4):
            check_budget()
            out.append(parallel_logits(net, tok.encoder.wte(prompts[i:i+4]), codes[i:i+4]).float())
        return torch.cat(out)

    with torch.no_grad():
        codes = torch.cat([tok.encode(sols[i:i+4]) for i in range(0, n, 4)])
        base = logits(codes)
    probabilities = base.softmax(-1)
    logp = base.log_softmax(-1)
    p_correct = probabilities.gather(-1, codes[..., None]).squeeze(-1)
    exclusion = base.clone().scatter(-1, codes[..., None], float("-inf")).softmax(-1)
    rates = torch.cat([torch.full((lk,), 0.75 - 0.5*k/(len(levels)-2) if k < len(levels)-1 else 0)
                       for k, lk in enumerate(levels)])
    natural_rate = float(((1-p_correct) * rates).mean())
    forced_rate = float(rates.mean())
    wrong_mass_per_level = []
    for k in range(len(levels)):
        lo, hi = bounds[k:k+2]
        e = exclusion[:, lo:hi]
        pp = probabilities[:, lo:hi]
        lp = logp[:, lo:hi]
        wrong_mass_per_level.append({"level": k, "p_canonical_mean": float(p_correct[:, lo:hi].mean()),
            "forced_wrong_expected_original_policy_nll": float(-(e * lp).sum(-1).mean()),
            "forced_wrong_mass_with_original_p_below_0_001": float((e * (pp < .001)).sum(-1).mean()),
            "natural_policy_error_rate": float((1-p_correct[:, lo:hi]).mean())})
    forced_nll = float((-(exclusion*logp).sum(-1)*rates).sum() / (n*rates.sum()))
    natural_error_nll = float((-(probabilities*(1-F.one_hot(codes, base.shape[-1]))*logp).sum(-1)*rates).sum() /
                              (((1-p_correct)*rates).sum()))
    # All counterfactuals are constructed once using the *released* policy.
    # Each arm receives the same code arrays, not its own changed policy.
    interventions = {}
    changed_rates = {}
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        wrong = torch.multinomial(exclusion.flatten(0, 1), 1, generator=g).view(codes.shape)
        mask = torch.rand(codes.shape, generator=torch.Generator().manual_seed(seed + 104729)) < rates
        for k in sources:
            counter = codes.clone()
            sl = slice(bounds[k], bounds[k+1])
            counter[:, sl] = torch.where(mask[:, sl], wrong[:, sl], codes[:, sl])
            interventions[(seed, k)] = counter
            changed_rates[f"{seed}:{k}"] = float(mask[:, sl].float().mean())
    checkpoint_paths = {"released": None}
    if weights is not None:
        for arm in ("uniform_wrong", "plausible_wrong"):
            checkpoint_paths[arm] = Path(weights) / f"{arm}_s0.pt"
    models, raw = {}, {}
    for arm, path in checkpoint_paths.items():
        digest = None
        if path is not None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        net.eval()
        clean = logits(codes)
        ce = F.cross_entropy(clean.flatten(0, 1), codes.flatten(), reduction="none").view(codes.shape)
        entropy = -(clean.log_softmax(-1)*clean.softmax(-1)).sum(-1)
        centered = clean - clean.mean(-1, keepdim=True)
        rows = []
        per_source = {}
        for k in sources:
            lo = bounds[k+1]
            seed_gaps, seed_kl, seed_response = [], [], []
            for seed in seeds:
                pert = logits(interventions[(seed, k)])
                pert_ce = F.cross_entropy(pert.flatten(0, 1), codes.flatten(), reduction="none").view(codes.shape)
                gap = (pert_ce[:, lo:] - ce[:, lo:]).sum(1)
                kl = (clean[:, lo:].softmax(-1) *
                      (clean[:, lo:].log_softmax(-1)-pert[:, lo:].log_softmax(-1))).sum(-1).mean(1)
                delta = ((pert-pert.mean(-1, keepdim=True))-centered)[:, lo:]
                # Quotient is invariant to a common logit multiplier per model.
                response = delta.square().mean((1, 2)) / centered[:, lo:].square().mean((1, 2))
                seed_gaps.append(gap)
                seed_kl.append(kl)
                seed_response.append(response)
                rows.append({"seed": seed, "source_level": k,
                             "target_nll_advantage": mean_se(gap), "output_KL_per_code": mean_se(kl),
                             "scale_normalized_response": mean_se(response)})
            vectors = {"target_nll_advantage": torch.stack(seed_gaps).mean(0),
                       "output_KL_per_code": torch.stack(seed_kl).mean(0),
                       "scale_normalized_response": torch.stack(seed_response).mean(0)}
            raw[(arm, k)] = vectors
            per_source[str(k)] = {key: mean_se(v) for key, v in vectors.items()}
        models[arm] = {"weights_sha256": digest, "clean_nll": mean_se(ce.sum(1)),
                       "clean_entropy": mean_se(entropy.sum(1)), "by_source": per_source,
                       "by_seed_and_source": rows}
        print(json.dumps({"arm": arm, "clean_nll": models[arm]["clean_nll"],
                          "by_source": per_source, "cpu_s": time.process_time()-cpu0}), flush=True)
    contrasts = {}
    if "plausible_wrong" in models:
        for k in sources:
            contrasts[str(k)] = {key: mean_se(raw[("plausible_wrong", k)][key]-raw[("uniform_wrong", k)][key])
                                 for key in raw[("plausible_wrong", k)]}
    record = {"protocol": "inference-only; fixed released-policy wrong codes; clean generated validation tail",
              "torch": torch.__version__, "precision": "CPU fp32; TF32 off; eval mode", "threads": 1,
              "n_puzzles": n, "counterfactual_seeds": seeds, "source_levels": sources,
              "mask_seed_offset": 104729, "counterfactual_changed_fraction_by_seed_source": changed_rates,
              "original_training_seed": 0, "training_steps_executed": 0,
              "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "released_asset_hashes": json.loads((ROOT / "tests/audit_assets.json").read_text()),
              "natural_expected_changed_fraction_all_slots": natural_rate,
              "forced_wrong_expected_changed_fraction_all_slots": forced_rate,
              "forced_wrong_original_policy_nll": forced_nll,
              "natural_error_original_policy_nll": natural_error_nll,
              "exclusion_by_level": wrong_mass_per_level, "models": models,
              "paired_plausible_minus_uniform": contrasts,
              "cpu_s": time.process_time()-cpu0, "wall_s": time.perf_counter()-t0,
              "original_recomputation": original_numbers()}
    body = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Adversarial on-policy analysis\n"
    body = body.split(MARKER)[0].rstrip()
    REPORT.write_text(body + "\n\n" + MARKER + "\n\n## Complete numeric record\n\n```json\n" +
                      json.dumps(record, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"cpu_s": record["cpu_s"], "wall_s": record["wall_s"],
                      "paired_plausible_minus_uniform": contrasts}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cpu_seconds", type=float, default=500)
    args = ap.parse_args()
    if args.run:
        run(args.weights, args.n, args.cpu_seconds)
