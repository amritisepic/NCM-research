"""Bounded diagnostic, opt in: python -B tests/test_audit3_cpu.py --run.

Uses the hash-pinned released model and generated Sudoku train/validation data.
No checkpoints or production files are written; results join the audit JSON.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from conftest import released_sudoku
from audit2_support import definitions
from ncmr.corruption import CorruptionCfg, build_hierarchy
from ncmr.jacobi_train import parallel_logits
from ncmr.sudoku import cells, is_solution
from ncmr.sudoku_data import SudokuTrain
from ncmr.train import lr_at

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/reviews/codex_audit3_numbers.json"


def run(steps=200, cpu_seconds=1050):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started, cpu0 = time.perf_counter(), time.process_time()
    bundle, _ = released_sudoku.__wrapped__()
    tok = bundle.vqvae.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    path = ROOT / "assets/sudoku_train/hard.npz"
    data = SudokuTrain("hard", str(path))
    vp, vs = (torch.from_numpy(x) for x in data.val(32))
    nets = {"uniform": bundle.net, "onpolicy": copy.deepcopy(bundle.net)}
    states = {}
    for name, net in nets.items():
        net.vqvae_local = [tok]
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.01},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ], lr=5e-5, betas=(0.9, 0.99))
        ns = definitions("scripts/train_sudoku_L.py", ["wte", "canon", "tf_loss"], dict(
            torch=torch, F=F, tok=tok, net=net, dev="cpu", parallel_logits=parallel_logits,
            build_hierarchy=build_hierarchy, corr=CorruptionCfg(mode=name),
            gen=torch.Generator().manual_seed(0)))
        states[name] = dict(net=net, opt=opt, params=params, ns=ns, curve=[])

    @torch.no_grad()
    def evaluate(state, step):
        net = state["net"].eval()
        nll, solved = 0.0, []
        for i in range(0, len(vp), 2):
            prompt, sol = vp[i:i+2], vs[i:i+2]
            idx = tok.encode(sol)
            logits = parallel_logits(net, tok.encoder.wte(prompt), idx).float()
            nll += F.cross_entropy(logits.flatten(0, 1), idx.flatten(), reduction="sum").item()
            codes = net.generate(None, prompt, temperature=0.0, top_k=0, top_p=1.0,
                                 cfg_scale=0.0, cfg_null_eos=11)
            ids = tok.decoder(tok.vq.idx_to_z_q(codes)).argmax(-1)
            for out, puz in zip(ids.tolist(), prompt.tolist()):
                board = cells(out)
                solved.append(board is not None and is_solution(board, cells(puz)))
        state["curve"].append(dict(step=step, clean_nll=nll / len(vp),
                                   solved=sum(solved), n=len(solved),
                                   solve_pct=100 * sum(solved) / len(solved),
                                   per_puzzle_solved=solved))
        net.train()

    evaluate(states["uniform"], 0)
    states["onpolicy"]["curve"] = copy.deepcopy(states["uniform"]["curve"])
    print("baseline", states["uniform"]["curve"][-1], flush=True)
    rng = np.random.default_rng(0)
    finished = 0
    for step in range(steps):
        if time.process_time() - cpu0 > cpu_seconds - 65:
            break
        prompt, sol = (torch.from_numpy(x) for x in data.batch(rng, 2))
        for name, state in states.items():
            # Greedy eval-mode policy draws no randomness: matched dropout per step.
            torch.manual_seed(1000 + step)
            state["net"].train()
            loss = state["ns"]["tf_loss"](prompt, sol)
            loss.backward()
            for group in state["opt"].param_groups:
                group["lr"] = lr_at(step, 5e-5, 200, 2500 if name == "uniform" else 1875, 0.1)
            torch.nn.utils.clip_grad_norm_(state["params"], 1.0)
            state["opt"].step()
            state["opt"].zero_grad(set_to_none=True)
            state["last_loss"] = float(loss.detach())
        finished = step + 1
        if finished % 25 == 0:
            print("step", finished, "cpu_s", round(time.process_time()-cpu0, 2),
                  {k: s["last_loss"] for k, s in states.items()}, flush=True)
        if finished == 100:
            for state in states.values():
                evaluate(state, finished)
    for state in states.values():
        if state["curve"][-1]["step"] != finished:
            evaluate(state, finished)
    record = dict(torch=torch.__version__, precision="CPU fp32; TF32 disabled", threads=1,
                  seed=0, batch_size=2, requested_steps=steps, completed_steps=finished,
                  max_cpu_seconds=cpu_seconds, cpu_s=time.process_time()-cpu0,
                  wall_s=time.perf_counter()-started, lr=5e-5, warmup=200,
                  validation="first 32 generated held-out puzzles; no released test scoring",
                  data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  assets=json.loads((ROOT / "tests/audit_assets.json").read_text()),
                  arms={name: s["curve"] for name, s in states.items()},
                  limitations="One training seed, batch 2, fp32, same steps not equal FLOPs; diagnostic only, not S1 replication.")
    # Asset URLs contain no run-local information. Keep the small numeric record public.
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing["cpu_experiment"] = record
    OUT.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "assets"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        run()
