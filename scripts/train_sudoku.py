"""Train our Sudoku models on generated puzzles (scripts/gen_sudoku.py). Resumable; CPU or CUDA.

  python scripts/train_sudoku.py tokenizer --config configs/sudoku/tok.yaml --level hard --run_dir runs/sdk_tok_hard
  python scripts/train_sudoku.py ncm  --config configs/sudoku/gen_s.yaml --level hard --tokenizer runs/sdk_tok_hard/final.pt --run_dir runs/sdk_ncm_hard
  python scripts/train_sudoku.py rncm --config configs/sudoku/gen_s.yaml --level hard --tokenizer runs/sdk_tok_hard/final.pt --run_dir runs/sdk_rncm_hard

`ncm` trains the upstream NextConceptTransformer exactly as NCM does (one parallel teacher-forced pass);
`rncm` trains the same network wrapped in RecurrentNCM (state carried across levels, backprop through
levels). Validation = greedy solve rate on our held-out generated puzzles; the released test set is used
only by --final (after training), never for model selection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.models import TiedOutputVQVAE, build_ncm_generator, n_params, tokenizer_kwargs  # noqa: E402
from ncmr.ncm_ref_loader import safe_pickle_load  # noqa: E402
from ncmr.recurrent import RecurrentNCM  # noqa: E402
from ncmr.sudoku import cells, propagation_rounds  # noqa: E402
from ncmr.sudoku_data import SudokuTrain  # noqa: E402
from ncmr.train import Trainer, pick_device  # noqa: E402

SUDOKU_LEVELS = [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["tokenizer", "ncm", "rncm"])
ap.add_argument("--config", required=True)
ap.add_argument("--level", default="hard")
ap.add_argument("--run_dir", required=True)
ap.add_argument("--tokenizer", default="")
ap.add_argument("--data", default="")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--final", action="store_true", help="after training, evaluate on the released test set")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
cfg = OmegaConf.load(a.config)
tc = cfg.train
dev = pick_device()
torch.manual_seed(int(tc.seed))
rng = np.random.default_rng(int(tc.seed))
data = SudokuTrain(a.level, a.data or None)
os.makedirs(a.run_dir, exist_ok=True)
OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))


def sudoku_tokenizer(**kw):
    kw = {"embed_dim": 256, "codebook": 16, "levels": SUDOKU_LEVELS, "in_vocab": 12, "context": 89, **kw}
    t = TiedOutputVQVAE(**tokenizer_kwargs(**kw))
    return t, kw


def load_tok(path):
    st = torch.load(path, map_location="cpu", weights_only=False)  # our own file
    t, _ = sudoku_tokenizer(**st["tok_kwargs"])
    t.load_state_dict(st["model"])
    for p in t.parameters():
        p.requires_grad_(False)
    return t.to(dev).eval()


extra = {}
if a.kind == "tokenizer":
    model, kw = sudoku_tokenizer(**OmegaConf.to_container(cfg.model, resolve=True))
    model = model.to(dev)
    extra["tok_kwargs"] = kw
else:
    tok = load_tok(a.tokenizer)
    gk = OmegaConf.to_container(cfg.model, resolve=True)
    net = build_ncm_generator(tok, prefix_len=91, query_level0=True, rope_base=10000, **gk).to(dev)
    net.config.tie_heads_to_codebook = False
    model = RecurrentNCM(net, carry=(a.kind == "rncm")).to(dev)  # carry=False is exactly upstream NCM
    extra.update({"tokenizer_path": os.path.abspath(a.tokenizer), "gen_kwargs": gk, "kind": a.kind})
params = [p for p in model.parameters() if p.requires_grad]
tr = Trainer(a.run_dir, params, max_lr=float(tc.lr), total_steps=int(tc.steps), warmup=int(tc.warmup),
             weight_decay=float(tc.weight_decay), ckpt_every=int(tc.ckpt_every), device=dev, amp=bool(tc.get("amp", True)))
tr.maybe_resume({"model": model}, rng)
print(f"[sudoku {a.kind}] params {n_params(model) / 1e6:.2f}M trainable {n_params(model, True) / 1e6:.2f}M device {dev}", flush=True)


def gen_inputs(prompt, sol):
    """Teacher-forcing tensors: z_prefix (raw-token prefix), z_in (levels 1..K-1 inputs), targets (all codes)."""
    with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
        idx = tok.encode(sol)
        z_in = tok.vq.idx_to_ncm_input(idx)
        z_prefix = tok.encoder.wte(prompt)
    return z_prefix, z_in, idx


@torch.no_grad()
def solve_rate(prompts, sols, bs=256):
    model.eval()
    ok, boards = [], []
    for s in range(0, len(prompts), bs):
        p = torch.as_tensor(prompts[s:s + bs], device=dev)
        g = model.generate(p, temperature=0.0)
        out = tok.decoder(tok.vq.idx_to_z_q(g)).argmax(-1).cpu().numpy()
        for j in range(len(out)):
            b = cells(out[j].tolist())
            boards.append(b)
            ok.append(b == cells(list(sols[s + j])))
    model.train()
    return float(np.mean(ok)), ok


val_p, val_s = data.val(int(tc.get("eval_n", 1000)))
last = {}
model.train()
while tr.step < int(tc.steps):
    prompt, sol = data.batch(rng, int(tc.batch_size))
    prompt, sol = torch.from_numpy(prompt).to(dev), torch.from_numpy(sol).to(dev)
    with tr.ctx:
        if a.kind == "tokenizer":
            from ncmr.models import tokenizer_loss
            loss, logs = tokenizer_loss(model, sol)
        else:
            z_prefix, z_in, idx = gen_inputs(prompt, sol)
            logits = model(z_prefix, z_in)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), idx.reshape(-1))
            logs = {}
    gn = tr.backward_step(loss, params)
    if tr.step % int(tc.get("log_every", 100)) == 0 or tr.step == 1:
        print(json.dumps(tr.log({"loss": round(loss.item(), 4), "gnorm": round(gn, 3), **logs})), flush=True)
    if tr.step % int(tc.eval_every) == 0 or tr.step == int(tc.steps):
        if a.kind == "tokenizer":
            model.eval()
            with torch.no_grad():
                y = torch.as_tensor(val_s, device=dev)
                rec = model.decoder(model.vq.idx_to_z_q(model.encode(y))).argmax(-1)
                last = {"val_token_acc": float((rec == y).float().mean()), "val_board_exact": float((rec == y).all(1).float().mean())}
            model.train()
        else:
            last = {"val_solved": solve_rate(val_p, val_s)[0]}
        print(json.dumps(tr.log({"eval": last})), flush=True)
    if tr.step % int(tc.ckpt_every) == 0:
        tr.save({"model": model}, rng, extra)

if not last and a.kind != "tokenizer":  # re-run of a finished run: evaluate once so result.json is complete
    last = {"val_solved": solve_rate(val_p, val_s)[0]}
tr.save({"model": model}, rng, extra)
torch.save({"model": model.state_dict(), **extra}, os.path.join(a.run_dir, "final.pt"))
res = {"run": os.path.basename(a.run_dir.rstrip("/")), "kind": a.kind, "level": a.level,
       "config": OmegaConf.to_container(cfg), "params_total": n_params(model), "steps": tr.step,
       "wall_s": tr.wall_prev + time.time() - tr.t_start, "device": dev, "final_eval_val": last}
if a.final and a.kind != "tokenizer":
    P = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", a.level, "puzzles.pickle"))
    S = safe_pickle_load(os.path.join(ROOT, "data", "sudoku", a.level, "solutions.pickle"))
    rate, ok = solve_rate(np.array(P), np.array(S))
    rounds = [propagation_rounds(cells(p)) for p in P]
    by = {}
    for name, sel in [("le6", lambda r, s: s and r <= 6), ("7-8", lambda r, s: s and 7 <= r <= 8),
                      ("9-10", lambda r, s: s and 9 <= r <= 10), ("ge11", lambda r, s: s and r >= 11),
                      ("search", lambda r, s: not s)]:
        m = [i for i, (r, s_) in enumerate(rounds) if sel(r, s_)]
        by[name] = {"n": len(m), "solved_pct": 100 * float(np.mean([ok[i] for i in m])) if m else None}
    res["test_released"] = {"solved_pct": 100 * rate, "by_propagation_depth": by, "nfe_per_sample": len(SUDOKU_LEVELS)}
json.dump(res, open(os.path.join(a.run_dir, "result.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1))
