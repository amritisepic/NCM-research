"""Exploratory, not registered: does fine-tuning on the generator's own codes damage the released Sudoku-hard model
the way wave 1's on-policy arms damaged the TinyStories model? CPU, fp32, a few hundred small steps per arm.

  python scripts/onpolicy_repro.py --arm onpol_sample --steps 200 --out results/onpolicy_repro/onpol_sample_s0.json

Arms (canonical labels and NCM's uniform rates, mean 0.5, linear 0.75 -> 0.25, unless noted):
  none           plain teacher forcing (no corruption)
  uniform        uniform random codes at masked positions (NCM's recipe; S1's A1u)
  onpol_sample   the model's own one-step samples (T = 1) given the clean coarser levels (W1's A3 policy)
  onpol_greedy   the model's own one-step argmax (S1's A2 policy)
  onpol_frozen   one-step samples (T = 1) from a frozen copy of the initial model: no policy feedback loop
  onpol_requant  onpol_sample with re-quantized labels and unmasked finer inputs (W1's A4 analogue)
  swap           the canonical codes of another training solution: well-formed codes that carry no information
                 about this puzzle and do not come from the model
  uniform_wrong  every masked code replaced by a uniformly random *different* code (all masked slots corrupted)
  plausible_wrong every masked code replaced by a sample from the model's own one-step distribution with the
                 canonical code excluded: same slots and rates as uniform_wrong, only the choice of wrong code differs
  plausible_wrong_requant  plausible_wrong with re-quantized labels and unmasked finer inputs
Evaluated every `eval_every` steps on the same `n_eval` held-out puzzles (our generated set's tail): sequential
greedy solve rate (D1, 12 NFE), clean teacher-forced NLL of the canonical codes per level, and round-trip consistency
of the generated hierarchy (share of generated codes equal to the canonical codes of the board they decode to).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr.corruption import CorruptionCfg, build_hierarchy  # noqa: E402
from ncmr.jacobi_train import parallel_logits  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm  # noqa: E402
from ncmr.sudoku import cells  # noqa: E402
from ncmr.sudoku_data import SudokuTrain  # noqa: E402
from ncmr.train import Trainer  # noqa: E402

ARMS = ("none", "uniform", "onpol_sample", "onpol_greedy", "onpol_frozen", "onpol_requant", "swap",
        "uniform_wrong", "plausible_wrong", "plausible_wrong_requant")
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=ARMS)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--eval_every", type=int, default=50)
ap.add_argument("--n_eval", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "sudoku", "hard"))
ap.add_argument("--data", default=os.path.join(ROOT, "assets", "sudoku_train", "hard.npz"))
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--sampled_eval", action="store_true", help="also decode at temperature 1 (seeded) and score it")
ap.add_argument("--save", default="", help="save the final weights here (an ignored path, e.g. runs/...)")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)

B_ = load_ncm(a.model, device="cpu")
net, tok = B_.net, B_.vqvae.eval()
for p in tok.parameters():
    p.requires_grad_(False)
for m in net.modules():   # S1's fine-tuning dropout
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.1
levels = [int(l) for l in tok.vq.levels]
K, S = len(levels), sum(levels)
frozen = copy.deepcopy(net).eval() if a.arm == "onpol_frozen" else None
data = SudokuTrain("hard", a.data)
val_p, val_s = (torch.as_tensor(np.asarray(v)) for v in data.val(a.n_eval))
mode = {"none": "none", "uniform": "uniform"}.get(a.arm, "onpolicy")
corr = CorruptionCfg(mode=mode, mean_rate=0.5, spread=0.5, retarget=a.arm.endswith("requant"))
gen = torch.Generator().manual_seed(a.seed * 7919)          # corruption masks: identical across arms
rep_gen = torch.Generator().manual_seed(a.seed * 7919 + 1)  # uniform_wrong replacement draws
params = [p for p in net.parameters() if p.requires_grad]
trainer = Trainer(tempfile.mkdtemp(prefix="onpol_repro_"), params, max_lr=a.lr, total_steps=a.steps, warmup=a.warmup,
                  weight_decay=0.01, betas=(0.9, 0.99), ckpt_every=10 ** 9, device="cpu", amp=False)


def canon(x):
    with torch.no_grad():
        return tok.encode(x)


def per_level(t):
    """(B, S) -> list of K per-level means."""
    return [float(c.float().mean()) for c in torch.split(t, levels, dim=1)]


def loss_fn(p, s):
    z = tok.encoder.wte(p)
    c = canon(s)
    if a.arm == "none":
        logits = parallel_logits(net, z, c)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), c.reshape(-1)), {}
    pol = None
    if a.arm.startswith("onpol"):
        src = frozen if frozen is not None else net
        was = net.training
        src.eval()
        with torch.no_grad():
            lg = parallel_logits(src, z, c).float()
        net.train(was)
        pol = lg.argmax(-1) if a.arm == "onpol_greedy" else torch.distributions.Categorical(logits=lg).sample()
    elif a.arm == "swap":
        pol = c.roll(1, 0)
    elif a.arm == "uniform_wrong":
        V = tok.vq.vqvae_vocab_sizes[0]
        pol = (c + torch.randint(1, V, c.shape, generator=rep_gen)) % V
    elif a.arm.startswith("plausible_wrong"):
        was = net.training
        net.eval()
        with torch.no_grad():
            lg = parallel_logits(net, z, c).float()
        net.train(was)
        lg.scatter_(-1, c[..., None], float("-inf"))
        pol = torch.distributions.Categorical(logits=lg).sample()
    h = build_hierarchy(tok, s, corr, policy_codes=pol, generator=gen)
    q0 = [net.build_l0_query_input(s.shape[0], h["inputs"].dtype)]
    logits = net(torch.cat([z, q0[0], h["inputs"]], 1))
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), h["targets"].reshape(-1))
    st = {"corrupted_frac_per_level": [round(f, 4) for f in h["corrupted_frac_per_level"]],
          "labels_changed": round(float((h["targets"] != h["canonical"]).float().mean()), 4)}
    if pol is not None:
        st["policy_differs_per_level"] = [round(f, 4) for f in per_level(pol != c)]
    return loss, st


@torch.no_grad()
def evaluate():
    net.eval()
    solved, solved_t1, ent, nll, rt = [], [], torch.zeros(K), torch.zeros(K), torch.zeros(K)
    for i in range(0, len(val_p), 100):
        p, s = val_p[i:i + 100], val_s[i:i + 100]
        g = net.generate(None, p, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
        ids = tok.decoder(tok.vq.idx_to_z_q(g)).argmax(-1)
        solved += [cells(r) == cells(t) for r, t in zip(ids.tolist(), s.tolist())]
        rt += torch.tensor(per_level(canon(ids) == g)) * len(p)
        if a.sampled_eval:   # temperature-1 sequential sampling (as W1's text samples), its own RNG stream
            with torch.random.fork_rng():
                torch.manual_seed(1234 + i)
                g1 = net.generate(None, p, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=None)
            ids1 = tok.decoder(tok.vq.idx_to_z_q(g1)).argmax(-1)
            solved_t1 += [cells(r) == cells(t) for r, t in zip(ids1.tolist(), s.tolist())]
        c = canon(s)
        lg = parallel_logits(net, tok.encoder.wte(p), c).float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), c.reshape(-1), reduction="none").view(c.shape)
        nll += torch.stack([x.sum(1) for x in torch.split(ce, levels, dim=1)], 1).sum(0)
        h = -(lg.log_softmax(-1) * lg.softmax(-1)).sum(-1)   # predictive entropy on clean contexts (hedging)
        ent += torch.stack([x.mean(1) for x in torch.split(h, levels, dim=1)], 1).sum(0)
    net.train()
    n = len(val_p)
    out = {"solved_pct": 100 * float(np.mean(solved)), "nll": float(nll.sum() / n),
           "nll_per_level": (nll / n).round(decimals=3).tolist(), "roundtrip_per_level": (rt / n).round(decimals=4).tolist(),
           "entropy_per_code_per_level": (ent / n).round(decimals=4).tolist(), "solved": [bool(v) for v in solved]}
    if solved_t1:
        out.update(solved_t1_pct=100 * float(np.mean(solved_t1)), solved_t1=[bool(v) for v in solved_t1])
    return out


out = {"arm": a.arm, "args": {k: v for k, v in vars(a).items() if k not in ("model", "data", "out", "save")}, "levels": levels,
       "evals": [], "log": []}
t0 = time.time()
net.train()
ev = evaluate()
out["evals"].append({"step": 0, "wall_s": round(time.time() - t0, 1), **ev})
print(json.dumps({"step": 0, "solved_pct": ev["solved_pct"], "solved_t1_pct": ev.get("solved_t1_pct"),
                  "nll": round(ev["nll"], 2)}), flush=True)
while trainer.step < a.steps:
    p, s = (torch.from_numpy(v) for v in data.batch(rng, a.batch))
    loss, st = loss_fn(p, s)
    gn = trainer.backward_step(loss, params)
    if trainer.step % 10 == 0 or trainer.step == 1:
        rec = {"step": trainer.step, "wall_s": round(time.time() - t0, 1), "loss": round(loss.item(), 4),
               "gnorm": round(gn, 3), **st}
        out["log"].append(rec)
        print(json.dumps(rec), flush=True)
    if trainer.step % a.eval_every == 0 or trainer.step == a.steps:
        ev = evaluate()
        out["evals"].append({"step": trainer.step, "wall_s": round(time.time() - t0, 1), **ev})
        print(json.dumps({"step": trainer.step, "solved_pct": ev["solved_pct"], "solved_t1_pct": ev.get("solved_t1_pct"),
                          "nll": round(ev["nll"], 2),
                          "roundtrip": ev["roundtrip_per_level"]}), flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
if a.save:
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    torch.save(net.state_dict(), a.save)
print("wrote", a.out)
