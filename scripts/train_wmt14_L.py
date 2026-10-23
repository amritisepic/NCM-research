"""Tier L WMT14 De-En: consistency fine-tune of the released 120M NCM so that exact Jacobi decoding across levels
converges in fewer passes (criterion d, text half). Tokenizer frozen. Resumable; CUDA or CPU.

Arms (config `arm`; pre-registered as J1 in docs/preregistration.md):
  ctrl     teacher forcing on reference translations (clean codes), the released conditional recipe
  consist  ctrl's loss on half of each batch; on the other half, self-consistency: for fresh sources, run the
           current model's greedy Jacobi trajectory to its fixed point (no grad, eval mode), sample `m` of the
           trajectory's input states before convergence (pass-1 input = all-zero codes), and train one parallel
           pass from each state to predict the fixed point's codes (the model's own sequential greedy output).
           Steps are cut so both arms use the same training FLOPs (trajectory passes counted).
Evaluation (fp32, as wave 0): sequential greedy decoding (11 NFE) and exact Jacobi (stop when no code changes:
output identical to sequential), mean passes, BLEU (lowercased sacreBLEU, as R020) on newstest2013 (val,
selection) and once on newstest2014 (test, data/wmt14/sources.pickle).

  python scripts/prep_wmt14.py --n 400000
  python scripts/train_wmt14_L.py --config configs/scale/L_wmt_consist.yaml --run_dir runs/L_wmt_consist_s0 --seed 0
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

from ncmr import flops as FL  # noqa: E402
from ncmr.jacobi_train import jacobi_decode, parallel_logits  # noqa: E402
from ncmr.models import n_params  # noqa: E402
from ncmr.ncm_ref_loader import load_ncm, safe_pickle_load  # noqa: E402
from ncmr.train import Calib, Trainer, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--run_dir", required=True)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--data", default=os.path.join(ROOT, "data", "wmt14", "train_pairs.npz"))
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--max_minutes", type=float, default=0)
ap.add_argument("--calib_steps", type=int, default=0)
ap.add_argument("--eval_only", action="store_true", help="no training: final eval of the released model (J0)")
ap.add_argument("--n_val", type=int, default=0, help="final eval: first n val sentences (0 = all)")
ap.add_argument("--n_test", type=int, default=0, help="final eval: first n test sentences (0 = all 3003)")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
cfg = OmegaConf.load(a.config)
tc, ec, kc = cfg.train, cfg.get("eval", {}), cfg.get("consist", {})
seed = int(a.seed if a.seed is not None else tc.get("seed", 0))
cfg.train.seed = seed
arm = str(cfg.arm)
if a.calib_steps:
    tc.steps, tc.eval_every, tc.ckpt_every = a.calib_steps, 10 ** 9, 10 ** 9
if a.eval_only:
    tc.steps = 0
dev = pick_device()
torch.manual_seed(seed)
rng = np.random.default_rng(seed)
os.makedirs(a.run_dir, exist_ok=True)
OmegaConf.save(cfg, os.path.join(a.run_dir, "config.yaml"))

B_ = load_ncm(os.path.join(ROOT, "artifacts", "wmt14"), device=dev)
net, tok, pt = B_.net, B_.vqvae.eval(), B_.pre_tokenizer
for p in tok.parameters():
    p.requires_grad_(False)
vq = tok.vq
levels = [int(l) for l in vq.levels]
K, S = len(levels), sum(levels)
P = int(net.prefix_len)
EOS, PAD = pt.eos_token_id, pt.pad_token_id
D = np.load(a.data)
tr_src, tr_tgt = D["src"], D["tgt"]
val_text = json.load(open(a.data.replace(".npz", "_val_text.json")))
m_states = int(kc.get("states_per_traj", 4))
fwd = FL.ncm_gen_fwd_per_seq(int(B_.cfg.model.n_embd), int(B_.cfg.model.n_layer), levels, prefix=P,
                             codebook=vq.vqvae_vocab_sizes[0], vq_dim=tok.embed_dim, query_level0=True)
bs = int(tc.batch_size)
n_c = bs // 2 if arm == "consist" else 0
# per step: (bs - n_c) TF + n_c consistency samples (3 units each) + trajectories for n_c / m sources (<= K + 1 passes)
fl_step = fwd * (3 * bs + (K + 1) * (n_c // max(m_states, 1)))
params = [p for p in net.parameters() if p.requires_grad]
trainer = Trainer(a.run_dir, params, max_lr=float(tc.lr), total_steps=int(tc.steps), warmup=int(tc.warmup),
                  weight_decay=float(tc.weight_decay), betas=(0.9, float(tc.get("beta2", 0.95))),
                  ckpt_every=int(tc.ckpt_every), device=dev, amp=bool(tc.get("amp", True)),
                  min_lr_frac=float(tc.get("min_lr_frac", 0.1)))
modules = {"model": net}
trainer.maybe_resume(modules, rng)
print(f"[wmt14-L] arm={arm} params {n_params(net) / 1e6:.2f}M | <= {fl_step / 1e12:.2f} TFLOP/step, "
      f"{fl_step * int(tc.steps) / 1e15:.1f} PFLOP total (trajectory passes at the cap) | device {dev}", flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def prefix(src):
    return tok.encoder.wte(src), src == PAD


def canon(tgt):
    with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
        return tok.encode(tgt)


def ce(logits, target):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))


def batch(n):
    i = rng.integers(0, len(tr_src), size=n)
    return torch.from_numpy(tr_src[i].astype(np.int64)).to(dev), torch.from_numpy(tr_tgt[i].astype(np.int64)).to(dev)


def tf_loss(src, tgt):
    z, pm = prefix(src)
    idx = canon(tgt)
    return ce(parallel_logits(net, z, idx, pm), idx)


traj_stats = {"passes": 0.0, "n": 0, "states_before_fp": 0.0}


def consist_loss(n):
    """Self-consistency on the current model's own greedy Jacobi trajectories."""
    g = max(1, n // m_states)
    src, _ = batch(g)
    z, pm = prefix(src)
    was = net.training
    net.eval()
    with torch.no_grad():
        out = jacobi_decode(net, lambda c, h: z, g, S, dev, K + 1, pad_mask=pm, keep_history=True)
    net.train(was)
    y = out["codes"]
    states = [torch.zeros_like(y)] + out["history"]            # input of pass t+1 = output of pass t
    hist = torch.stack(states, 1)                               # (g, T+1, S)
    first_fp = (hist == y[:, None]).all(-1).float().argmax(1)   # first state equal to the fixed point (>= 1)
    t = (torch.rand(g, m_states, device=dev) * first_fp[:, None]).long()   # uniform over states before it
    inp = hist[torch.arange(g, device=dev)[:, None], t].reshape(-1, S)[:n]
    tgt = y.repeat_interleave(m_states, 0)[:n]
    zz, pmm = z.repeat_interleave(m_states, 0)[:n], pm.repeat_interleave(m_states, 0)[:n]
    traj_stats["passes"] += float(out["nfe"].float().sum())
    traj_stats["states_before_fp"] += float(first_fp.sum())
    traj_stats["n"] += g
    return ce(parallel_logits(net, zz, inp, pmm), tgt)


def detok(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return pt.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


@torch.no_grad()
def evaluate(srcs, refs, bs_=128, sequential=True, caps=(6, 8, 9)):
    """fp32 eval: exact Jacobi passes + BLEU of its output (= sequential output); optionally verifies that.
    caps: also the output after at most c passes (fixed-cap Jacobi, J1a), with BLEU, mean passes and length ratio."""
    import sacrebleu
    net.eval()
    hyp, nfe, same = [], [], []
    hyp_cap = {c: [] for c in caps}
    with torch.autocast(device_type=dev, enabled=False):
        for s in range(0, len(srcs), bs_):
            ids = [(pt.encode(t)[:P] + [PAD] * P)[:P] for t in srcs[s:s + bs_]]
            x = torch.tensor(ids, device=dev)
            z, pm = prefix(x)
            out = jacobi_decode(net, lambda c, h: z, len(x), S, dev, K + 1, pad_mask=pm, keep_history=True)
            if sequential:
                seq = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=EOS)
                same += (seq == out["codes"]).all(1).tolist()
            with torch.autocast(device_type=dev, enabled=False):
                toks = tok.decoder(vq.idx_to_z_q(out["codes"])).argmax(-1).tolist()
            hyp += [detok(r) for r in toks]
            nfe += out["nfe"].tolist()
            for c in caps:      # done samples keep their codes in later history entries
                hc = out["history"][min(c, len(out["history"])) - 1]
                with torch.autocast(device_type=dev, enabled=False):
                    hyp_cap[c] += [detok(r) for r in tok.decoder(vq.idx_to_z_q(hc)).argmax(-1).tolist()]
    net.train()
    res = {"n": len(srcs), "exact_passes_mean": float(np.mean(nfe)), "exact_passes_hist": np.bincount(nfe).tolist(),
           "sequential_nfe": K, "exact_nfe_saving_pct": 100 * (1 - float(np.mean(nfe)) / K),
           "bleu_lc": sacrebleu.corpus_bleu(hyp, [list(refs)], lowercase=True).score}
    b = sacrebleu.corpus_bleu(hyp, [list(refs)], lowercase=True)
    res["len_ratio"] = b.sys_len / max(b.ref_len, 1)
    res["capped"] = {}
    for c in caps:
        bc = sacrebleu.corpus_bleu(hyp_cap[c], [list(refs)], lowercase=True)
        res["capped"][str(c)] = {"passes_mean": float(np.mean(np.minimum(nfe, c))), "bleu_lc": bc.score,
                                 "len_ratio": bc.sys_len / max(bc.ref_len, 1),
                                 "identical_to_exact_pct": 100 * float(np.mean([u == v for u, v in zip(hyp_cap[c], hyp)]))}
    res["nfe_per_sentence"] = nfe
    if sequential:
        res["fixed_point_equals_sequential_pct"] = 100 * float(np.mean(same))
    return res, hyp


@torch.no_grad()
def val_nll(n=1000):
    net.eval()
    vs = torch.from_numpy(D["val_src"][:n].astype(np.int64)).to(dev)
    vt = torch.from_numpy(D["val_tgt"][:n].astype(np.int64)).to(dev)
    tot = 0.0
    for s in range(0, len(vs), 100):
        z, pm = prefix(vs[s:s + 100])
        idx = canon(vt[s:s + 100])
        with trainer.ctx:
            lg = parallel_logits(net, z, idx, pm).float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), idx.reshape(-1), reduction="sum").item()
    net.train()
    return tot / len(vs)


# ------------------------------------------------------------------------------------------------ loop
net.train()
deadline = time.time() + 60 * a.max_minutes if a.max_minutes else None
calib = Calib(a.run_dir, a.calib_steps, fl_step) if a.calib_steps else None
n_curve = int(ec.get("n_curve", 500))
while trainer.step < int(tc.steps):
    with trainer.ctx:
        src, tgt = batch(bs - n_c)
        loss_tf = tf_loss(src, tgt)
        if n_c:
            loss_c = consist_loss(n_c)
            loss = ((bs - n_c) * loss_tf + n_c * loss_c) / bs
            logs = {"loss_tf": round(loss_tf.item(), 4), "loss_consist": round(loss_c.item(), 4)}
        else:
            loss, logs = loss_tf, {}
    trainer.backward(loss)
    gn = trainer.step_opt(params)
    if calib:
        calib.tick(trainer.step)
        continue
    if trainer.step % int(tc.get("log_every", 50)) == 0 or trainer.step == 1:
        if traj_stats["n"]:
            logs.update({"traj_passes": round(traj_stats["passes"] / traj_stats["n"], 3),
                         "traj_states_before_fp": round(traj_stats["states_before_fp"] / traj_stats["n"], 3)})
            traj_stats.update(passes=0.0, n=0, states_before_fp=0.0)
        print(json.dumps(trainer.log({"loss": round(loss.item(), 4), "gnorm": round(gn, 3), **logs})), flush=True)
    if trainer.step % int(tc.eval_every) == 0 or trainer.step == int(tc.steps):
        ev, _ = evaluate(val_text["src"][:n_curve], val_text["ref"][:n_curve], sequential=False)
        ev.pop("nfe_per_sentence")
        ev["val_nll"] = val_nll()
        print(json.dumps(trainer.log({"eval": ev})), flush=True)
    if trainer.step % int(tc.ckpt_every) == 0:
        trainer.save(modules, rng)
    if deadline and time.time() > deadline:
        trainer.save(modules, rng)
        print("[stop] time budget reached; resume by re-running the same command", flush=True)
        sys.exit(0)

if calib:
    calib.report(trainer.step, {"arm": arm, "batch_size": bs})
    sys.exit(0)
if not a.eval_only:
    trainer.save(modules, rng)
    torch.save({"model": net.state_dict()}, os.path.join(a.run_dir, "final.pt"))
nv = a.n_val or len(val_text["src"])
final = {"val_nll": val_nll()}
final["val"], _ = evaluate(val_text["src"][:nv], val_text["ref"][:nv])
srcs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "sources.pickle"))
refs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "reference_translations.pickle"))
nt = a.n_test or len(srcs)
final["test"], hyp = evaluate(srcs[:nt], refs[:nt])
json.dump({"hyp_test": hyp, "nfe_test": final["test"].pop("nfe_per_sentence")}, open(os.path.join(a.run_dir, "hyp_test.json"), "w"))
final["val"].pop("nfe_per_sentence", None)
res = {"run": os.path.basename(a.run_dir.rstrip("/")), "arm": arm, "config": OmegaConf.to_container(cfg),
       "params_total": n_params(net), "steps": trainer.step, "flops_est_total_upper": fl_step * trainer.step,
       "wall_s": trainer.wall_prev + time.time() - trainer.t_start, "device": dev,
       "gpu": torch.cuda.get_device_name() if dev == "cuda" else None, "torch": torch.__version__, "final_eval": final}
json.dump(res, open(os.path.join(a.run_dir, "result.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1))
