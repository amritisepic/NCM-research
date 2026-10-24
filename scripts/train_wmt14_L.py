"""Tier L WMT14 De-En: consistency fine-tune of the released 120M NCM so that exact Jacobi decoding across levels
converges in fewer passes (criterion d, text half). Tokenizer frozen. Resumable; CUDA or CPU.

Arms (config `arm`; pre-registered as J1 in docs/preregistration.md):
  ctrl     teacher forcing on reference translations (clean codes), the released conditional recipe
  consist  ctrl's loss on half of each batch; on the other half, self-consistency: for fresh sources, run the
           current model's greedy Jacobi trajectory to its fixed point (no grad, eval mode), sample `m` of the
           trajectory's input states before convergence (pass-1 input = all-zero codes), and train one parallel
           pass from each state to predict the fixed point's codes (the model's own sequential greedy output).
           Steps are cut so both arms use the same training FLOPs (trajectory passes counted).
  consist_frozen  J2-F: consist with the trajectories and fixed points taken from a frozen copy of the released
           generator (eval mode, no grad, never updated or checkpointed) instead of the current model; the student,
           its loss and every draw are consist's.
Evaluation (fp32, as wave 0): sequential greedy decoding (11 NFE) and exact Jacobi (stop when no code changes:
output identical to sequential), mean passes, BLEU (lowercased sacreBLEU, as R020) on newstest2013 (val,
selection) and once on newstest2014 (test, data/wmt14/sources.pickle).

  python scripts/prep_wmt14.py --n 400000
  python scripts/train_wmt14_L.py --config configs/scale/L_wmt_consist.yaml --run_dir runs/L_wmt_consist_s0 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
if dev == "cuda":   # strict fp32 for the no-grad teacher and all evaluation, as wave 0 decoded (codex audit2 A2-4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(seed)
rng = np.random.default_rng(seed)
os.makedirs(a.run_dir, exist_ok=True)


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# run identity, validated before anything in run_dir is overwritten (codex audit2 A2-3)
ckpt_path, manifest_path = os.path.join(a.run_dir, "ckpt.pt"), os.path.join(a.run_dir, "run_manifest.json")
if a.eval_only and os.path.exists(ckpt_path):
    sys.exit(f"--eval_only evaluates the released model, but {a.run_dir} holds a training checkpoint")
_base = os.path.join(ROOT, "artifacts", "wmt14")
manifest = {"config": OmegaConf.to_container(cfg, resolve=True), "data_sha256": digest(a.data),
            "base_config_sha256": digest(os.path.join(_base, "config.yaml")),
            "base_sha256": {n: digest(os.path.join(_base, n)) for n in ("tokenizer.pt", "generator.pt")},   # audit 4, A4-2
            "device": dev}
if os.path.exists(ckpt_path) and not a.calib_steps:
    if not os.path.exists(manifest_path) or json.load(open(manifest_path)) != manifest:
        sys.exit(f"{a.run_dir} holds a checkpoint of a different (or unrecorded) run: use a fresh run directory")
if not a.calib_steps:
    json.dump(manifest, open(manifest_path, "w"), indent=1)
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
if arm not in ("ctrl", "consist", "consist_frozen"):
    raise ValueError(f"unknown arm {arm!r}")
n_c = bs // 2 if arm in ("consist", "consist_frozen") else 0
if m_states < 1 or (n_c and (bs % 2 or n_c % m_states)):
    raise ValueError("consistency needs an even batch and a half-batch divisible by states_per_traj")
# per step: (bs - n_c) TF + n_c consistency samples (3 units each) + trajectories for n_c / m sources, always run for
# exactly K + 1 passes (full budget), so the executed generator proxy equals this number (codex audit2 A2-7)
fl_step = fwd * (3 * bs + (K + 1) * (n_c // max(m_states, 1)))
# J2-F: the teacher is the released generator itself, copied before any checkpoint is restored into `net`; the
# manifest's base hashes pin it, so it is rebuilt identically on resume and is not part of the checkpoint
teacher = None
if arm == "consist_frozen":
    teacher = copy.deepcopy(net).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
params = [p for p in net.parameters() if p.requires_grad]
trainer = Trainer(a.run_dir, params, max_lr=float(tc.lr), total_steps=int(tc.steps), warmup=int(tc.warmup),
                  weight_decay=float(tc.weight_decay), betas=(0.9, float(tc.get("beta2", 0.95))),
                  ckpt_every=int(tc.ckpt_every), device=dev, amp=bool(tc.get("amp", True)),
                  min_lr_frac=float(tc.get("min_lr_frac", 0.1)))
modules = {"model": net}
resume_extra = trainer.maybe_resume(modules, rng)
print(f"[wmt14-L] arm={arm} params {n_params(net) / 1e6:.2f}M | {fl_step / 1e12:.2f} TFLOP/step, "
      f"{fl_step * int(tc.steps) / 1e15:.1f} PFLOP total (generator proxy) | device {dev}", flush=True)


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


_rs = resume_extra.get("run_stats", {})
traj_stats = {"passes": 0.0, "n": 0, "states_before_fp": 0.0, "degenerate": 0,
              # run-level totals for J1b's 1% degenerate-source rule: never reset by logging, checkpointed, restored
              "run_sources": int(_rs.get("sources", 0)), "run_degenerate": int(_rs.get("degenerate", 0))}


def run_stats():
    return {"sources": traj_stats["run_sources"], "degenerate": traj_stats["run_degenerate"]}


def consist_loss(n, sources, teacher_net=None):
    """Self-consistency on greedy Jacobi trajectories (n // m_states sources): the current model's own (J2), or
    those of `teacher_net`, a frozen copy of the released generator (J2-F). The student is always `net`.
    Returns (loss or None, number of training states used)."""
    g = n // m_states
    z, pm = prefix(sources[:g])
    tnet = net if teacher_net is None else teacher_net
    was = tnet.training
    tnet.eval()
    # the teacher is the fp32 greedy map that exact decoding evaluates, not the bf16 training map (codex audit2 A2-4)
    with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
        out = jacobi_decode(tnet, lambda c, h: z.float(), g, S, dev, K + 1, pad_mask=pm, keep_history=True, full_budget=True,
                            initial_fixed_ok=True)   # static prefix (codex audit 4, A4-4)
    tnet.train(was)
    if not bool(out["stop"].eq(1).all()):
        raise RuntimeError("consistency teacher did not reach a fixed point in K + 1 passes")
    y = out["codes"]
    states = [torch.zeros_like(y)] + out["history"]            # input of pass t+1 = output of pass t
    hist = torch.stack(states, 1)                               # (g, K + 2, S)
    first_fp = (hist == y[:, None]).all(-1).float().argmax(1)   # first state equal to the fixed point
    keep = first_fp > 0   # J1b: a source whose all-zero start is already its fixed point has no pre-fixed-point state
    traj_stats["passes"] += float(out["nfe"].float().sum())
    traj_stats["states_before_fp"] += float(first_fp.sum())
    traj_stats["n"] += g
    traj_stats["degenerate"] += int((~keep).sum())
    traj_stats["run_sources"] = traj_stats.get("run_sources", 0) + g
    traj_stats["run_degenerate"] = traj_stats.get("run_degenerate", 0) + int((~keep).sum())
    # J1b drops degenerate sources from the loss; their student rows still run, as zero-loss padding, so the executed
    # proxy equals the registered one (codex audit 4, A4-3)
    pad_z = z[~keep].repeat_interleave(m_states, 0)
    pad_pm = pm[~keep].repeat_interleave(m_states, 0)
    pad_inp = torch.zeros(len(pad_z), S, dtype=torch.long, device=dev)
    if not bool(keep.any()):
        lg = parallel_logits(net, pad_z, pad_inp, pad_pm)
        return lg[..., 0].float().sum() * 0.0, 0
    hist, y, z, pm, first_fp = hist[keep], y[keep], z[keep], pm[keep], first_fp[keep]
    gk = len(y)
    t = (torch.rand(gk, m_states, device=dev) * first_fp[:, None]).long()   # uniform over states strictly before it
    inp = hist[torch.arange(gk, device=dev)[:, None], t].reshape(-1, S)
    tgt = y.repeat_interleave(m_states, 0)
    zz, pmm = z.repeat_interleave(m_states, 0), pm.repeat_interleave(m_states, 0)
    lg = parallel_logits(net, torch.cat([zz, pad_z], 0), torch.cat([inp, pad_inp], 0), torch.cat([pmm, pad_pm], 0))
    return ce(lg[:len(tgt)], tgt), len(tgt)


def detok(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return pt.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


@torch.no_grad()
def evaluate(srcs, refs, bs_=128, sequential=True, caps=None):
    """fp32 eval: exact Jacobi passes + BLEU of its output (= sequential output); optionally verifies that.
    caps: also the output after at most c passes (fixed-cap Jacobi, J1a), with BLEU, mean passes and length ratio;
    default every cap 1..K+1 so any comparator cap exists (codex audit2 A2-10).
    Returns (metrics, exact hypotheses, {cap: capped hypotheses})."""
    import sacrebleu
    caps = tuple(range(1, K + 2)) if caps is None else tuple(caps)
    net.eval()
    hyp, nfe, same = [], [], []
    hyp_cap = {c: [] for c in caps}
    with torch.autocast(device_type=dev, enabled=False):
        for s in range(0, len(srcs), bs_):
            ids = [(pt.encode(t)[:P] + [PAD] * P)[:P] for t in srcs[s:s + bs_]]
            x = torch.tensor(ids, device=dev)
            z, pm = prefix(x)
            out = jacobi_decode(net, lambda c, h: z, len(x), S, dev, K + 1, pad_mask=pm, keep_history=True,
                                initial_fixed_ok=True)   # static prefix (codex audit 4, A4-4)
            if sequential:   # disagreements are recorded per sentence (reported, not silently averaged)
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
        res["fixed_point_differs_from_sequential_idx"] = [i for i, v in enumerate(same) if not v]
    return res, hyp, hyp_cap


@torch.no_grad()
def val_nll(n=1000):
    net.eval()
    vs = torch.from_numpy(D["val_src"][:n].astype(np.int64)).to(dev)
    vt = torch.from_numpy(D["val_tgt"][:n].astype(np.int64)).to(dev)
    tot = 0.0
    for s in range(0, len(vs), 100):
        z, pm = prefix(vs[s:s + 100])
        idx = canon(vt[s:s + 100])
        with torch.autocast(device_type=dev, enabled=False):   # fp32 scoring (codex audit 4, A4-1)
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
        # one common draw of bs rows per step in every arm; consist uses the first bs - n_c for teacher forcing and
        # every m-th of the rest as trajectory sources (codex audit2 A2-6)
        src, tgt = batch(bs)
        loss_tf = tf_loss(src[:bs - n_c], tgt[:bs - n_c])
        loss, logs = loss_tf, {}
        if n_c:
            loss_c, n_used = consist_loss(n_c, src[bs - n_c:][::m_states], teacher)
            if loss_c is not None:
                loss = ((bs - n_c) * loss_tf + n_used * loss_c) / (bs - n_c + n_used)
                logs = {"loss_tf": round(loss_tf.item(), 4), "loss_consist": round(loss_c.item(), 4)}
    trainer.backward(loss)
    gn = trainer.step_opt(params)
    if calib:
        calib.tick(trainer.step)
        continue
    if trainer.step % int(tc.get("log_every", 50)) == 0 or trainer.step == 1:
        if traj_stats["n"]:
            logs.update({"traj_passes": round(traj_stats["passes"] / traj_stats["n"], 3),
                         "traj_states_before_fp": round(traj_stats["states_before_fp"] / traj_stats["n"], 3),
                         "traj_degenerate": traj_stats["degenerate"]})
            traj_stats.update(passes=0.0, n=0, states_before_fp=0.0, degenerate=0)
        print(json.dumps(trainer.log({"loss": round(loss.item(), 4), "gnorm": round(gn, 3), **logs})), flush=True)
    if trainer.step % int(tc.eval_every) == 0 or trainer.step == int(tc.steps):
        # every evaluation keeps ordered hypotheses and per-sentence passes, and checks exact = sequential (J1b; audit 4, A4-5)
        ev, curve_hyp, curve_cap = evaluate(val_text["src"][:n_curve], val_text["ref"][:n_curve], sequential=True)
        json.dump({"hyp_val": curve_hyp, "hyp_val_capped": curve_cap, "nfe_val": ev.pop("nfe_per_sentence")},
                  open(os.path.join(a.run_dir, f"hyp_val_step{trainer.step}.json"), "w"))
        ev["val_nll"] = val_nll()
        print(json.dumps(trainer.log({"eval": ev})), flush=True)
    if trainer.step % int(tc.ckpt_every) == 0:
        trainer.save(modules, rng, extra={"run_stats": run_stats()})
    if deadline and time.time() > deadline:
        trainer.save(modules, rng, extra={"run_stats": run_stats()})
        print("[stop] time budget reached; resume by re-running the same command", flush=True)
        sys.exit(0)

if calib:
    calib.report(trainer.step, {"arm": arm, "batch_size": bs})
    sys.exit(0)
if not a.eval_only:
    trainer.save(modules, rng, extra={"run_stats": run_stats()})
    torch.save({"model": net.state_dict()}, os.path.join(a.run_dir, "final.pt"))
nv = a.n_val or len(val_text["src"])
final = {"val_nll": val_nll()}
final["val"], hyp_val, hyp_val_cap = evaluate(val_text["src"][:nv], val_text["ref"][:nv])
srcs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "sources.pickle"))
refs = safe_pickle_load(os.path.join(ROOT, "data", "wmt14", "reference_translations.pickle"))
nt = a.n_test or len(srcs)
final["test"], hyp, hyp_cap = evaluate(srcs[:nt], refs[:nt])
# sentence-ordered exact and capped hypotheses for the J1a paired bootstrap (codex audit2 A2-10)
json.dump({"hyp_test": hyp, "hyp_test_capped": hyp_cap, "nfe_test": final["test"].pop("nfe_per_sentence")},
          open(os.path.join(a.run_dir, "hyp_test.json"), "w"))
json.dump({"hyp_val": hyp_val, "hyp_val_capped": hyp_val_cap, "nfe_val": final["val"].pop("nfe_per_sentence")},
          open(os.path.join(a.run_dir, "hyp_val.json"), "w"))
res = {"run": os.path.basename(a.run_dir.rstrip("/")), "arm": arm, "config": OmegaConf.to_container(cfg),
       "params_total": n_params(net), "steps": trainer.step, "generator_flops_proxy_total": fl_step * trainer.step,
       "wall_s": trainer.wall_prev + time.time() - trainer.t_start, "device": dev,
       "gpu": torch.cuda.get_device_name() if dev == "cuda" else None, "torch": torch.__version__, "final_eval": final,
       # J1b: if more than 1% of trajectory sources over the run are degenerate, report it and do not read H12
       "trajectory_sources": traj_stats["run_sources"], "degenerate_sources": traj_stats["run_degenerate"],
       "degenerate_pct": 100 * traj_stats["run_degenerate"] / max(traj_stats["run_sources"], 1)}
json.dump(res, open(os.path.join(a.run_dir, "result.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1))
