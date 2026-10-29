"""W2M-G eval-only measures (docs/preregistration.md, "W2M-G"): the released generator and W2M's seed-0 final
checkpoints M0, M2, M3, M4 (`<run_root>/L_ts_m*_s<seed>/final.pt`). fp32, TF32 off, eval mode, for every forward,
sample and score.

  python scripts/w2mg_eval.py --run_root <runs dir> --run_dir <out dir> [--seed 0] [--arms m0,m2,m3,m4]
                                                                                       (writes <out dir>/result.json)

  (ii)  crossed replay: (M2, M2), (M3, M3) and the two crossed cells at cuts 2, 4, 6; 8 groups of 256 (seeds 0-7)
        each; Gen-PPL (GPT-2 large, TinyStories-33M), mean token log loss L = log of the group's Gen-PPL, shares
        pi_c / sigma_c / interaction with the group jackknife, the readings, the sanity check and the direction-1 gate
  (iv)  M4 at constant flag f in {0, 0.5, 1} on the same 2048 noise streams; rescue R(f); if and only if the overturn
        rule is met on seeds 0-7 by some f in {0.5, 1}, that f and the pure cells it needs are rerun on seeds 8-15
  (i)   JS(M2, M3) on the clean bank (examples("val", 512)) and on 512 samples each (seeds 900, 901) of the released
        model, M2 and M3; ratios to the clean bank with a window bootstrap
  (iii) root reliance on the clean bank (8 unigram roots per window, seed 2027) for the released model, M0, M2, M3, M4
  (iv)  diagnostics: canonical-label NLL and KL(P_f=0 || P_f) at each f on the clean, corrupted and M4-sample banks

Resumable by cell: every finished cell (metrics, sample ids, groups), bank and per-window array is cached under
<out>/cells/; a rerun skips them and reproduces result.json. A partial result.json ("complete": false) is written as
cells finish. <out>/manifest.json pins the protocol, the SHA-256 of the checkpoints, the released assets and val data, and the device /
GPU / runtime: a run directory made under any other is refused.

--arms (W2M-GR, "W2M-GR" in the registration): the default m0,m2,m3,m4 is W2M-G exactly. `--arms m2,m3` loads only the
released model, M2 and M3 (`<run_root>/<dir>_s<seed>/final.pt`), and runs (i), (ii) and (iii) with (iii)'s models released,
m2, m3: no M4 cells, no m4f0 bank, no corrupted bank, no (iv). The arm set is part of manifest.json and of the result's
protocol block (a run directory refuses another arm set). For a run seed >= 1 the sanity reference of (ii) is W2C's final-eval
GPT-2 large Gen-PPL of that seed (results/colab/w2c1_*/..., read at the start; a missing file exits); seed 0 keeps W2M's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr import w2m as W  # noqa: E402
from ncmr import w2mg as G  # noqa: E402
from ncmr.corruption import CorruptionCfg  # noqa: E402
from ncmr.data import examples  # noqa: E402
from ncmr.train import pick_device  # noqa: E402

ARM_DIRS = {"m0": "L_ts_m0_uniform", "m2": "L_ts_m2_frozen_pw", "m3": "L_ts_m3_uniform_wrong", "m4": "L_ts_m4_frozen_pw_flags"}
SCORER_IDS = {"gpt2-large": "gpt2-large", "TinyStories-33M": "roneneldan/TinyStories-33M"}
BANKS = ("released", "m2", "m3", "m4f0")        # sample banks (m4f0: for (iv)'s KL only)
RHO_MODELS = ("released", "m0", "m2", "m3", "m4")
ALL_ARMS = ("m0", "m2", "m3", "m4")             # W2M-G (the default)
REDUCED_ARMS = ("m2", "m3")                     # W2M-GR: the released model, M2 and M3 only; (i), (ii), (iii)
GX_ARMS = G.GX_ARMS                             # W2M-GX: six checkpoints, new replay cells only
# where W2C stage 1's final-eval Gen-PPL (the sanity reference of (ii) at run seed >= 1) is saved: arm -> (notebook dir, run)
W2C_RESULTS = {"m2": ("w2c1_L_ts_m2", "L_ts_m2_frozen_pw_s{s}__result.json"),
               "m3": ("w2c1_L_ts_m3", "L_ts_m3_uniform_wrong_s{s}__result.json")}


def parse_arms(arms):
    """Canonical full W2M-G, reduced W2M-GR, or six-checkpoint W2M-GX arm set."""
    names = [a.strip().lower() for a in (arms.split(",") if isinstance(arms, str) else arms) if str(a).strip()]
    if set(names) == set(GX_ARMS) and len(names) == len(GX_ARMS):
        return GX_ARMS
    unknown = sorted(set(names) - set(ARM_DIRS))
    if unknown:
        sys.exit(f"unknown arm(s) {unknown}: choose from {sorted(ARM_DIRS)}")
    got = tuple(a for a in ALL_ARMS if a in names)
    if len(got) != len(names):
        sys.exit(f"duplicate arm in --arms {names}")
    if "m2" not in got or "m3" not in got:
        sys.exit("--arms must contain m2 and m3 (the crossed replay needs both)")
    if got not in (ALL_ARMS, REDUCED_ARMS):
        sys.exit(f"--arms {','.join(got)} is not a registered arm set: use {','.join(ALL_ARMS)} (W2M-G) or {','.join(REDUCED_ARMS)} (W2M-GR)")
    return got


def banks_for(arms):
    return tuple(b for b in BANKS if b != "m4f0" or "m4" in arms)


def rho_models(arms):
    return ("released", *arms)


def w2c_reference(seed):
    """W2M-GR: the sanity reference of (ii) for run seed s >= 1 -- W2C's final-eval GPT-2 large Gen-PPL of M2_s and M3_s,
    read from the repository's saved results. Returns ({arm: value}, {arm: repo-relative file}); a missing or unreadable file exits."""
    ref, src = {}, {}
    for arm, (nb, name) in W2C_RESULTS.items():
        rel = f"results/colab/{nb}/{name.format(s=seed)}"
        path = os.path.join(ROOT, *rel.split("/"))
        if not os.path.exists(path):
            sys.exit(f"missing {rel}: W2C's final-eval Gen-PPL is the sanity reference of (ii) at seed {seed}")
        try:
            with open(path) as f:
                v = float(json.load(f)["final_eval"]["genppl_gpt2-large"])
        except (KeyError, TypeError, ValueError) as e:
            sys.exit(f"{rel} has no final_eval['genppl_gpt2-large'] ({type(e).__name__}: {e})")
        if not (math.isfinite(v) and v > 0):
            sys.exit(f"{rel}: final_eval['genppl_gpt2-large'] = {v} is not a positive finite number")
        ref[arm], src[arm] = v, rel
    return ref, src


@dataclass
class Env:
    """Everything the run needs. `nets`: name -> generator in eval mode; `flag`: M4's FlagVector; `evaluate(ids, groups)`:
    evaluate_samples' dict for the (re-encoded) samples; `ckpt_sha256` / `runs`: provenance of the weights."""
    device: str
    tok: object
    bpe: object
    L0: torch.Tensor
    nets: dict
    flag: object
    clean_x: torch.Tensor
    evaluate: object
    ckpt_sha256: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)      # released assets, val data, device and runtime (pinned by manifest.json)

    def prefix_emb(self, b):
        return self.tok.encoder.wte(torch.full((b, 1), G.EOS, dtype=torch.long, device=self.device))

    def member(self, name, f=0.0):
        return G.Member(self.nets[name], self.flag if name == "m4" else None, f)


# ------------------------------------------------------------------------------------------------ io helpers
def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def save_npy(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_json(obj):
    """Canonical JSON content hash, independent of checkout line endings."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def fkey(f):
    return f"{float(f):g}"


# ------------------------------------------------------------------------------------------------ environment
def clean_windows(P, device):
    """The clean bank: the first `n_clean` (512) val windows, even-index validation items, as scripts/mech_eval_w2m.py."""
    return torch.from_numpy(examples("val", P.n_clean)).to(device)


def make_evaluate(bpe, device):
    from ncmr.eval_text import evaluate_samples
    return lambda ids, groups: evaluate_samples(ids, bpe.decode, groups, scorers=tuple(SCORER_IDS.values()), device=device)


def load_env(run_root, seed, device, P=G.Protocol(), arms=ALL_ARMS):
    from ncmr.ncm_ref_loader import load_ncm
    import copy
    B = load_ncm(os.path.join(ROOT, "artifacts", "tinystories"), device=device)
    tok = B.vqvae.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    released = B.net.eval()
    arms = parse_arms(arms)
    nets, flag, sha, runs = {"released": released}, None, {}, {"released": "released"}
    for n in arms:
        base, sd = (n.split("_s")[0], int(n.split("_s")[1])) if n in GX_ARMS else (n, seed)
        d = ARM_DIRS[base]
        p = os.path.join(run_root, f"{d}_s{sd}", "final.pt")
        if not os.path.exists(p):
            sys.exit(f"missing final checkpoint for {n}: {d}_s{sd}/final.pt")
        ck = torch.load(p, map_location=device, weights_only=False)
        nets[n] = copy.deepcopy(released)
        nets[n].load_state_dict(ck["model"])
        nets[n].eval()
        if "flag" in ck:
            flag = W.FlagVector(tok.embed_dim).to(device)
            flag.load_state_dict(ck["flag"])
        elif n == "m4":
            sys.exit("M4's checkpoint has no flag state")
        sha[n], runs[n] = sha256_file(p), f"{d}_s{sd}/final.pt"
        del ck
    L0 = B.ckpt_extra["inference"]["level0_unigram"].float().to(device)
    return Env(device, tok, B.pre_tokenizer, L0, nets, flag, clean_windows(P, device), make_evaluate(B.pre_tokenizer, device),
               sha, runs, run_identity(device))


def run_identity(device):
    """What else a cached cell depends on (Codex PR 16): the released generator, tokenizer, unigram (in generator.pt) and BPE
    files, the validation windows, and the device / GPU / runtime that draws the random streams (CPU and CUDA generators
    give different streams for the same seed, so a cross-device continuation would break the common random numbers)."""
    from ncmr.data import DATA_DIR
    art = os.path.join(ROOT, "artifacts", "tinystories")
    return {"released_sha256": {n: sha256_file(os.path.join(art, n)) for n in
                                ("config.yaml", "tokenizer.pt", "generator.pt", "bpe_vocab_8192.pickle", "bpe_merges_8192.pickle")},
            "val_sha256": sha256_file(os.path.join(DATA_DIR, "valid_bpe_8192.pickle")),
            "device": str(device), "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
            "torch": torch.__version__, "cuda": torch.version.cuda}


# ------------------------------------------------------------------------------------------------ cells
def cell_specs(P, replicate=False, fs=None, arms=ALL_ARMS):
    """The cells of (ii) / (iv) on seeds 0-7, or (replicate) the pure cells and the given f's on seeds 8-15. Without M4 in
    `arms` there are no M4 cells."""
    if tuple(arms) == GX_ARMS:
        if replicate:
            raise ValueError("W2M-GX has no replication sweep")
        return gx_cell_specs(P)
    sfx, seeds = ("_rep", list(P.replicate_seeds)) if replicate else ("", list(P.replay_seeds))
    mk = lambda key, kind, prefix, suffix, cut=None, f=None: {"key": key + sfx, "kind": kind, "prefix": prefix,
                                                              "suffix": suffix, "cut": cut, "f": f, "seeds": seeds,
                                                              "group_n": P.group_n}
    out = [mk("m2_m2", "pure", "m2", "m2"), mk("m3_m3", "pure", "m3", "m3")]
    if not replicate:
        for c in P.cuts:
            out += [mk(f"m2to3_c{c}", "crossed", "m2", "m3", c), mk(f"m3to2_c{c}", "crossed", "m3", "m2", c)]
    for f in ((P.f_sweep if fs is None else fs) if "m4" in arms else ()):
        out.append(mk(f"m4_f{fkey(f)}", "m4", "m4", "m4", None, float(f)))
    return out


def gx_cell_specs(P):
    """66 registered new cells plus six resampled pure sanity cells; levels 1..11 are explicit."""
    out = []
    def add(key, kind, assignment, cut=None, level=None):
        out.append({"key": key, "kind": kind, "assignment": assignment, "cut": cut, "level": level,
                    "seeds": list(P.replay_seeds), "group_n": P.group_n})
    for s in G.GX_SEEDS:
        for a in ("m2", "m3"):
            add(f"pure_{a}_s{s}", "pure", [f"{a}_s{s}"] * 11)
    for s in G.GX_SEEDS:
        for t in G.GX_SEEDS:
            if s == t:
                continue
            for a in ("m2", "m3"):
                for b in ("m2", "m3"):
                    add(f"X_{a}_s{s}_to_{b}_s{t}_c6", "cross_seed", [f"{a}_s{s}"] * 5 + [f"{b}_s{t}"] * 6, cut=6)
    for s in G.GX_SEEDS:
        for level in range(1, 6):
            for direction, base, swap in (("up", "m3", "m2"), ("down", "m2", "m3")):
                assignment = [f"{base}_s{s}"] * 11
                assignment[level - 1] = f"{swap}_s{s}"
                add(f"S_{direction}_s{s}_l{level}", "single_level", assignment, level=level)
        for c in (3, 5):
            for a, b in (("m2", "m3"), ("m3", "m2")):
                add(f"C_{a}to{b[1:]}_s{s}_c{c}", "crossed", [f"{a}_s{s}"] * (c - 1) + [f"{b}_s{s}"] * (12 - c), cut=c)
    return out


def model_fn(env, spec):
    if "assignment" in spec:
        return G.assigned([env.member(name) for name in spec["assignment"]])
    if spec["kind"] == "m4":
        return G.pure(env.member("m4", spec["f"]))
    if spec["kind"] == "pure":
        return G.pure(env.member(spec["prefix"]))
    return G.crossed(env.member(spec["prefix"]), env.member(spec["suffix"]), spec["cut"])


def summarise(res):
    """evaluate_samples' dict -> the per-cell metrics stored in result.json (log loss per group = log of the group's Gen-PPL)."""
    out = {k: res[k] for k in ("n", "entropy", "len_tokens", "dup_adj1_per_1k", "dup_adj2_per_1k")}
    for sc in G.SCORERS:
        per = [float(v) for v in res[f"genppl_{sc}_per_group"]]
        sd = float(np.std(per, ddof=1)) if len(per) > 1 else 0.0
        out[f"genppl_{sc}"] = {"mean": float(np.mean(per)), "sd": sd, "se": sd / math.sqrt(len(per)), "per_group": per,
                               "log_loss_per_group": [math.log(v) for v in per]}
    return out


def run_cell(env, P, spec):
    ids, groups = [], []
    fn = model_fn(env, spec)
    for sd in spec["seeds"]:
        o = G.decode_codes(env.tok, G.sample_codes(fn, env.tok, env.L0, P.group_n, sd, env.device))
        ids += o
        groups += [sd] * len(o)
    return {"spec": spec, "metrics": summarise(env.evaluate(G.reencode(env.bpe, ids), groups)), "sample_ids": ids,
            "groups": groups}


def get_cell(env, P, spec, cdir, log):
    path = os.path.join(cdir, spec["key"] + ".json")
    t0 = time.time()
    if os.path.exists(path):
        with open(path) as f:
            cell = json.load(f)
        if cell["spec"] != json.loads(json.dumps(spec)):
            sys.exit(f"cached cell {spec['key']} has a different spec than this run: use a fresh --run_dir")
        tag = "cached"
    else:
        cell = run_cell(env, P, spec)
        write_json(path, cell)
        tag = f"{time.time() - t0:.0f}s"
    m = cell["metrics"]
    log(f"cell {spec['key']} ({tag}): " + "  ".join(f"{sc} {m[f'genppl_{sc}']['mean']:.2f}" for sc in G.SCORERS)
        + f"  entropy {m['entropy']:.3f}  dup_adj1/1k {m['dup_adj1_per_1k']:.2f}")
    return cell


def public_cell(cell):
    return {"spec": cell["spec"], **cell["metrics"], "n_groups": len(set(cell["groups"])),
            **({"group_seeds": list(dict.fromkeys(cell["groups"]))} if "assignment" in cell["spec"] else {})}


# ------------------------------------------------------------------------------------------------ banks
def canonical_codes(env, x):
    with torch.no_grad(), torch.autocast(device_type=env.device, enabled=False):
        return env.tok.vq.z_to_idx(env.tok._encode_pre_quant(x))


def bank_codes(env, P, name, cdir, log):
    path = os.path.join(cdir, f"bank_{name}.npy")
    if os.path.exists(path):
        return torch.from_numpy(np.load(path).astype(np.int64)).to(env.device)
    member = env.member("m4", 0.0) if name == "m4f0" else env.member(name)
    t0 = time.time()
    codes = torch.cat([G.sample_codes(G.pure(member), env.tok, env.L0, P.group_n, sd, env.device) for sd in P.bank_seeds])
    save_npy(path, codes.cpu().numpy().astype(np.int16))
    log(f"bank {name} ({time.time() - t0:.0f}s): {codes.shape[0]} samples, seeds {list(P.bank_seeds)}")
    return codes


def build_corrupted_bank(env, P):
    """The plausible-wrong corrupted windows of the mech eval: `W.corrupted_contexts` from the RELEASED generator at
    seed 2026 with W1's rates on the clean bank."""
    rates = CorruptionCfg(mode="mech", mean_rate=P.corrupt_mean_rate, spread=P.corrupt_spread).rates(len(env.tok.vq.levels))
    return W.corrupted_contexts("plausible_wrong", env.tok, env.nets["released"], env.L0, env.prefix_emb, env.clean_x,
                                env.device, P.corrupt_seed, rates)


# ------------------------------------------------------------------------------------------------ (i), (iii), (iv) diagnostics
def analysis_i(env, P, canon, banks, cdir, log):
    arrs = {}
    for name, codes in (("clean", canon), *[(b, banks[b]) for b in ("released", "m2", "m3")]):
        path = os.path.join(cdir, f"js_{name}.npy")
        if not os.path.exists(path):
            save_npy(path, G.js_bank(env.nets["m2"], env.nets["m3"], env.tok, env.prefix_emb, codes, env.device,
                                     P.js_levels, P.eval_bs))
        arrs[name] = np.load(path)
    ratios = {b: G.bootstrap_ratio(arrs[b], arrs["clean"], P.n_boot, P.boot_seed) for b in ("released", "m2", "m3")}
    out = {"levels": list(P.js_levels), "J": {b: G.pool_js(a) for b, a in arrs.items()}, "ratios": ratios,
           "reading": G.read_i({b: {k: ratios[b][k] for k in ("r", "lo", "hi")} for b in ratios})}
    log(f"(i) J {out['J']['clean']['J']:.4g} clean; ratios " + " ".join(f"{b} {ratios[b]['r']:.2f}" for b in ratios)
        + f"; reading {out['reading']}")
    return out


def analysis_iii(env, P, canon, cdir, log, models=RHO_MODELS):
    paths = {n: os.path.join(cdir, f"rho_{n}.npy") for n in models}
    missing = [n for n in models if not os.path.exists(paths[n])]
    if missing:
        r = G.root_reliance({n: env.nets[n] for n in missing}, env.tok, env.L0, canon, env.device, P.n_roots, P.root_seed)
        for n in missing:
            save_npy(paths[n], r["rho"][n])
    rho = {n: np.load(paths[n]) for n in models}
    d = G.paired_ci(rho["m3"], rho["m2"], P.n_boot, P.boot_seed)
    out = {"n_roots": P.n_roots, "root_seed": P.root_seed, "unit": "nats per window (8 level-1 codes)",
           "rho": {n: {"mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(len(a)))} for n, a in rho.items()},
           "m3_minus_m2": d, "reading": G.read_iii(d["lo"], d["hi"])}
    log(f"(iii) rho " + " ".join(f"{n} {out['rho'][n]['mean']:.3f}" for n in models) + f"; reading {out['reading']}")
    return out


def analysis_iv_diag(env, P, canon, banks, cdir, log):
    path = os.path.join(cdir, "an_iv_diag.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    net, flag, vq = env.nets["m4"], env.flag, env.tok.vq
    fs, l0 = list(P.f_sweep), int(vq.levels[0])
    ctx = build_corrupted_bank(env, P)
    clean_in = vq.idx_to_ncm_input(canon)

    def summ(a, base):
        d = a - base
        return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(len(a))), "delta_vs_f0_mean": float(d.mean()),
                "delta_vs_f0_se": float(d.std(ddof=1) / math.sqrt(len(d)))}

    nll = {"clean": {}, "corrupted": {}}
    for f in fs:
        nll["clean"][fkey(f)] = G.nll_const_flag(net, flag, env.prefix_emb, clean_in, canon[:, l0:], f, env.device, P.eval_bs)
        nll["corrupted"][fkey(f)] = G.nll_const_flag(net, flag, env.prefix_emb, ctx["inputs"], ctx["targets"], f, env.device,
                                                     P.eval_bs)
    true = G.nll_const_flag(net, flag, env.prefix_emb, ctx["inputs"], ctx["targets"], 0.0, env.device, P.eval_bs,
                            flags=ctx["flags"])
    base = {k: nll[k][fkey(fs[0])] for k in nll}
    out = {"nll_per_window": {k: {f: summ(a, base[k]) for f, a in v.items()} for k, v in nll.items()}}
    out["nll_per_window"]["corrupted"]["true_flags"] = summ(true, base["corrupted"])
    kl = {}
    for name, inp in (("clean", clean_in), ("corrupted", ctx["inputs"]), ("m4_samples", vq.idx_to_ncm_input(banks["m4f0"]))):
        kl[name] = {fkey(f): v for f, v in G.kl_const_flag(net, flag, env.prefix_emb, inp, fs, env.device,
                                                           max(1, P.eval_bs // 2)).items()}
    out["kl_f0_to_f"] = kl
    log("(iv) diagnostics: KL(f=0||f) clean " + " ".join(f"f={k} {v:.4g}" for k, v in kl["clean"].items()))
    write_json(path, out)
    return out


# ------------------------------------------------------------------------------------------------ readings from cells
def log_losses(cell, sc):
    return cell["metrics"][f"genppl_{sc}"]["log_loss_per_group"]


def analysis_ii(P, cells, ref=None, source=None):
    """`ref` / `source`: the sanity reference of seed >= 1 (W2C's final eval, W2M-GR) and its files; None: W2M's (W2M-G)."""
    need = ["m2_m2", "m3_m3"] + [f"{p}_c{c}" for c in P.cuts for p in ("m2to3", "m3to2")]
    if any(k not in cells for k in need):
        return None
    sh = {sc: {str(c): G.shares(log_losses(cells["m2_m2"], sc), log_losses(cells["m3_m3"], sc),
                                log_losses(cells[f"m2to3_c{c}"], sc), log_losses(cells[f"m3to2_c{c}"], sc)) for c in P.cuts}
          for sc in G.SCORERS}
    reading = {sc: G.read_ii(sh[sc]["4"]["pi"], sh[sc]["6"]["pi"], sh[sc]["6"]["sigma"]) for sc in G.SCORERS} \
        if {4, 6} <= set(P.cuts) else None
    g = lambda k: cells[k]["metrics"]["genppl_gpt2-large"]
    sanity = G.sanity_ii(g("m2_m2")["mean"], g("m2_m2")["se"], g("m3_m3")["mean"], g("m3_m3")["se"], ref, source)
    out = {"delta_log_loss": {sc: sh[sc][str(P.cuts[0])]["delta"] for sc in G.SCORERS}, "shares": sh, "reading": reading,
           "reading_primary": None if reading is None else reading["gpt2-large"],
           ("sanity_vs_w2m" if ref is None else "sanity_vs_w2c_final_eval"): sanity}
    if 2 in P.cuts:
        a, b = sh["gpt2-large"]["2"]["pi"], sh["TinyStories-33M"]["2"]["pi"]
        out["direction1_gate"] = {"pi_2_gpt2-large": a, "pi_2_TinyStories-33M": b,
                                  "rule": "pi_2 >= 0.25 (GPT-2 large log loss) and pi_2 > 0 (TinyStories-33M)",
                                  "passes": G.gate_ii(a, b)}
    return out


def iv_rows(cells, sfx, fs):
    """Per f in `fs`: R(f) per scorer, entropy, dup and the overturn rule, on the cells with suffix `sfx`."""
    c22, c33 = cells.get("m2_m2" + sfx), cells.get("m3_m3" + sfx)
    if c22 is None or c33 is None:
        return None
    gm = lambda c, sc: c["metrics"][f"genppl_{sc}"]["mean"]
    rows = {}
    for f in fs:
        c4 = cells.get(f"m4_f{fkey(f)}{sfx}")
        if c4 is None:
            continue
        R = {sc: G.rescue(gm(c22, sc), gm(c4, sc), gm(c33, sc)) for sc in G.SCORERS}
        m4, m3 = c4["metrics"], c33["metrics"]
        rows[fkey(f)] = {"R": R, "genppl": {sc: gm(c4, sc) for sc in G.SCORERS}, "entropy": m4["entropy"],
                         "dup_adj1_per_1k": m4["dup_adj1_per_1k"],
                         "rule_met": G.iv_rule(R, m4["entropy"], m4["dup_adj1_per_1k"], m3["entropy"], m3["dup_adj1_per_1k"])
                         if f in (0.5, 1.0) else None,
                         "rescue_ge_0.8_both_scorers": bool(all(R[sc] >= 0.8 for sc in G.SCORERS))}
    return {"m2_m2": {sc: gm(c22, sc) for sc in G.SCORERS}, "m3_m3": {sc: gm(c33, sc) for sc in G.SCORERS},
            "m3_entropy": c33["metrics"]["entropy"], "m3_dup_adj1_per_1k": c33["metrics"]["dup_adj1_per_1k"], "f": rows}


def analysis_iv(P, cells, diag):
    seeds07 = iv_rows(cells, "", P.f_sweep)
    if seeds07 is None or len(seeds07["f"]) < len(P.f_sweep):
        return {"diagnostics": diag} if diag is not None else None
    sweep = [f for f in P.f_sweep if f in (0.5, 1.0)]
    met07 = {f: bool(seeds07["f"][fkey(f)]["rule_met"]) for f in sweep}
    rep, met_rep = None, None
    hit = [f for f in sweep if met07[f]]
    if hit:
        rep = iv_rows(cells, "_rep", hit)
        if rep is not None and len(rep["f"]) == len(hit):
            met_rep = {f: bool(rep["f"][fkey(f)]["rule_met"]) for f in hit}
    return {"seeds_0_7": seeds07, "seeds_8_15": rep, "rule_met_seeds_0_7": {fkey(f): v for f, v in met07.items()},
            "rule_met_seeds_8_15": None if met_rep is None else {fkey(f): v for f, v in met_rep.items()},
            "reading": G.read_iv(met07, met_rep),
            "note": "Astra A4's four outcome patterns are not classified here: the diagnostics are reported descriptively",
            "diagnostics": diag}


# ------------------------------------------------------------------------------------------------ driver
def protocol_block(env, P, seed, arms=ALL_ARMS):
    if tuple(arms) == GX_ARMS:
        return {**P.to_json(), "experiment": "W2M-GX", "amendment": "W2M-GXa", "run_seed": 0,
                "arms": list(arms), "runs": env.runs, "ckpt_sha256": env.ckpt_sha256,
                "cells": {"new_cells": 66, "pure_cells": 6, "total": 72, "groups_per_cell": len(P.replay_seeds),
                          "samples_per_group": P.group_n, "crossed_cuts": [3, 5, 6],
                          "cut_meaning": "prefix model samples levels 1..c-1, suffix model c..11",
                          "assignment_meaning": "assignment[j-1] samples level j, j=1..11"},
                "pure_source": "kappa pure means resampled, sanity within 3 SE of W2M-GR; delta and all other baselines from W2M-GR",
                "sampler": "ancestral, T=1, Gumbel-max, one generator per group seeded with the group seed (common random numbers)",
                "ci": "paired delete-one-group jackknife, 8 groups", "scorers": SCORER_IDS,
                "precision": "fp32, TF32 off, eval mode", "torch": torch.__version__, "identity": env.identity}
    return {**P.to_json(), "run_seed": seed, "arms": list(arms), "runs": env.runs, "ckpt_sha256": env.ckpt_sha256,
            "cells": {"groups_per_cell": len(P.replay_seeds), "samples_per_group": P.group_n,
                      "crossed_cuts": list(P.cuts), "cut_meaning": "prefix model samples levels 1..c-1, suffix model c..11"},
            "clean_bank": f'examples("val", {P.n_clean})',
            "corrupted_bank": f"plausible_wrong, seed {P.corrupt_seed}, W1 rates" if "m4" in arms else None,
            "sampler": "ancestral, T=1, Gumbel-max, one generator per group seeded with the group seed (common random numbers)",
            "ci": "percentile bootstrap over windows, both banks resampled", "scorers": SCORER_IDS,
            "precision": "fp32, TF32 off, eval mode", "torch": torch.__version__, "identity": env.identity}


def check_manifest(out_dir, env, P, seed, arms=ALL_ARMS):
    man = {"protocol": P.to_json(), "run_seed": seed, "arms": list(arms), "ckpt_sha256": env.ckpt_sha256, "identity": env.identity}
    path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != json.loads(json.dumps(man)):
            sys.exit("this run directory was made with a different protocol, seed, arm set, checkpoint, released asset, val data, "
                     "device or runtime (manifest.json): use a fresh --run_dir")
    else:
        write_json(path, man)


def run(env, out_dir, P=G.Protocol(), seed=0, log=print, arms=ALL_ARMS):
    arms = parse_arms(arms)
    if arms == GX_ARMS:
        return run_gx(env, out_dir, P, seed, log)
    if set(env.nets) != {"released", *arms}:
        sys.exit(f"the environment holds models {sorted(env.nets)} but --arms is {','.join(arms)}: it must hold exactly the released model and the arms")
    ref, src = (None, None) if seed == 0 else w2c_reference(seed)        # before anything is written: a missing file exits early
    os.makedirs(out_dir, exist_ok=True)
    cdir = os.path.join(out_dir, "cells")
    os.makedirs(cdir, exist_ok=True)
    check_manifest(out_dir, env, P, seed, arms)
    cells, an = {}, {}
    with_m4 = "m4" in arms

    def write(complete):
        res = {"protocol": protocol_block(env, P, seed, arms), "complete": complete,
               "cells": {k: public_cell(c) for k, c in cells.items()}}
        ii = analysis_ii(P, cells, ref, src)
        if ii is not None:
            res["ii"] = ii
            if "direction1_gate" in ii:
                res["direction1_gate"] = ii["direction1_gate"]
        for k in ("i", "iii", "iv"):
            if an.get(k) is not None:
                res[k] = an[k]
        write_json(os.path.join(out_dir, "result.json"), res)
        return res

    def do(spec):
        cells[spec["key"]] = get_cell(env, P, spec, cdir, log)
        write(False)

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    specs = cell_specs(P, arms=arms)
    for spec in specs:
        do(spec)
    canon = canonical_codes(env, env.clean_x)
    banks = {b: bank_codes(env, P, b, cdir, log) for b in banks_for(arms)}
    an["iii"] = analysis_iii(env, P, canon, cdir, log, rho_models(arms))
    write(False)
    an["i"] = analysis_i(env, P, canon, banks, cdir, log)
    write(False)
    if not with_m4:                                   # W2M-GR: no (iv), no M4 bank, no corrupted bank
        res = write(True)
        log(f"arms {','.join(arms)}: (iv) not run; direction-1 gate {res.get('direction1_gate', {}).get('passes')}")
        return res
    diag = analysis_iv_diag(env, P, canon, banks, cdir, log)
    an["iv"] = analysis_iv(P, cells, diag)
    write(False)
    hit = [f for f in (0.5, 1.0) if f in P.f_sweep and an["iv"]["seeds_0_7"]["f"][fkey(f)]["rule_met"]]
    if hit:
        log(f"(iv) overturn rule met on seeds 0-7 for f = {hit}: replicating on seeds {P.replicate_seeds[0]}-{P.replicate_seeds[-1]}")
        for spec in cell_specs(P, replicate=True, fs=hit, arms=arms):
            do(spec)
        an["iv"] = analysis_iv(P, cells, diag)
    res = write(True)
    log(f"(iv) reading {an['iv']['reading']}; direction-1 gate {res.get('direction1_gate', {}).get('passes')}")
    return res


def run_gx(env, out_dir, P, seed=0, log=print):
    """One resumable run loads all training seeds. Re-sampling pure cells avoids reusing cached GR noise."""
    if seed != 0 or set(env.nets) != {"released", *GX_ARMS}:
        sys.exit("W2M-GX needs --seed 0 and all six checkpoints M2_s / M3_s, s=1,2,3")
    if P.to_json() != G.GXProtocol().to_json():
        sys.exit("W2M-GX refuses a changed registered protocol")
    if not re.search(r"\bL4\b", env.identity.get("gpu") or ""):
        sys.exit("W2M-GX requires an L4 GPU")
    # Before sampling, validate GR references and their exact checkpoint identities.
    from analyze_w2mgx import load_references
    refs, bad = load_references(os.path.join(ROOT, "results", "colab"), env.ckpt_sha256)
    if bad:
        sys.exit("W2M-GX reference rejected: " + "; ".join(bad))
    env.identity = {**env.identity, "w2mgr_reference_sha256": {str(s): sha256_json(refs[s]) for s in G.GX_SEEDS}}
    os.makedirs(out_dir, exist_ok=True)
    cdir = os.path.join(out_dir, "cells")
    os.makedirs(cdir, exist_ok=True)
    check_manifest(out_dir, env, P, seed, GX_ARMS)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    cells = {}
    for spec in cell_specs(P, arms=GX_ARMS):
        cells[spec["key"]] = public_cell(get_cell(env, P, spec, cdir, log))
        write_json(os.path.join(out_dir, "result.json"), {"protocol": protocol_block(env, P, 0, GX_ARMS),
                                                        "complete": False, "cells": cells})
    from analyze_w2mgx import summarize
    res = {"protocol": protocol_block(env, P, 0, GX_ARMS), "complete": True, "cells": cells, **summarize(cells, refs)}
    write_json(os.path.join(out_dir, "result.json"), res)
    return res


@contextmanager
def gx_budget(out_dir, limit_seconds=210 * 60):
    """Cumulative cap across sessions, including load/score/cache time. Colab's alarm interrupts an unfinished cell.

    An unclosed session is charged through the next start, conservatively including disconnected time.
    Finished cells remain atomic and resumable; an interrupted cell is re-sampled from its original group seed.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "runtime_budget.json")
    now = time.time()
    ledger = {"spent_seconds": 0.0, "active_since": None, "limit_seconds": limit_seconds}
    if os.path.exists(path):
        with open(path) as f:
            ledger = json.load(f)
        if ledger.get("limit_seconds") != limit_seconds:
            sys.exit("runtime budget cap differs: use the registered 210-minute cap")
    spent = float(ledger["spent_seconds"])
    if ledger.get("active_since") is not None:
        spent += max(0.0, now - ledger["active_since"])
    remaining = limit_seconds - spent
    if remaining <= 0:
        write_json(path, {"spent_seconds": spent, "active_since": None, "limit_seconds": limit_seconds})
        sys.exit("W2M-GX cumulative 210 L4-minute cap reached")
    write_json(path, {"spent_seconds": spent, "active_since": now, "limit_seconds": limit_seconds})
    def expired(signum, frame):
        raise SystemExit("W2M-GX cumulative 210 L4-minute cap reached; unfinished cell discarded")
    old = None
    if hasattr(signal, "SIGALRM"):
        old = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        if old is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        write_json(path, {"spent_seconds": spent + max(0.0, time.time() - now), "active_since": None,
                          "limit_seconds": limit_seconds})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help="directory holding the run directories (final.pt)")
    ap.add_argument("--run_dir", required=True, help="output directory; writes result.json")
    ap.add_argument("--seed", type=int, default=0, help="run seed of the arms (directory suffix _s<seed>)")
    ap.add_argument("--arms", default=",".join(ALL_ARMS),
                    help="m0,m2,m3,m4 (W2M-G), m2,m3 (W2M-GR), or " + ",".join(GX_ARMS) + " (W2M-GX, all seeds)")
    a = ap.parse_args()
    arms = parse_arms(a.arms)
    device = pick_device()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if arms == GX_ARMS and a.seed != 0:
        ap.error("W2M-GX loads all training seeds in one run; use --seed 0")
    P = G.GXProtocol() if arms == GX_ARMS else G.Protocol()
    if arms == GX_ARMS:
        if not hasattr(signal, "SIGALRM"):
            ap.error("W2M-GX requires the Colab/Linux runtime for its enforced cumulative time cap")
        with gx_budget(a.run_dir):
            run(load_env(a.run_root, a.seed, device, arms=arms, P=P), a.run_dir, P, a.seed, arms=arms)
    else:
        run(load_env(a.run_root, a.seed, device, arms=arms, P=P), a.run_dir, P, a.seed, arms=arms)
    print("wrote result.json")


if __name__ == "__main__":
    main()
