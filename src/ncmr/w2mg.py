"""W2M-G (docs/preregistration.md, "W2M-G -- eval-only: shared context banks, crossed replay, root reliance, M4 flag
routing"): the pure pieces of scripts/w2mg_eval.py.

  sample_codes            level-by-level ancestral sampler over several models (prefix model for levels < c, suffix model
                          from c on), T = 1, Gumbel-max, one generator per sample group, noise drawn in level order with
                          fixed shapes (common random numbers); optional constant flag f * w on every level's input
  decode_codes / reencode as scripts/train_gen.py's `draw_samples` / `_sample_metrics`
  js_bank / pool_js       (i) Jensen-Shannon divergence of two models' predictive distributions on a bank's codes
  bootstrap_ratio         (i) window bootstrap of J_b / J_clean
  root_reliance           (iii) rho per window: NLL of level 1 under drawn roots minus under the true root
  nll_const_flag / kl_const_flag   (iv) diagnostics of the constant-flag sweep
  shares / jackknife      (ii) prefix, suffix and interaction shares with the group jackknife
  read_i / read_ii / gate_ii / read_iii / iv_rule / read_iv   the registered readings, thresholds exactly as written

Every forward, sample and score is fp32 with TF32 off and the modules in eval mode (`fp32_eval`).
"""
from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import cached_decode as CD
from . import w2m as W
from .data import EOS, PAD, cut_at_eos
from .train import fp32_eval

SCORERS = ("gpt2-large", "TinyStories-33M")              # keys of evaluate_samples: genppl_<scorer>
W2M_GENPPL = {"m2": 42.62, "m3": 21.69}                   # W2M's GPT-2 large Gen-PPL of M2 and M3 (sanity check of (ii))


@dataclass(frozen=True)
class Protocol:
    """The registered counts and seeds (defaults); tests shrink the counts."""
    n_clean: int = 512                                    # clean bank = examples("val", 512)
    group_n: int = 256                                    # samples per group (cells and sample banks)
    bank_seeds: tuple = (900, 901)                        # sample banks: 2 groups of 256, the same for every model
    replay_seeds: tuple = tuple(range(8))                 # (ii), (iv) cells
    replicate_seeds: tuple = tuple(range(8, 16))          # (iv) replication, only if the overturn rule is met
    cuts: tuple = (2, 4, 6)
    js_levels: tuple = (2, 3, 4, 5)
    n_boot: int = 2000
    boot_seed: int = 0
    root_seed: int = 2027
    n_roots: int = 8
    corrupt_seed: int = 2026                              # plausible-wrong corrupted bank (the mech eval's)
    corrupt_mean_rate: float = 0.5                        # W1's rates: CorruptionCfg(mode="mech", mean_rate=.5, spread=.5)
    corrupt_spread: float = 0.5
    f_sweep: tuple = (0.0, 0.5, 1.0)
    eval_bs: int = 32                                     # window batch of the bank forwards (no big vocabulary tensor beyond it)

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


# ====================================================================================================== sampler
class Member:
    """One sampling model: the network, and optionally a FlagVector with a constant flag value `f` that adds f * w to
    every level-1..K-1 input (M4's constant-flag sweep). f = 0 with a FlagVector adds exactly 0."""

    def __init__(self, net, flag: "W.FlagVector | None" = None, f: float = 0.0):
        if flag is None and f != 0.0:
            raise ValueError("a constant flag needs a FlagVector")
        self.net, self.flag, self.f = net, flag, float(f)

    def add_flag(self, x_in: torch.Tensor) -> torch.Tensor:
        if self.flag is None:
            return x_in
        return self.flag(x_in, torch.full(x_in.shape[:2], self.f, dtype=x_in.dtype, device=x_in.device))


def pure(member: Member):
    """model_for_level of a pure cell (the same model at every level)."""
    return lambda j: member


def crossed(prefix: Member, suffix: Member, cut: int):
    """model_for_level of a crossed cell: `prefix` samples levels 1..cut-1, `suffix` levels cut..K-1."""
    return lambda j: prefix if j < cut else suffix


@torch.no_grad()
def sample_codes(model_for_level, tok, L0: torch.Tensor, n: int, seed: int, device: str, prefix_id: int = EOS,
                 trace: list | None = None) -> torch.Tensor:
    """Ancestral samples (n, S) of one group. Level 0 ~ the unigram `L0`; level j = 1..K-1 from the level-j call of
    `model_for_level(j)` (`cached_decode._level_call`: it attends to the prefix and itself, so its logits are the
    teacher-forced ones) at T = 1 by Gumbel-max. ONE generator seeded with `seed` is drawn in level order: the level-0
    multinomial, then at each level a uniform tensor of the logits' shape, whichever model is used there, so any two
    samplers with the same seed see the same level-0 codes and the same noise (common random numbers). `trace`
    (tests): receives each level's logits (n, l_j, V)."""
    vq, lv = tok.vq, [int(l) for l in tok.vq.levels]
    g = torch.Generator(device=device).manual_seed(int(seed))
    members = [model_for_level(j) for j in range(1, len(lv))]
    nets = {id(m.net): m.net for m in members}
    pre = torch.full((1, 1), int(prefix_id), dtype=torch.long, device=device)
    with ExitStack() as st:
        for net in nets.values():
            st.enter_context(fp32_eval(net, device))
        with fp32_eval(tok, device):
            caches = {i: CD.PrefixCache(net, *CD.prefix_inputs(net, pre)).expand(n) for i, net in nets.items()}
            codes = [torch.multinomial(L0.float(), n * lv[0], replacement=True, generator=g).view(n, lv[0])]
            z_hat = torch.zeros(n, lv[-1], vq.embed_dim, device=device)
            for j in range(1, len(lv)):
                z_hat, x_in = vq.get_next_autoregressive_input(j - 1, z_hat, vq.embeddings[j - 1](codes[-1]))
                m = members[j - 1]
                lg = CD._level_call(m.net, caches[id(m.net)], m.add_flag(x_in), j).float()
                if trace is not None:
                    trace.append(lg.detach().clone())
                u = torch.rand(lg.shape, device=device, generator=g).clamp_min(1e-20)
                codes.append((lg - torch.log(-torch.log(u))).argmax(-1))
    return torch.cat(codes, 1)


@torch.no_grad()
def decode_codes(tok, codes: torch.Tensor) -> list[list[int]]:
    """As `draw_samples`: decoder argmax, cut at EOS, PAD dropped (fp32, autocast off)."""
    with fp32_eval(tok, codes.device.type):
        ids = tok.decoder(tok.vq.idx_to_z_q(codes)).argmax(-1)
    return [[t for t in cut_at_eos(r) if t != PAD] for r in ids.tolist()]


def reencode(bpe, ids: list[list[int]]) -> list[list[int]]:
    """As `_sample_metrics`: the token statistics are computed on the BPE re-encoding of the decoded text."""
    return [bpe.encode(bpe.decode(s)) for s in ids]


# ====================================================================================================== (i) JS divergence
def js_from_logits(la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence (nats) of softmax(la) and softmax(lb) over the last dim, in float64: symmetric,
    >= 0, 0 for identical logits."""
    lp, lq = la.double().log_softmax(-1), lb.double().log_softmax(-1)
    lm = torch.logaddexp(lp, lq) - math.log(2.0)
    p, q = lp.exp(), lq.exp()
    zero = torch.zeros_like(p)
    kp = torch.where(p > 0, p * (lp - lm), zero).sum(-1)
    kq = torch.where(q > 0, q * (lq - lm), zero).sum(-1)
    return (0.5 * (kp + kq)).clamp_min(0.0)


def level_span(levels, j: int) -> tuple[int, int]:
    """[lo, hi) of level j's positions among the logits (levels 1..K-1 in order)."""
    lo = sum(levels[1:j])
    return lo, lo + levels[j]


@torch.no_grad()
def js_bank(net_a, net_b, tok, prefix_emb, codes: torch.Tensor, device: str, js_levels=(2, 3, 4, 5), bs: int = 32):
    """(n, len(js_levels)) float64: per window and level, the mean over the level's positions of JS(P_a, P_b), both models
    teacher-forced on `codes` (n, S), zero flags. Batches of `bs` windows, so no full-vocabulary tensor spans more."""
    levels = [int(l) for l in tok.vq.levels]
    rows = []
    for s in range(0, codes.shape[0], bs):
        inp = tok.vq.idx_to_ncm_input(codes[s:s + bs])
        la = W.forward_logits(net_a, prefix_emb, inp, device, bs=bs)
        lb = W.forward_logits(net_b, prefix_emb, inp, device, bs=bs)
        per = []
        for j in js_levels:
            lo, hi = level_span(levels, j)
            per.append(js_from_logits(la[:, lo:hi], lb[:, lo:hi]).mean(1))
        rows.append(torch.stack(per, 1).cpu())
        del la, lb
    return torch.cat(rows).numpy()


def pool_js(per_window: np.ndarray) -> dict:
    """J per level (mean over positions and windows) and J_b = their mean over the registered levels."""
    per_level = per_window.mean(0)
    return {"per_level": [float(v) for v in per_level], "J": float(per_level.mean())}


def bootstrap_ratio(js_b: np.ndarray, js_clean: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """r_b = J_b / J_clean with a 95% percentile CI from a window bootstrap that resamples BOTH banks (independent
    window draws per resample, one default_rng(seed)); also the per-level ratios."""
    rng = np.random.default_rng(seed)
    ib = rng.integers(0, js_b.shape[0], (n_boot, js_b.shape[0]))
    ic = rng.integers(0, js_clean.shape[0], (n_boot, js_clean.shape[0]))
    jb, jc = js_b[ib].mean(1), js_clean[ic].mean(1)                      # (n_boot, L)
    pt_b, pt_c = js_b.mean(0), js_clean.mean(0)

    def ci(boot, point):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"r": float(point), "lo": float(lo), "hi": float(hi)}

    out = ci(jb.mean(1) / jc.mean(1), pt_b.mean() / pt_c.mean())
    out["per_level"] = [ci(jb[:, k] / jc[:, k], pt_b[k] / pt_c[k]) for k in range(js_b.shape[1])]
    return out


# ====================================================================================================== (iii) root reliance
def draw_roots(L0: torch.Tensor, n: int, k: int = 8, seed: int = 2027) -> torch.Tensor:
    """(n, k) roots from the unigram: a CPU generator seeded `seed`, so the draws are the same for every model (and device)."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.multinomial(L0.detach().cpu().float(), n * k, replacement=True, generator=g).view(n, k)


@torch.no_grad()
def level1_nll(net, tok, roots: torch.Tensor, targets: torch.Tensor, device: str, prefix_id: int = EOS,
               bs: int = 256) -> torch.Tensor:
    """(n,) NLL (nats) of `targets` (n, l_1) under level 1's call with the input rebuilt from the level-0 codes `roots`
    (n,) -- zero flags, fp32."""
    vq = tok.vq
    out = []
    with fp32_eval(net, device), fp32_eval(tok, device):
        pre = torch.full((1, 1), int(prefix_id), dtype=torch.long, device=device)
        cache = CD.PrefixCache(net, *CD.prefix_inputs(net, pre))
        for s in range(0, roots.shape[0], bs):
            r, t = roots[s:s + bs].to(device), targets[s:s + bs].to(device)
            z_hat = torch.zeros(r.shape[0], int(vq.levels[-1]), vq.embed_dim, device=device)
            _, x_in = vq.get_next_autoregressive_input(0, z_hat, vq.embeddings[0](r[:, None]))
            lg = CD._level_call(net, cache.expand(r.shape[0]), x_in, 1).float()
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1), reduction="none").view(t.shape)
            out.append(ce.sum(1).cpu())
    return torch.cat(out)


def root_reliance(nets: dict, tok, L0: torch.Tensor, canon: torch.Tensor, device: str, n_roots: int = 8,
                  seed: int = 2027) -> dict:
    """(iii) per window rho = mean over the `n_roots` drawn roots of NLL_L1(drawn root) - NLL_L1(true root), level 1's
    canonical targets kept. The roots are drawn once, so every model sees the same draws. Returns {"roots": (n, k),
    "rho": {name: (n,) float64 array}}."""
    assert int(tok.vq.levels[0]) == 1, "root reliance is defined for one level-0 code"
    n = canon.shape[0]
    roots = draw_roots(L0, n, n_roots, seed)
    l1 = int(tok.vq.levels[1])
    tgt = canon[:, 1:1 + l1]
    rho = {}
    for name, net in nets.items():
        true = level1_nll(net, tok, canon[:, 0], tgt, device)
        drawn = torch.stack([level1_nll(net, tok, roots[:, d], tgt, device) for d in range(n_roots)], 1)
        rho[name] = (drawn.double() - true.double()[:, None]).mean(1).numpy()
    return {"roots": roots, "rho": rho}


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of a - b over windows with a 95% percentile window-bootstrap CI (paired: the same windows in both)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, d.shape[0], (n_boot, d.shape[0]))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(d.shape[0])), "lo": float(lo), "hi": float(hi)}


# ====================================================================================================== (iv) diagnostics
@torch.no_grad()
def nll_const_flag(net, flag, prefix_emb, inputs, targets, f, device: str, bs: int = 32, flags=None) -> np.ndarray:
    """(n,) clean-label NLL per window with the constant flag f on every input position (or the given `flags`)."""
    fl = torch.full(inputs.shape[:2], float(f), device=inputs.device) if flags is None else flags
    return W.input_nll(W.forward_logits(net, prefix_emb, inputs, device, flag, fl, bs), targets).double().cpu().numpy()


@torch.no_grad()
def kl_const_flag(net, flag, prefix_emb, inputs, fs, device: str, bs: int = 16) -> dict:
    """{f: mean over windows and the positions of levels 1..K-1 of KL(P_{f=0} || P_f)} for the constant flags `fs`."""
    tot = {f: 0.0 for f in fs}
    n = inputs.shape[0]
    for s in range(0, n, bs):
        inp = inputs[s:s + bs]
        p = W.forward_logits(net, prefix_emb, inp, device, bs=bs).log_softmax(-1)
        pr = p.exp()
        for f in fs:
            q = W.forward_logits(net, prefix_emb, inp, device, flag, torch.full(inp.shape[:2], float(f), device=inp.device),
                                 bs=bs).log_softmax(-1)
            kl = torch.where(pr > 0, pr * (p - q), torch.zeros_like(pr)).sum(-1)
            tot[f] += float(kl.double().sum())
            del q, kl
        del p, pr
    return {f: tot[f] / (n * inputs.shape[1]) for f in fs}


# ====================================================================================================== (ii) shares
def _shares(l22, l33, l23, l32):
    """prefix share pi, suffix share sigma, interaction 1 - pi - sigma from the cells' mean log losses."""
    delta = l22 - l33
    if delta == 0:
        return (math.nan, math.nan, math.nan)
    pi, sg = (l23 - l33) / delta, (l32 - l33) / delta
    return (pi, sg, 1.0 - pi - sg)


def shares(l22, l33, l23, l32) -> dict:
    """Shares of one cut from the per-group log losses (arrays over the same groups, in the same order) of (M2, M2),
    (M3, M3), (M2 -> M3) and (M3 -> M2) with the jackknife SE over groups (leave one group out)."""
    a = [np.asarray(v, dtype=np.float64) for v in (l22, l33, l23, l32)]
    G = a[0].shape[0]
    assert all(v.shape == (G,) for v in a) and G >= 2
    est = _shares(*[v.mean() for v in a])
    loo = np.array([_shares(*[np.delete(v, i).mean() for v in a]) for i in range(G)])          # (G, 3)
    se = np.sqrt((G - 1) / G * ((loo - loo.mean(0)) ** 2).sum(0))
    out = {"delta": float(a[0].mean() - a[1].mean())}
    for k, name in enumerate(("pi", "sigma", "interaction")):
        out[name], out[name + "_se"] = float(est[k]), float(se[k])
    return out


# ====================================================================================================== readings
def read_i(ratios: dict) -> str:
    """(i) ratios: {bank: {"r", "lo", "hi"}} for the three generated banks. holds: every r >= 3 and every lower bound
    > 1.2; overturned: every upper bound <= 1.2; else mixed."""
    v = list(ratios.values())
    if all(x["r"] >= 3 and x["lo"] > 1.2 for x in v):
        return "holds"
    if all(x["hi"] <= 1.2 for x in v):
        return "overturned"
    return "mixed"


def read_ii(pi4: float, pi6: float, sigma6: float) -> str:
    """(ii) early-state selection: pi_4 >= 0.5 and pi_6 >= 0.75; conditional rendering: sigma_6 >= 0.8; else
    distributed. (Both met at once -- possible only with a negative interaction -- is returned as 'both'.)"""
    early, cond = (pi4 >= 0.5 and pi6 >= 0.75), sigma6 >= 0.8
    if early and cond:
        return "both"
    return "early-state selection" if early else "conditional rendering" if cond else "distributed"


def gate_ii(pi2_large: float, pi2_ts: float) -> bool:
    """Direction-1 (L1R) run gate: pi_2 >= 0.25 under GPT-2 large log loss AND pi_2 > 0 under TinyStories-33M."""
    return bool(pi2_large >= 0.25 and pi2_ts > 0)


def sanity_ii(g22: float, se22: float, g33: float, se33: float, ref: dict | None = None, source: dict | None = None) -> dict:
    """G(M2, M2), G(M3, M3) (GPT-2 large) within 3 SE of the reference {"m2", "m3"}: W2M's 42.62 and 21.69 by default
    (W2M-G); W2M-GR passes W2C's final-eval Gen-PPL of the seed (`source`: where each value was read, recorded). A miss is
    reported, not fatal."""
    r = W2M_GENPPL if ref is None else ref
    z22, z33 = (g22 - r["m2"]) / se22, (g33 - r["m3"]) / se33
    out = {"G_m2m2": g22, "z_m2m2": float(z22), "G_m3m3": g33, "z_m3m3": float(z33), "ref": dict(r),
           "within_3se": bool(abs(g22 - r["m2"]) <= 3 * se22 and abs(g33 - r["m3"]) <= 3 * se33)}
    if source is not None:
        out["source"] = dict(source)
    return out


def read_iii(lo: float, hi: float) -> str:
    """(iii) paired 95% CI of rho(M3) - rho(M2): holds if it lies above 0, overturned if below, else neither."""
    return "holds" if lo > 0 else "overturned" if hi < 0 else "neither"


def iv_rule(R: dict, entropy: float, dup: float, m3_entropy: float, m3_dup: float) -> bool:
    """(iv) overturn rule for one f: R(f) >= 0.8 under both scorers, entropy >= M3's - 0.05, duplicates per 1k <= M3's + 2."""
    return bool(all(R[s] >= 0.8 for s in SCORERS) and entropy >= m3_entropy - 0.05 and dup <= m3_dup + 2)


def rescue(g22: float, g4: float, g33: float) -> float:
    """R(f) = (G(M2, M2) - G(M4, f)) / (G(M2, M2) - G(M3, M3)) for one scorer."""
    return (g22 - g4) / (g22 - g33)


def read_iv(met_0_7: dict, met_8_15: dict | None) -> str:
    """(iv) met_0_7: {f: rule met on seeds 0-7} for f in {0.5, 1}; met_8_15: {f: rule met on seeds 8-15} for the f's
    that were run there (None: not run yet). holds: no f meets the rule on 0-7; overturned: some f meets it on 0-7 and
    again on 8-15; mixed: only on 0-7."""
    hit = [f for f, v in met_0_7.items() if v]
    if not hit:
        return "holds"
    if met_8_15 is None:
        return "pending replication"
    return "overturned" if any(met_8_15.get(f) for f in hit) else "mixed"
