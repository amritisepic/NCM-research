"""W2M (docs/preregistration.md, "W2M -- tier-L mechanism arms"): corruption arms M0-M7 for the text trainer, and the
eval-only mechanism measures of scripts/mech_eval_w2m.py.

Training side (`MechArm`, used by scripts/train_gen.py when `corruption.mode: mech`):
  * ONE mask stream: its own generator, drawn once per optimizer step for the whole batch (so the slots do not depend on
    the micro-batch partition) and sliced per micro-batch. The draw order is fixed: first the per-row clean-row key,
    then one uniform per slot of levels 0..K-2. Every arm that uses W1's rates therefore corrupts exactly the same
    slots; M7 thresholds the same uniforms at its own rates.
  * 25% of the rows of every batch (exactly round(0.25 B), the rows with the smallest clean-row keys) have no masked
    slot, so that all-zero flags are trained.
  * Replacement / proposal draws use their own generator; flag shuffling a third. All three are in the checkpoint.
  * Proposals (T = 1, fp32, TF32 off, eval mode, no grad): uniform (M0, M7), the current policy (M1), the frozen
    released generator with the canonical code excluded (M2, M4, M5), uniform-wrong (M3), the current policy given
    the level-0 codes actually used (M6).
  * Flags (M4 truthful, M5 shuffled within level per row): the sampled mask propagated with the tokenizer's own
    up-sampling to full length (the support of the interpolation; the nearest-repeat block for a learned level-0
    up-sampler) and reduced with the same area down-sampling that builds the generator's inputs; they enter as
    `f * w` added to the code inputs, `w` zero-initialised (`FlagVector`).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .train import fp32_eval

# arm -> (proposal kind, flag mode)
ARMS = {
    "m0": ("uniform", None),
    "m1": ("policy", None),
    "m2": ("frozen_wrong", None),
    "m3": ("uniform_wrong", None),
    "m4": ("frozen_wrong", "truthful"),
    "m5": ("frozen_wrong", "shuffled"),
    "m6": ("policy_used_parent", None),
    "m7": ("uniform", None),
}
# L1R (docs/preregistration.md, "L1R"): M2's proposals exactly, plus the target reassignment of src/ncmr/l1r.py. Kept out of
# `ARMS` so that the M0-M7 sets enumerated by the W2M tests and notebooks are unchanged; `MechArm` accepts both.
L1R_ARMS = {"m2r": ("frozen_wrong", None)}
ALL_ARMS = {**ARMS, **L1R_ARMS}
# A3's mean logged per-level disagreement over its logged steps <= 600 (as registered; tests recompute it from
# results/colab/w1_L_ts_onpol/L_ts_onpol_s0__metrics.jsonl)
A3_DISAGREEMENT = (0.735, 0.691, 0.601, 0.505, 0.398, 0.304, 0.239, 0.103, 0.171, 0.167, 0.145, 0.0)
PROPOSAL_FORWARDS = {"uniform": 0, "uniform_wrong": 0, "policy": 1, "policy_used_parent": 1, "frozen_wrong": 1}


def m7_level_rates(d, vocab_sizes):
    """Uniform replacement at probability p draws the canonical code again with probability 1/V, so the realized
    disagreement is p (1 - 1/V): p_k = d_k / (1 - 1/V_k)."""
    assert len(d) == len(vocab_sizes)
    return tuple(min(1.0, float(dk) / (1.0 - 1.0 / float(v))) for dk, v in zip(d, vocab_sizes))


class FlagVector(nn.Module):
    """The only architecture change (M4 / M5): a learned vector `w`, zero at initialisation, added to the code inputs
    as `f * w` (f in [0, 1] per input position). With w = 0 the inputs, hence the logits, are the released model's."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim))

    def forward(self, inputs: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        return inputs + flags[..., None].to(inputs.dtype) * self.w.to(inputs.dtype)


def _resi_halo(vq, k: int) -> int:
    """Half-width of level k's residual convolution (quant_resi, applied after the up-sampling in idx_to_ncm_input):
    it spreads a replaced code's footprint by this many full-length positions on each side (Codex PR 12)."""
    if not vq.quant_resi_enabled:
        return 0
    conv = vq.quant_resi[k / max(vq._qr_num_levels - 1, 1)]
    return int(conv.kernel_size[0]) // 2


def _footprint(vq, m: torch.Tensor, k: int, L: int) -> torch.Tensor:
    """(B, l_k) bool replaced-code mask of level k -> (B, L) bool: positions of the full-length reconstruction that
    receive a contribution from a replaced code, under the tokenizer's own up-sampling of level k followed by its
    residual convolution."""
    lk = m.shape[1]
    mf = m.float()
    if k == 0 and vq.learned_l1_sampling:
        # learned level-0 up-sampler (initialised as nearest-repeat): the code's own block
        fp = mf.repeat_interleave(L // lk, dim=1) > 0
    elif vq.learned_all_sampling:
        raise NotImplementedError("learned_all_sampling has no scalar footprint")
    else:
        fp = vq._up_to_full(mf[..., None], k, L)[..., 0] > 0
    r = _resi_halo(vq, k)
    if r:
        fp = F.max_pool1d(fp.float()[:, None], kernel_size=2 * r + 1, stride=1, padding=r)[:, 0] > 0
    return fp


@torch.no_grad()
def mask_flags(vq, masks: list, L: int | None = None) -> torch.Tensor:
    """Truthful flags (B, S - l_0), laid out like the generator's inputs (levels 1..K-1). For the level-k input at
    position q: the share of q's span of the full-length sequence whose cumulative reconstruction (levels < k)
    contains a replaced ancestor. Built from the sampled mask only (never from `used != canonical`)."""
    levels = [int(l) for l in vq.levels]
    L = levels[-1] if L is None else L
    contaminated = torch.zeros(masks[0].shape[0], L, dtype=torch.bool, device=masks[0].device)
    out = []
    for k in range(len(levels) - 1):
        contaminated = contaminated | _footprint(vq, masks[k], k, L)
        out.append(F.interpolate(contaminated.float()[:, None], size=(levels[k + 1],), mode="area")[:, 0])
    return torch.cat(out, 1)


def shuffle_within_level(flags: torch.Tensor, sizes: list, gen: torch.Generator) -> torch.Tensor:
    """M5: each row's flags permuted within each level by an independent RNG (same per-row, per-level multiset)."""
    out = []
    for blk in torch.split(flags, sizes, dim=1):
        perm = torch.rand(blk.shape, generator=gen, device=blk.device).argsort(1)
        out.append(blk.gather(1, perm))
    return torch.cat(out, 1)


class MaskPlan:
    """The slots of one optimizer step: `clean` (B,) bool rows left clean; `masks` one (B, l_k) bool tensor per level."""

    def __init__(self, clean, masks, row0: int = 0):
        # row0: this slice's first row within the optimizer step's full batch (M2R's plan is keyed by (step, row))
        self.clean, self.masks, self.row0 = clean, masks, row0

    def rows(self, a: int, b: int) -> "MaskPlan":
        return MaskPlan(self.clean[a:b], [m[a:b] for m in self.masks], self.row0 + a)


class MechArm:
    def __init__(self, arm: str, tok, net, L0, prefix_emb, device: str, seed: int, rates, clean_frac: float = 0.25,
                 policy_temp: float = 1.0, frozen=None, fwd=None, dry: bool = False):
        assert arm in ALL_ARMS, arm
        self.arm, (self.prop, self.flag_mode) = arm, ALL_ARMS[arm]
        self.tok, self.net, self.L0, self.prefix_emb, self.device = tok, net, L0, prefix_emb, device
        self.fwd = fwd if fwd is not None else net
        self.levels = [int(l) for l in tok.vq.levels]
        self.K, self.S, self.l0 = len(self.levels), sum(self.levels), self.levels[0]
        self.vocab = [int(v) for v in tok.vq.vqvae_vocab_sizes]
        self.rates, self.clean_frac, self.policy_temp = [float(r) for r in rates], float(clean_frac), float(policy_temp)
        assert len(self.rates) == self.K and self.rates[-1] == 0.0, "the finest level is never corrupted"
        # dry: an arm that only replays the mask / replacement generators (src/ncmr/l1r.py); it never calls `proposals`
        if self.prop == "frozen_wrong" and not dry:
            assert frozen is not None, "M2 / M4 / M5 need the frozen copy of the released generator"
        self.frozen, self.dry = frozen, dry
        self.l1r = None            # M2R: the L1RPlan (set by scripts/train_gen.py); None for every other arm
        base = int(seed) * 7919
        mk = lambda off: torch.Generator(device=device).manual_seed(base + off)
        self.mask_gen, self.repl_gen, self.flag_gen = mk(2), mk(104729), mk(15485863)
        z = [0.0] * self.K
        self.stats = {"n": 0, "n_clean": 0, "sum_all": list(z), "cnt_all": list(z), "sum_corr": list(z), "cnt_corr": list(z)}

    # ------------------------------------------------------------------ properties
    @property
    def proposal_forwards(self) -> int:
        return PROPOSAL_FORWARDS[self.prop]

    @staticmethod
    def frozen_copy(net):
        """Frozen copy of the released generator; call before any checkpoint restore."""
        f = copy.deepcopy(net).eval()
        for p in f.parameters():
            p.requires_grad_(False)
        return f

    # ------------------------------------------------------------------ the mask stream
    def draw_plan(self, B: int) -> MaskPlan:
        g, dev = self.mask_gen, self.device
        key = torch.rand(B, generator=g, device=dev)
        n_clean = int(round(self.clean_frac * B))
        clean = torch.zeros(B, dtype=torch.bool, device=dev)
        if n_clean:
            clean[key.argsort()[:n_clean]] = True
        masks = []
        for k, lk in enumerate(self.levels):
            if k == self.K - 1:
                masks.append(torch.zeros(B, lk, dtype=torch.bool, device=dev))
                continue
            u = torch.rand(B, lk, generator=g, device=dev)       # drawn at every level in every arm
            masks.append((u < self.rates[k]) & ~clean[:, None])
        return MaskPlan(clean, masks)

    # ------------------------------------------------------------------ proposals
    @torch.no_grad()
    def draw_root(self, canon):
        """Frozen-wrong level 0 (M2 / M4 / M5 / M2R): (B, l0) replacement codes ~ the unigram without the canonical code.
        Shared by `proposals` and the M2R replay (src/ncmr/l1r.py), so both draw identically from `repl_gen`."""
        B, l0 = canon.shape[0], self.l0
        probs0 = self.L0[None].expand(B * l0, -1).clone()
        probs0.scatter_(1, canon[:, :l0].reshape(-1, 1), 0.0)               # level 0: unigram without the canonical code
        return torch.multinomial(probs0, 1, generator=self.repl_gen).view(B, l0)

    @torch.no_grad()
    def draw_fine(self, probs):
        """Frozen-wrong levels >= 1: one sample per position from probs (B, S - l0, V) (fp32), from `repl_gen`."""
        return torch.multinomial(probs.flatten(0, 1), 1, generator=self.repl_gen).view(probs.shape[:-1])

    @torch.no_grad()
    def skip_fine(self, B: int, V: int):
        """Advance `repl_gen` exactly as `draw_fine` does for B rows over a V-wide output, without the generator: the
        draw consumes the generator by the probability tensor's shape (and dtype), not its values, so a dummy of the
        same shape / dtype / device is enough (the M2R replay checks this against the real stream at every micro-batch)."""
        self.draw_fine(torch.full((B, self.S - self.l0, V), 1.0 / V, dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def canonical(self, x):
        # the canonical codes are the labels: fp32 with TF32 off, like every teacher used as a target (audit 8, A8-2)
        with fp32_eval(self.tok, self.device):
            return self.tok.vq.z_to_idx(self.tok._encode_pre_quant(x))

    def _logits(self, model, fwd, ctx_codes):
        with fp32_eval(model, self.device):
            z_in = self.tok.vq.idx_to_ncm_input(ctx_codes)
            return fwd(torch.cat([self.prefix_emb(ctx_codes.shape[0]), z_in], 1)).float()

    @torch.no_grad()
    def proposals(self, canon, plan: MaskPlan):
        """(B, S) replacement codes for every slot (the masked ones are used). Wrong-by-construction kinds never
        return the canonical code."""
        B, g, dev, l0 = canon.shape[0], self.repl_gen, self.device, self.l0
        if self.prop in ("uniform", "uniform_wrong"):
            parts = []
            for k, lk in enumerate(self.levels):
                c = canon[:, sum(self.levels[:k]):sum(self.levels[:k + 1])]
                if k == self.K - 1:
                    parts.append(c)                              # never masked
                elif self.prop == "uniform":
                    parts.append(torch.randint(0, self.vocab[k], (B, lk), device=dev, generator=g))
                else:
                    parts.append((c + torch.randint(1, self.vocab[k], (B, lk), device=dev, generator=g)) % self.vocab[k])
            return torch.cat(parts, 1)
        if self.prop == "frozen_wrong":
            r0 = self.draw_root(canon)
            logits = self._logits(self.frozen, self.frozen, canon)
            logits.scatter_(-1, canon[:, l0:, None], float("-inf"))
            samp = self.draw_fine(torch.softmax(logits / self.policy_temp, -1))
            return torch.cat([r0, samp], 1)
        # policy / policy_used_parent: A3's policy (level 0 from the unigram, T = 1, one step given clean coarser levels)
        lv0 = torch.multinomial(self.L0, B * l0, replacement=True, generator=g).view(B, l0)
        ctx = canon
        if self.prop == "policy_used_parent":
            used0 = torch.where(plan.masks[0], lv0, canon[:, :l0])          # the level-0 codes actually used
            ctx = torch.cat([used0, canon[:, l0:]], 1)
        logits = self._logits(self.net, self.fwd, ctx)
        probs = torch.softmax(logits / self.policy_temp, -1)
        samp = torch.multinomial(probs.flatten(0, 1), 1, generator=g).view(logits.shape[:-1])
        return torch.cat([lv0, samp], 1)

    # ------------------------------------------------------------------ flags
    @torch.no_grad()
    def flags(self, plan: MaskPlan):
        if self.flag_mode is None:
            return None
        f = mask_flags(self.tok.vq, plan.masks)
        if self.flag_mode == "shuffled":
            f = shuffle_within_level(f, self.levels[1:], self.flag_gen)
        return f

    # ------------------------------------------------------------------ bookkeeping
    def record(self, used, targets, plan: MaskPlan):
        """Realized per-level disagreement (used != label), over all rows and over the rows not left clean."""
        dis = [d.float() for d in torch.split((used != targets), self.levels, dim=1)]
        keep = ~plan.clean
        st = self.stats
        st["n"] += int(used.shape[0])
        st["n_clean"] += int(plan.clean.sum())
        for k, d in enumerate(dis):
            st["sum_all"][k] += float(d.sum())
            st["cnt_all"][k] += float(d.numel())
            st["sum_corr"][k] += float(d[keep].sum())
            st["cnt_corr"][k] += float(d[keep].numel())

    def realized(self) -> dict:
        st = self.stats
        r = lambda s, c: [round(a / b, 4) if b else 0.0 for a, b in zip(s, c)]
        return {"disagreement_all_rows_per_level": r(st["sum_all"], st["cnt_all"]),
                "disagreement_corrupted_rows_per_level": r(st["sum_corr"], st["cnt_corr"]),
                "rows": st["n"], "clean_rows": st["n_clean"]}

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        return {"mask_rng": self.mask_gen.get_state(), "repl_rng": self.repl_gen.get_state(),
                "flag_rng": self.flag_gen.get_state(), "stats": copy.deepcopy(self.stats)}

    def load_state_dict(self, st: dict):
        self.mask_gen.set_state(st["mask_rng"].cpu())
        self.repl_gen.set_state(st["repl_rng"].cpu())
        self.flag_gen.set_state(st["flag_rng"].cpu())
        self.stats = copy.deepcopy(st["stats"])


# ====================================================================================================== eval-only
def input_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-window clean-label NLL (nats): logits (B, S - l0, V), targets (B, S - l0)."""
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    return ce.view(targets.shape).sum(1)


@torch.no_grad()
def forward_logits(net, prefix_emb, inputs, device: str, flag: FlagVector | None = None, flags=None, bs: int = 64,
                   hidden: bool = False, keep_logits: bool = True):
    """fp32, TF32 off, eval mode. Returns logits (and the last hidden states after ln_f at the target positions if
    `hidden`), in batches of `bs`. `flag`/`flags` add f * w to the inputs; omitted = all-zero flags. keep_logits=False
    (hidden only) returns None for the logits, so a hidden-state pass over many windows does not hold them."""
    outs, hids = [], []
    cap = {}
    h = net.transformer.ln_f.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o)) if hidden else None
    try:
        with fp32_eval(net, device):
            for s in range(0, inputs.shape[0], bs):
                inp = inputs[s:s + bs]
                if flag is not None and flags is not None:
                    inp = flag(inp, flags[s:s + bs])
                lg = net(torch.cat([prefix_emb(inp.shape[0]), inp], 1))
                if keep_logits:
                    outs.append(lg.float())
                if hidden:
                    hids.append(cap["h"][:, net.prefix_len:].float())
    finally:
        if h is not None:
            h.remove()
    lg = torch.cat(outs) if keep_logits else None
    return (lg, torch.cat(hids)) if hidden else lg


def make_perturbations(canon: torch.Tensor, levels: list, vocab: list, js, seed: int) -> dict:
    """(a) The perturbation of each window, shared by every model: for level j one fixed random position and a
    uniformly random *different* code, from a numpy stream that depends only on `seed`."""
    rng = np.random.default_rng(seed)
    B = canon.shape[0]
    out = {}
    for j in js:
        pos = rng.integers(0, levels[j], B)
        shift = rng.integers(1, vocab[j], B)
        out[j] = (torch.as_tensor(pos), torch.as_tensor(shift))
    return out


@torch.no_grad()
def context_dependence_kl(net, tok, prefix_emb, canon, perturb: dict, device: str, bs: int = 64) -> dict:
    """(a) mean KL(clean || perturbed) of the predictive distributions of the levels finer than j, when one code at
    level j is changed (the finer inputs are rebuilt from the changed codes; finer codes stay canonical). Computed and
    summed over window batches of `bs`, so no full-vocabulary tensor spans more than `bs` windows (Codex PR 12)."""
    levels = [int(l) for l in tok.vq.levels]
    vocab = [int(v) for v in tok.vq.vqvae_vocab_sizes]
    tot = {int(j): 0.0 for j in perturb}
    cnt = {int(j): 0 for j in perturb}
    for s in range(0, canon.shape[0], bs):
        cb = canon[s:s + bs]
        clean = forward_logits(net, prefix_emb, tok.vq.idx_to_ncm_input(cb), device, bs=bs)
        rows = torch.arange(cb.shape[0], device=canon.device)
        for j, (pos, shift) in perturb.items():
            c = cb.clone()
            idx = sum(levels[:j]) + pos[s:s + bs].to(canon.device)
            c[rows, idx] = (c[rows, idx] + shift[s:s + bs].to(canon.device)) % vocab[j]
            assert bool((c != cb).sum(1).eq(1).all())
            pert = forward_logits(net, prefix_emb, tok.vq.idx_to_ncm_input(c), device, bs=bs)
            lo = sum(levels[1:j + 1])                                    # first logit position of a level > j
            lp_c, lp_p = clean[:, lo:].log_softmax(-1), pert[:, lo:].log_softmax(-1)
            p_c = lp_c.exp()
            kl = torch.where(p_c > 0, p_c * (lp_c - lp_p), torch.zeros_like(p_c)).sum(-1)
            tot[int(j)] += float(kl.double().sum())
            cnt[int(j)] += kl.numel()
            del pert, lp_c, lp_p, p_c, kl
        del clean
    return {j: tot[j] / cnt[j] for j in tot}


def auc_score(scores: torch.Tensor, y: torch.Tensor):
    """Area under the ROC curve (Mann-Whitney, ties get average ranks); None if one class is missing."""
    y = y.bool()
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    u, inv, cnt = torch.unique(scores.double(), return_inverse=True, return_counts=True)
    rank = (cnt.cumsum(0) - (cnt - 1) / 2.0)[inv]
    return float((rank[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_probe_auc(h_tr, y_tr, h_te, y_te, l2: float = 1e-2, iters: int = 100):
    """(b) Logistic probe (L2 on standardised features, full-batch L-BFGS): AUC on the held-out windows. Shapes
    (windows, positions, features) / (windows, positions)."""
    d = h_tr.shape[-1]
    Xtr, Xte = h_tr.reshape(-1, d).float(), h_te.reshape(-1, d).float()
    ytr, yte = y_tr.reshape(-1), y_te.reshape(-1)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(d, device=Xtr.device, requires_grad=True)
    b = torch.zeros((), device=Xtr.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    yf = ytr.float()

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xtr @ w + b, yf) + l2 * (w * w).sum() / 2
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return auc_score(Xte @ w + b, yte)


@torch.no_grad()
def corrupted_contexts(kind: str, tok, net, L0, prefix_emb, x: torch.Tensor, device: str, seed: int, rates) -> dict:
    """(b) / (c) The corrupted windows shared by every model: W1's rates on every window (no clean rows), replacements as
    M2 (`kind='plausible_wrong'`: one-step samples of the frozen released generator, canonical excluded) or M3
    (`'uniform_wrong'`). `net` must be the released generator. Returns inputs, truthful flags, canonical labels
    (B, S - l_0) and the per-position contamination label."""
    from .corruption import CorruptionCfg, build_hierarchy
    arm = {"plausible_wrong": "m2", "uniform_wrong": "m3"}[kind]
    m = MechArm(arm, tok, net, L0, prefix_emb, device, seed, rates, clean_frac=0.0, policy_temp=1.0,
                frozen=MechArm.frozen_copy(net) if arm == "m2" else None)
    plan = m.draw_plan(x.shape[0])
    canon = m.canonical(x)
    repl = m.proposals(canon, plan)
    h = build_hierarchy(tok, x, CorruptionCfg(mode="mech"), policy_codes=repl, masks=plan.masks, canonical=canon)
    flags = mask_flags(tok.vq, plan.masks)
    return {"inputs": h["inputs"], "flags": flags, "targets": canon[:, m.l0:], "contaminated": flags > 0,
            "masks": plan.masks, "used": h["used"], "canonical": canon}


@torch.no_grad()
def flag_use(net, flag: FlagVector, prefix_emb, ctx: dict, device: str, flags=None, bs: int = 64) -> dict:
    """(c) Clean-label NLL per window on the corrupted windows with the true flags minus with zero flags (negative =
    the flags help), mean over windows. `flags` overrides the truthful flags (e.g. shuffled ones for M5)."""
    f = ctx["flags"] if flags is None else flags
    nll_true = input_nll(forward_logits(net, prefix_emb, ctx["inputs"], device, flag, f, bs), ctx["targets"])
    nll_zero = input_nll(forward_logits(net, prefix_emb, ctx["inputs"], device, bs=bs), ctx["targets"])
    return {"nll_true_flags": float(nll_true.mean()), "nll_zero_flags": float(nll_zero.mean()),
            "delta_per_window": float((nll_true - nll_zero).mean())}
