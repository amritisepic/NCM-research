"""Training hierarchies for the NCM generator: clean (upstream), NCM's conditioning corruption, and our
self-correcting variants (B1) and soft targets (B7).

All variants run one pass over the levels, mirroring upstream `z_to_idx` + `idx_to_ncm_input`:
  inputs[k]  = area-downsample (to l_k) of the cumulative reconstruction built from the codes *used* at
               levels < k (what the generator sees as conditioning for level k);
  targets[k] = the label for level k:
      clean labels (NCM):  the canonical code of the clean encoding;
      re-quantized (B1):   the nearest code to the *true residual left after the used (possibly corrupted)
                           coarser codes*, i.e. the code that best repairs the earlier errors.
Codes used at corrupted positions: uniform random codes (NCM) or the generator's own samples (on-policy).
Soft targets (B7): softmax over the top-M nearest codes of that same residual, temperature relative to the
typical nearest/second-nearest distance gap of the level.
With rates = 0 and clean labels this reproduces upstream inputs/targets exactly (unit-tested).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CorruptionCfg:
    mode: str = "none"          # none | uniform | onpolicy
    mean_rate: float = 0.5      # mean per-level corruption probability over levels 0..K-2
    spread: float = 0.5         # rates go linearly from mean*(1+spread) at level 0 to mean*(1-spread) at level K-2
    retarget: bool = False      # B1: re-quantize labels against the true residual
    soft_tau: float = 0.0       # B7: >0 enables soft labels (relative temperature)
    soft_topm: int = 16
    soft_gap: tuple | None = None   # per-level distance gap that sets the soft-label temperature; None = batch mean
                                    # (depends on the micro-batch; train_gen.py freezes a calibrated value, W1b)
    level_rates: tuple | None = None   # explicit per-level rates (overrides mean_rate/spread), for diagnostics

    def rates(self, n_levels: int) -> list[float]:
        if self.level_rates is not None:
            return [float(r) for r in self.level_rates]
        if self.mode == "none" or self.mean_rate <= 0:
            return [0.0] * n_levels
        m = n_levels - 1
        hi, lo = self.mean_rate * (1 + self.spread), self.mean_rate * (1 - self.spread)
        r = [hi + (lo - hi) * k / max(m - 1, 1) for k in range(m)]
        return [min(max(x, 0.0), 1.0) for x in r] + [0.0]   # the finest level never conditions anything


@torch.no_grad()
def build_hierarchy(vqvae, x: torch.Tensor, cfg: CorruptionCfg, policy_codes: torch.Tensor | None = None,
                    generator: torch.Generator | None = None):
    """x: (B, L) token ids. policy_codes: (B, S) codes sampled from the generator given clean inputs (on-policy;
    level 0 = what inference uses there, e.g. unigram samples).
    Returns dict(inputs=(B, S - l_0, D) NCM inputs for levels 1.., targets=(B, S) labels for every level,
    canonical=(B, S) clean codes, used=(B, S) codes fed forward, z_used=(B, L, D) their full reconstruction,
    soft_idx/soft_w=(B, S, M) or None, corrupted_frac=float, corrupted_frac_per_level=list of K floats)."""
    vq = vqvae.vq
    with torch.autocast(device_type=x.device.type, enabled=False):
        z = vqvae._encode_pre_quant(x).float()
        B, L, C = z.shape
        levels = [int(l) for l in vq.levels]
        K = len(levels)
        rates = cfg.rates(K)
        z_used = torch.zeros_like(z)     # reconstruction from the codes actually fed forward
        z_clean = torch.zeros_like(z)    # reconstruction from canonical codes
        inputs, targets, canon, s_idx, s_w, used_all = [], [], [], [], [], []
        n_corr, n_tot, acc = 0, 0, 0
        frac_lvl = []                    # per level: share of fed-forward codes that differ from the labels
        gap_lvl = []                     # per level: soft-label gap used
        for k, lk in enumerate(levels):
            if k >= 1:
                inputs.append(F.interpolate(z_used.transpose(1, 2), size=(lk,), mode="area").transpose(1, 2))
            E = vq.embeddings[k].weight

            def nearest(resid):
                u = vq._down_for_quant(resid, k, lk) if k != K - 1 else resid
                d = vq._distances(u.reshape(-1, C), E).view(B, lk, -1)
                return d

            d_clean = nearest(z - z_clean)
            c_canon = d_clean.argmin(-1)
            d_lab = nearest(z - z_used) if cfg.retarget else d_clean
            c_lab = d_lab.argmin(-1)
            canon.append(c_canon)
            targets.append(c_lab)
            if cfg.soft_tau > 0:
                top_d, top_i = d_lab.topk(cfg.soft_topm, dim=-1, largest=False)
                if cfg.soft_gap is not None:
                    gap = torch.tensor(float(cfg.soft_gap[k]), device=x.device)
                else:
                    gap = (top_d[..., 1] - top_d[..., 0]).mean().clamp_min(1e-6)
                gap_lvl.append(float(gap))
                s_w.append(torch.softmax(-(top_d - top_d[..., :1]) / (cfg.soft_tau * gap), -1))
                s_idx.append(top_i)
            used = c_lab.clone()
            if rates[k] > 0:
                m = torch.rand(B, lk, device=x.device, generator=generator) < rates[k]
                if cfg.mode == "onpolicy" and policy_codes is not None:
                    repl = policy_codes[:, acc:acc + lk]
                else:
                    repl = torch.randint(0, E.shape[0], (B, lk), device=x.device, generator=generator)
                used = torch.where(m, repl, used)
                n_corr += int((used != c_lab).sum())
            frac_lvl.append(float((used != c_lab).float().mean()))
            n_tot += B * lk
            used_all.append(used)
            for zz, cc in ((z_used, used), (z_clean, c_canon)):
                q = vq.embeddings[k](cc)
                q = vq._up_to_full(q, k, L) if k != K - 1 else q
                zz += vq._apply_quant_resi(q, k)
            acc += lk
    return {"inputs": torch.cat(inputs, 1), "targets": torch.cat(targets, 1), "canonical": torch.cat(canon, 1),
            "used": torch.cat(used_all, 1), "z_used": z_used,
            "soft_idx": torch.cat(s_idx, 1) if s_idx else None, "soft_w": torch.cat(s_w, 1) if s_w else None,
            "corrupted_frac": n_corr / max(n_tot, 1), "corrupted_frac_per_level": frac_lvl,
            "soft_gap_per_level": gap_lvl}


def soft_ce(logits: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against sparse soft labels: -sum_j w_j log p(idx_j). logits (N, V), idx/w (N, M)."""
    lp = F.log_softmax(logits.float(), -1)
    return -(lp.gather(-1, idx) * w).sum(-1).mean()
