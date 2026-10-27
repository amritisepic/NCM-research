"""R043's pre-reading check of W2M (docs/preregistration.md, "W2Ma"), by simulation: how much information do the truthful
flags carry at W1's rates, and how much of probe (b)'s AUC does the level identity alone give?

  python scripts/w2m_flag_power.py            (writes results/w2m_flag_power.json and prints it; CPU, seconds)

Runs the repository's `mask_flags` / `shuffle_within_level` on masks drawn exactly like `MechArm.draw_plan` (clean-row key,
then one uniform per slot of levels 0..K-2), with a stand-in VQ that follows the footprint code path (linear up-sampling,
level 0 covering the whole window, kernel-3 residual-conv halo). It simulates the mask geometry, not the tokenizer's weights.
  (a) saturation setting: 4096 rows, 25% clean rows, seed 0.   (b) probe setting: no clean rows, seed 5 (probe (b)'s contexts).
Shuffling uses a fresh generator (seed SHUFFLE_SEED), as MechArm's own flag generator is separate from the mask stream.
What reproduces R043 (variants tried: log of the build session): the |truthful - shuffled| mean over CORRUPTED rows only
(0.00144; over all rows 0.00108); the probe setting draws NO clean-row key (with the key drawn, seed 5 gives 99.65% /
2.25 negatives / AUC 0.947 -- the same distribution, another sample); the level-only score is the indicator "not level 1"
(the per-level contamination rate gives AUC 0.993). L0 replaced in 75.65% of corrupted rows rounds to 75.7, R043 says
75.6 (one row of 3072); reported, not forced.
"""
from __future__ import annotations

import json
import os
import sys
import types

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ncmr import w2m as W  # noqa: E402
from ncmr.corruption import CorruptionCfg  # noqa: E402

LEVELS = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
RATES = [0.75 - 0.05 * k for k in range(11)] + [0.0]       # W1's rates: linear 0.75 -> 0.25 over levels 0..K-2, finest never
SHUFFLE_SEED = 0


class _Conv:
    kernel_size = (3,)


class _ResI:
    def __getitem__(self, key):
        return _Conv()


def stand_in_vq():
    return types.SimpleNamespace(
        levels=list(LEVELS), learned_l1_sampling=True, learned_all_sampling=False, quant_resi_enabled=True,
        quant_resi=_ResI(), _qr_num_levels=4,
        _up_to_full=lambda q, k, L: F.interpolate(q.transpose(-1, -2), size=(L,), mode="linear").transpose(-1, -2))


def draw_masks(B, clean_frac, seed, rates=RATES, levels=LEVELS, draw_key=True):
    """The masks of `MechArm.draw_plan` on a CPU generator seeded `seed`: (clean (B,), [mask (B, l_k)]). `draw_key=False`
    skips the clean-row key draw (only with clean_frac = 0), which is how R043's probe-setting sample was drawn."""
    g = torch.Generator().manual_seed(seed)
    assert draw_key or not clean_frac
    key = torch.rand(B, generator=g) if draw_key else None
    n_clean = int(round(clean_frac * B))
    clean = torch.zeros(B, dtype=torch.bool)
    if n_clean:
        clean[key.argsort()[:n_clean]] = True
    masks = []
    for k, lk in enumerate(levels):
        if k == len(levels) - 1:
            masks.append(torch.zeros(B, lk, dtype=torch.bool))
            continue
        u = torch.rand(B, lk, generator=g)
        masks.append((u < rates[k]) & ~clean[:, None])
    return clean, masks


def saturation(B=4096, clean_frac=0.25, seed=0):
    vq = stand_in_vq()
    clean, masks = draw_masks(B, clean_frac, seed)
    flags = W.mask_flags(vq, masks)
    sizes = LEVELS[1:]
    sh = W.shuffle_within_level(flags, sizes, torch.Generator().manual_seed(SHUFFLE_SEED))
    cor = ~clean
    fc = flags[cor]
    per_level = [float((b == 1).float().mean()) for b in torch.split(fc, sizes, dim=1)]
    diff = (flags - sh).abs()
    return {
        "rows": B, "clean_rows": int(clean.sum()), "seed": seed,
        "l0_replaced_corrupted_rows": float(masks[0][cor].any(1).float().mean()),
        "flag_eq1_per_level_corrupted_rows": per_level,           # level 1 first (index 0 = level 1)
        "flag_gt0_per_level_corrupted_rows": [float((b > 0).float().mean()) for b in torch.split(fc, sizes, dim=1)],
        "flag_eq1_all_slots_corrupted_rows": float((fc == 1).float().mean()),
        "flag_gt0_all_slots_corrupted_rows": float((fc > 0).float().mean()),
        "level1_flag_constant_within_row": bool((fc[:, :sizes[0]] == fc[:, :1]).all()),
        "mean_abs_truthful_minus_shuffled_corrupted_rows": float(diff[cor].mean()),
        "mean_abs_truthful_minus_shuffled_all_rows": float(diff.mean()),
    }


def level_only_aucs(contaminated, sizes, w_clean_mask=None):
    """AUC of scores that know only the level of a position, label = contaminated (flag > 0), pooled over positions."""
    lev = torch.cat([torch.full((s,), k) for k, s in enumerate(sizes)]).expand_as(contaminated)
    y = contaminated
    rate = torch.tensor([float(b.float().mean()) for b in torch.split(y, sizes, dim=1)])      # per-level contamination rate
    is1 = (lev != 0).double()                                                                # indicator "not level 1"
    return {"indicator_not_level1": W.auc_score(is1, y), "per_level_rate": W.auc_score(rate[lev], y)}


def probe_setting(B=4096, seed=5):
    vq = stand_in_vq()
    sizes = LEVELS[1:]
    clean, masks = draw_masks(B, 0.0, seed, draw_key=False)
    flags = W.mask_flags(vq, masks)
    y = flags > 0
    neg = ~y
    l0_clean = ~masks[0].any(1)
    out = {"rows": B, "seed": seed, "window_positions": int(sum(sizes)),
           "contaminated_fraction": float(y.float().mean()),
           "negatives_per_window": float(neg.float().sum(1).mean()),
           "negatives_at_level1_fraction": float(neg[:, :sizes[0]].float().sum() / neg.float().sum())}
    out["auc_level_only_all_windows"] = level_only_aucs(y, sizes)
    # exploratory variant: windows whose L0 is not replaced
    out["auc_level_only_l0_clean_windows"] = level_only_aucs(y[l0_clean], sizes)
    out["l0_clean_windows"] = int(l0_clean.sum())
    return out


def main():
    cfg_rates = CorruptionCfg(mode="mech", mean_rate=0.5, spread=0.5).rates(len(LEVELS))
    assert all(abs(a - b) < 1e-9 for a, b in zip(cfg_rates, RATES)), (cfg_rates, RATES)
    res = {"procedure": __doc__.strip().splitlines()[0] + " | " + "masks as MechArm.draw_plan on CPU generator(seed); "
           "stand-in VQ levels %s, learned L0, kernel-3 halo; shuffle generator fresh, seed %d; rates %s"
           % (LEVELS, SHUFFLE_SEED, [round(r, 2) for r in RATES]),
           "saturation_25pct_clean_seed0": saturation(), "probe_setting_no_clean_seed5": probe_setting()}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "w2m_flag_power.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
