# Draft (not a registration): tier-L mechanism arms for the on-policy collapse

Status: draft for the user's decision (scope, compute, one architecture change). Nothing here is registered. It
reconciles the adversarial review's planned arms (`docs/reviews/codex_adversarial_onpolicy.md`, "What the planned
tier-L arms must decide") with the literature report's 2x2 (`docs/literature_advice_2026-10-24.md`, section 2), in
light of R036 (precision of the wave-1 policy forward is an untested competitor; R031's Sudoku hedging replicates, its
"greedy hides it" does not).

## Question
Wave 1's on-policy arms (A3 / A4 / A5) collapsed (Gen-PPL 32-41 vs 16.6). Leading account (b'): replacements drawn
from the model's own distribution are plausible, hence undetectable, so the model learns to discount its own coarse
codes. Live competitors: (i) loss of denoising coverage at A3's lower realized corruption rate, (ii) used-parent
mismatch (L1 proposals sampled under the canonical L0 while 75% of windows carry a unigram L0), (iii) moving-policy
feedback, (iv) the bf16 policy forward that only on-policy arms had (R036).

## Common protocol (all arms)
Released TinyStories NCM, PR 5's text trainer (fp32 policy forward, separate policy RNG), seed 0, the same data order,
600 steps (the wave-1 collapse appeared by step 300 and did not grow; R031), lr / schedule as W1 scaled to 600 steps,
canonical targets, one common mask stream (identical slots in every arm that corrupts), 25% of examples left clean in
every arm. Evaluated at steps 100 / 300 / 600 (curve) and 600 (final, W1b protocol: 16 x 256 fp32 samples, Gen-PPL
under both scorers, entropy, MAUVE, bigram gain and adjacent duplicates, clean NLL per level in fp32).
Mechanism measurements on the final checkpoints (eval-only, same windows for every arm):
- coarse-context dependence: change one code at level j (j in 1-6) on clean val windows, all else fixed; KL of the
  finer levels' predictive distributions and change in decoded local order;
- replacement detectability: a linear probe on the generator's hidden states separating clean from replaced positions
  (plausible-wrong and uniform-wrong), AUC on held-out windows;
- for flag arms: flag use, i.e. the change in predictions when only the flags are toggled on identical inputs.

## Arms
| id | arm | tests | prediction of (b') | overturned / weakened if |
|---|---|---|---|---|
| M0 | uniform NCM corruption (control, 600 steps) | baseline at this length | healthy (about A1) | - |
| M1 | A3 replicate: current-policy samples, T = 1, fp32 policy | validity: does the collapse reproduce at 600 steps with fp32 policy? | collapses as A3 | M1 healthy: the bf16 policy (iv) or the old draw order mattered, and the rest of the design must be re-thought before reading it |
| M2 | frozen plausible-wrong, no flags | (b') without feedback | damages (Gen-PPL rise >= half of A3's excess) | benign at equal slots while M1 collapses: favours feedback (iii) or used-parent mismatch (ii) |
| M3 | uniform-wrong, no flags (same slots as M2) | plausibility is the cause | healthy | as damaging as M2: plausibility is not needed |
| M4 | frozen plausible-wrong + truthful flags | detectability is the remedy | M2's damage removed by >= 80%, flags demonstrably used | removes < 20% despite demonstrated flag use |
| M5 | frozen plausible-wrong + shuffled flags | flag capacity / regime separation control | damages as M2 | rescues like M4: the rescue is regime separation, not detection |
| M6 | current-policy, L1 proposals sampled under the L0 actually used | used-parent mismatch (ii) | collapses as M1 | healthy while M1 collapses: (ii) is sufficient |
| M7 | uniform corruption at A3's per-level realized disagreement (estimated from A3's saved trajectory) | lost denoising coverage (i) | healthy | reproduces most of M1's damage: (i) is sufficient |

M2-M5 are the literature 2x2 (plausible vs uniform wrong x truthful vs shuffled flags; M3 is the uniform cell without
flags, M3f/M3s would complete the factorial). Flags signal replacement attempts from the sampled mask, not
`used != canonical` (which leaks correctness), and are propagated to every finer level whose cumulative residual input
contains a replaced ancestor. Inference uses all-zero flags. "Frozen" = proposals from a frozen copy of the released
model, one-step, T = 1, canonical code excluded (as R031's Sudoku arms).

## What needs your decision
1. **Architecture:** the flag arms add a learned per-position flag embedding to the generator's code inputs
   (zero-initialised, so step 0 equals the released model). This is the only model change.
2. **Scope and compute** (A100, measured rates 2.1 s/step uniform, 3.0 s/step with a policy forward, plus ~0.7 CU final
   eval and ~0.3 CU mechanism eval per arm):
   - (A) all 8 arms: ~27 CU;
   - (B) M0, M1, M2, M3, M4, M5 (validity + the 2x2 core): ~21 CU;
   - (C) M0, M1, M2, M4, M6 (validity, the flag rescue, used-parent): ~18 CU.
   Single seed in every option: each arm can challenge a large effect but cannot establish a mechanism (criterion c
   needs replication). Each run is < 15 CU; the total is above the ~15 CU planned in STATE.
3. Build order after approval: register (each arm's prediction and overturning result as above), implement on a
   `claude/wip-` branch, Codex review, deep audit, then the notebooks.
