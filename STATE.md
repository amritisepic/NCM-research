# STATE — read this first

_Last updated: 2026-10-28, W2M-GR built and merged (R064, PR 18; deep audit 13); `w2mgr_eval` cleared. Before that, deep audit 12 (R063: W2C claim, no P0/P1; W2M-GR registered). Before that, W2C stage 1 read (R062; PR 17 merged). Before that, deep audit 11 (R061: E1 claim audit; claim drafted for the user). Before that, session 9 continued (R058-R060: W2M-G, M3-0, E1b read). Before that, session 10 (R057: R056 package registered, built, merged in PR 16 after Codex and deep audit 10; user bought 100 CU). Before that, session 9 (R052-R055; plan `docs/plan_2026-10-27.md`; handoff `docs/prompts/handoff_2026-10-27.md`). Branches: `claude/ncm-research` (notebooks clone it) and
`claude/ncm-phase2-analysis-finetune-pkh64d` (older mirror). Repo is PUBLIC: commit nothing identifying (no
personal names/emails, no absolute local paths, no session links, no credentials). Goals, success bar and rules:
`docs/brief.md`._

## Current phase
**Latest (R063, 2026-10-28):** **deep audit 12 (W2C claim): no P0 / P1; every R062 number reproduces; executed traces
confirm the matched comparison.** Claim wording narrowed (comparative to M3; "plausible-wrong" defined as frozen
released-model draws given canonical coarser levels; each seed at least +18.36). Not a finding about the levels until
the localization replicates: **W2M-GR registered** (`733ea77`, user-approved; W2M-G (i)-(iii) on the seed 1-3
checkpoints, thresholds fixed in advance, ~2-2.5 CU on L4). Its code is being built on `claude/wip-w2mgr`; Codex PR
and a delta deep audit before launch.
**R062 (2026-10-28):** **W2C stage 1: damage replicates and the dissociation holds (registered readings).**
M2 - M3 Gen-PPL (GPT-2 large) = +18.36 / +19.51 / +18.76 at seeds 1-3 (mean 18.88, s_p 0.56), TinyStories-33M +29.8,
duplicates +12.3 per 1k, MAUVE -0.41, all 16 sample groups M2 > M3 in every seed; clean NLL M2 - M3 = +4.03
nats/window (0.5%, same sign every seed). **Believed to meet criterion (c) as registered; not claimed**: the user is
told first, and a claim audit (Tier 2) comes before any claim. Scope: one protocol (continuation of the released
TinyStories model, 25% clean rows, W1's rates, 600 steps), one domain. The level localization (R058) is one seed.
PR 17 (A11-1) merged at `01ed57a`; `results/e1b_analysis.json` reproduces byte-identical. No Colab run in flight.
**R061 (2026-10-28):** **deep audit 11 (E1 claim): no P0; every R054 / R060 number recomputes.** One P1
(A11-1: `scripts/analyze_e1b.py` did not check the e1b records against the pinned plan; R060's reading is unchanged) is
fixed on PR 17 (Codex review clean, gate passes), **not merged** until the user reports W2C finished. **The E1 claim is
drafted for the user** (`docs/claim_e1_2026-10-28.md`): levels 8 and finer given canonical levels 0-7, the registered
items, modelwise 95%, Monte Carlo error only. Resampling items leaves thin text margins (5% quantiles 0.82 TinyStories,
0.80 WMT14).
**R058-R060 (2026-10-28):** **E1b: E1's registered (c) replication bar is met**: at cut 8 the lower bound of
R_8 / N_8 is 0.894 (TinyStories), 0.859 (WMT14, was 0.763) and 0.993 (Sudoku); E1a's 0.90 bar only in Sudoku. Not yet
claimed: audited in R061, the claim waits for the user. **W2M-G:** M2 and M3 diverge 5.5-6.9x more on generated
contexts than on real data; the damage forms at levels 2-5 and is amplified by M2's own later levels (interaction
0.43-0.47), not at L1 (pi_2 = 0.009), so **the L1R gate fails and L1R is not run**; M2's L1 leans 41% less on the root
as the channel model predicted, at no cost; M4's flag is a regime switch (f = 1 reproduces M2). **M3-0:** without clean
rows the control returns to 16.50 Gen-PPL (I = +5.19 ± 0.09) while canonical NLL worsens by 106 nats (78% at L8-L11):
the clean rows caused W2M's control drift. One seed for W2M-G and M3-0.

Phase 2. Wave 1 (W1/W1a/W1b, seed 0, 6 arms) is analysed: registered verdicts in R030, exploratory analysis in R031.
No arm beats the control by the registered bar (A2 re-quantized labels is closest: -0.31 Gen-PPL, bar 0.35). Every
on-policy arm collapsed (Gen-PPL 32-41 vs 16.6); two independent defect hunts (Codex audit 3 and mine) found no defect
that explains it. Leading explanation: corruption by the model's own plausible codes is undetectable, so the model
learns to discount its own coarse codes (Huszár's scheduled-sampling improperness, made visible by residual
quantization); a CPU test on Sudoku supports the hedging step (R031), not yet the link to the text collapse. As registered, A4 and A5 nominally advance against the
collapsed A3; the user dropped their seeds and holds A1's (R033). Since then (R034): deep audits 3 (completed), 4 (WMT14
trainer: three P1s, fixed in PR 8, amendment J1c) and 5 (the W1d diagnostic: no P0/P1) and Codex's adversarial review of
the mechanism ((b') survives, not established; a used-parent mismatch at L0/L1 is a live competitor). **WMT14 pair (R036, seed 0):
J2 misses its screen** (val 9.36 passes, BLEU 18.53 vs floor 22.55; test 8.89 passes at -6.6 BLEU): dead for criterion (d)
as registered. Exploratory: 16% of test sentences collapse onto two input-independent strings reached in ~3 passes; on
the rest J2 saves 9%. The user had J2-F (frozen released-model teacher) registered as a contingent arm; J2's miss
triggers it (registered after this result, stated in the text); not built. Nothing is cleared to launch: `w1d_requant_diag`
waits for PR 7 and J2-F for its code, review and deep audit; Codex is at its usage limit (since 2026-10-24 16:36 UTC). Advisory, unregistered literature report:
`docs/literature_advice_2026-10-24.md`. **W1d (R039):** reproduction gate exact; A2's Gen-PPL gain exists only at T = 1
(+0.22 at T = 0.8, +0.05 at 0.9, -0.33 at 1.0): registered reading "trade-off", A2 not pursued, no W1c.
**J2-F (R042):** the frozen teacher removes J2's collapse (shared outputs 0.3% vs 16.2%; prediction held) but saves
only 5.6% of test passes: misses the screen, kill met, no seeds; the consistency line stops. Lead (val only): an
output-stability stop at 7.80 passes (29% below 11) and BLEU -0.54 vs J0, inside (d)'s text thresholds in logical
passes; not a claim (one seed, post-hoc rule, no timing; ~2.3x the positions). Flagged to the user; D1 decides. **W2M pre-reading (R043):**
flags saturate at W1's rates (M5 is a near-replicate of M4) and probe (b) is confounded by level identity (AUC 0.95
from the level alone); amendment **W2Ma** (readings only). Astra review triaged; re-plan proposed in
`docs/plan_2026-10-27.md` (primary target criterion (c); (d) gets one wall-clock gate): **approved** (R043-R044; (d) keeps
the wall-clock rule). **S0 (R045):** the left-neighbour code adds 6.2 nats/window over levels 1-5 (1.7%; gate needs >= 10
and >= 5%): Astra's coupled sampler is dropped, E2 not run. **Registered before any run (R046):** D1, J1d (contingent),
E1. Their code (+ the W2Ma probe fix, flag-power script) is **merged** (PR 14, 2026-10-27): Codex clean after three
P1s fixed; deep audit 9 found no P0 / P1 (R048). Notebooks `d1_batch1_wallclock`, `e1_equivalence`, `w2m_mech_eval` ready.
**W2M (R049, seed 0, M5 skipped):** the collapse reproduces (M1 +20.8 Gen-PPL vs M0); frozen plausible-wrong M2 +20.9,
uniform-wrong M3 0.0, used-parent M6 +20.5, uniform at A3's rates M7 +0.4, truthful row flags M4 +9.9 (R4 = 0.53).
Registered reading **mixed** (M4 short of 0.8; nothing overturned); **sanity fails** (M0 21.7 vs A0 16.2: the control
drifted, prime suspect the 25% clean rows). Exploratory: the collapsed arms keep the control's clean NLL (within 4
nats), so the damage lives in generation-time contexts. J1a bootstrap and the WMT BLEU split: R050.
**D1 (R052):** the text gate fails (J2-F + stable x1 takes 1.41x J0 sequential's batch-1 latency, fp32, A100; Sudoku's
verifier stop 0.44x): criterion (d) is closed on text, J1d is not run. **Mechanism eval (R053):** M4 uses its flags
(-35 nats/window), so W2M's registered reading is final: mixed; plausible-wrong windows are detectable at the row level
by every model (AUC >= 0.92 from level 2); the collapsed arms are not less coarse-context dependent on clean contexts.
**E1 (R054; analysis merged, PR 15, Codex clean):** at cut 8 most code NLL is representation choice in released TinyStories
(lower bound 0.875) and Sudoku (0.993) but not established in WMT14 (0.763: 18 censored pairs hold 21% of N_8):
prediction 1 holds in 2 of 3 domains, E1 misses its (c) bar; prediction 2 unreadable (censoring). **Astra round 3**
(R051) triaged in R055 / `docs/plan_2026-10-27.md` section 10: W2C stage 1 (M2 / M3, seeds 1-3, 21.9 CU), the M3
no-clean pilot, an expanded W2M-G (shared banks, crossed replay, root reliance, M4 flag sweep), the L1 target
reassignment kill gated on W2M-G, E1b (continue sampling on censored items, ~1.5 CU); flag stage 2 cut. Awaiting the user.

## Environment
- **Codex:** reviews PRs (bot `chatgpt-codex-connector`; 2-15 min per review); a review triggered only by opening a PR may
  leave no receipt the gate counts, so request one explicitly. Deep audits requested as PR comments need a Codex cloud
  environment for the repo, which does not exist: run audits locally (`scripts/review/codex_local.sh` with the Codex
  desktop CLI, or Codex desktop). Usage limits: hit 2026-10-24 ~02:35 UTC, reset ~06:52 UTC; **hit again at 16:36 UTC**
  (PR 7's re-request) and not reset at the time of writing: no gated merge until it resets. PR 15 (E1 analysis) merged 2026-10-27 (R054). PR 14 (D1 / E1 / W2Ma eval code) merged 2026-10-27 (R048). PR 12 (W2M) merged 2026-10-27 (R040). Merged: PRs 2-11 (5, 6, 7, 9, 10, 11 in session 5, R037-R038). Limit reset by 22:57 UTC 2026-10-24.
- Local Codex CLI (audit 11): the Codex desktop app bundles `codex.exe` in its `resources` folder next to a Linux
  `codex` binary, which Git Bash picks first ("Exec format error"). Put a `codex` shim that execs `codex.exe` first on
  PATH (in MSYS form, `/c/...`) before `scripts/review/codex_local.sh`.
- `scripts/review/codex_pr.py wait` on this Windows laptop: crashes decoding the bot's emoji (cp1252) unless run with
  `PYTHONUTF8=1`, and returns early (gate FAIL) when Codex posts its "review running" status comment. Workaround used:
  poll the `codex-review-state` comment, then `codex_pr.py status`. `scripts/review/` is edit-protected: the user fixes.
- Laptop: an RTX 4050 (6 GB) is reachable from the shell now (S0 ran on it, R045), but it dropped off the bus during
  the E1 prototype ("GPU is lost"): reboot before using it again; keep sampling batches small (it spills to host RAM).
  Earlier note: no GPU access from the shell (blocked); `.venv` has torch 2.6 CPU, which has the no-grad/autocast
  weight-cache bug (R031): never train on-policy under autocast with it. Codex's audit installed torch 2.11.0+cpu in a
  temp folder (usable via PYTHONPATH). Tests: `PYTHONPATH=tests/_deps;tests;src python -m pytest tests --ignore=tests/_deps`
  (97 pass with PR 5, on torch 2.11 and 2.6).
- Cloud agent: no GPU (4-core Xeon). Fresh containers need `pip install torch omegaconf hydra-core regex
  transformers sacrebleu pyarrow` and `bash scripts/fetch_assets.sh checkpoints` (+ `sudoku`, `tinystories`, `wmt14`).
- Colab: the user launches notebooks from `notebooks/` (generated by `notebooks/make_notebooks.py`). Since PR 3 every
  notebook records the git SHA, UTC times, GPU and resume point of each session (`<run>__provenance.jsonl`,
  `_meta.json`) and refuses to resume a run directory under changed code or from an unprovenanced checkpoint.
  A100-40GB ~5.4 CU/h (w0).

## Established (measured unless marked)
- Sudoku reproduced (99.4 / 96.4 / 86.3); success falls with propagation depth (R012).
- Released TinyStories NCM: Gen-PPL 16.16 ± 0.11 (W1b protocol: 16 x 256, fp32), entropy 3.97, MAUVE 0.827 at
  4096 vs 4096 (0.888 at 1024 vs 1024 in w0: different MAUVE scales, R031).
- Coarse codes are window bags of words (R011). Straddling levels L8-L10 carry near-chance linearly readable content
  (R016) and 43% of the NLL (R008).
- **Oracle-intervention curve (R020, corrected R023):** supplying real levels 0..j and sampling the rest moves Gen-PPL
  16.0 -> 8.6; levels 1-3 hold 48% of that gap, 4-5 39%, 6 5%, 7-11 7% (L8-L10: 4%). Not an attribution.
- **Wave 1 (R030):** A1 control 16.60, A2 re-quantized 16.29, A6 soft labels 17.46, A3 / A4 / A5 on-policy 41.4 /
  36.8 / 32.2 (seed 0). Clean NLL and sample quality split again: A2 / A4 gain 113 / 154 nats of clean NLL, 88-98% of it
  in L8-L11 (~4-7% of quality); A4 has the best NLL and the second-worst samples (R031).
- **On-policy collapse character (R031):** pooled vocabulary as human-like as the control's, local order broken
  (bigram gain 2.13 -> 1.71 nats/token, adjacent duplicates 6.9 -> 17.7 per 1k), story plans intact; clean-NLL damage
  at L3-L6; appears by step 300 and does not grow. Not a defect found by two hunts.
- R014 (corrected R031): generator one-step samples keep 98% of decoded tokens, uniform codes 43% (decoder ceilings).
- Exact Jacobi across levels (training-free, R020): Sudoku-hard 6.41 passes (-46.6% logical NFE, identical output,
  1.87x by position count); verifier stop 2.37 passes, 90.3% solved (vs 86.15%); WMT14 11.07 vs 11 (no saving), BLEU
  22.63; TinyStories no saving. Any criterion-(d) claim needs batch-1 wall-clock and an attention-inclusive cost.
- R021: repair levels built from the finest Sudoku codebook fix <= 63% of failures even with an oracle.
- Throughput (A100): tier L 39-41 TFLOPS (2.1 s/step uniform, 3.0 on-policy, 3.7 two-pass), tier M NCM 24.7, AR 44.2.

## Live hypotheses / bets (ranked: significance x chance of a clean result / CU; plan `docs/plan_2026-10-27.md`)
0. **E1 (R054, R060, R061):** after E1b, prediction 1 holds in all three released models (lower bounds 0.894 / 0.859 /
   0.993), so E1's (c) replication bar is met as registered. Deep audit 11 found no P0 (A11-1 fixed on PR 17). The
   claim is drafted (`docs/claim_e1_2026-10-28.md`) and waits for the user. R054 (before E1b): WMT14 not established
   (censoring). Original entry: E1, how much of the code NLL is representation choice (codes that decode to
   the same text): `R_k = N_k - T_k` at cuts >= 8 (cuts 5-7 rarely hit: prototype), released TinyStories / WMT14 /
   Sudoku + A1 / A2 / A4 / J1 / J2-F, ~2-3 CU, primary criterion-(c) candidate. D1, batch-1 wall-clock gate for (d)
   (~1.5 CU; text prior ~20%). S0 failed (R045): coupled sampler dropped.
1. ~~Criterion (d)~~ — **closed on text (R052)**: D1's gate fails (J2-F + stable x1 1.41x J0 sequential, fp32 A100);
   Sudoku's verifier stop is 0.44x sequential. J1d not run; no more Jacobi training.
2. **Mechanism of the on-policy collapse (candidate for criterion c).** W2M (R049, one seed): plausible-wrong
   replacement is sufficient and the tested competitors are not needed; truthful row flags halve the damage; clean NLL
   is unaffected. Mech eval (R053): flag use -35 nats/window, reading final (mixed). Next (proposed, plan section 10, R055):
   W2C stage 1 (M2 / M3 seeds 1-3), M3 without clean rows, expanded W2M-G; random-row flag stage cut. Earlier notes: undetectable corruption -> the model discounts
   its own coarse codes. Sudoku CPU (R031, matched slots and rates): plausible wrong codes raise predictive entropy on
   clean contexts by 74% (uniform: 5%) and clean NLL by 32 nats (2.7), and cut reliance on the true L6 code by 76% (R034,
   Codex's adversarial test). R036's paired rerun (ADV-1 fixed) reproduces the hedging (clean NLL +32 vs +3) but not
   R031's "greedy hides it": solve rates fall ~5 points in both arms. No Sudoku collapse. Suggestive; competitors open
   (used-parent mismatch at L0/L1, lost denoising coverage, feedback, and the bf16 policy forward that only on-policy
   arms had: R036 corrects "every arm alike"). J2's collapse (R036) is a second moving-target feedback failure. Next: short tier-L arms (~15 CU) designed from the adversarial review and the literature
   report's 2x2 (plausible-wrong vs uniform-wrong replacements x truthful vs shuffled flags, common masks), plus a
   used-parent L1 proposal ablation and an A3 replicate with an fp32 policy; measure dependence on coarse context, not only entropy. Also an
   eval-only temperature check of A3 (W1d covers A0-A2 only; hedging predicts A3's gap to A1 shrinks at lower T).
3. **Clean code NLL is not a quality proxy (candidate for criterion c):** four observations; make it rigorous with an
   equivalence-class / decoded-token NLL on the 7 W1 checkpoints (~1 CU, checkpoints on Drive) vs Gen-PPL.
4. ~~A2 (Infinity-style re-quantized hierarchy)~~: dropped (R039, W1d "trade-off": the gain reverses at T = 0.8). Side
   observation: A1's 0.69 shortfall to A0 is entirely in sampling levels 0-3 (oracle j >= 3 closes it).
5. Coarse-plan interventions (levels 1-5 hold 86% of the oracle gap): eval-only, entropy-matched (~1 CU); speculative.
- On hold: B6/C5 Sudoku repair (S1, ~7.5 CU with its controls); M1 tier M (criterion b has no candidate);
  C2 red-black steps (P1c step 2 not built). **Plan proposes cutting** S1, M1 and A1 seeds 1-2 (awaiting the user).

## Killed / demoted
- **B1a on-policy corruption with canonical labels** (W1 A3/A4/A5): not supported at tier L, and destructive
  (R030/R031); as an improvement method, dropped unless corruption is made detectable. Sudoku's greedy on-policy arm
  (S1 A2) recommended for dropping: the greedy policy changes only a few % of codes.
- B7 soft labels (A6): not supported (+0.85 Gen-PPL).
- B2-trained consistency for text Jacobi (J2 moving teacher, R036; J2-F frozen teacher, R042): neither saves >= 25%
  of exact passes at < 1 BLEU; no seeds.
- C4 carry for Sudoku (R019; scoped R023). Fixing tokenizer loss on TinyStories. Level-skipping early exit on Sudoku.
- C7, C8, C11 demoted. Training-free Jacobi on text (no saving). Token-aligned schedule demoted (R020).
- P1b withdrawn (R023).

## Compute
- **Paper-scale costs (2026-10-28):** measured training throughput of the 88.6M generator on an A100-40GB is 37.9 TFLOPS effective (~26 PFLOP per CU), about half what `docs/compute_model.md` / `docs/phase1b_scaleup.md` assumed. The paper's full TinyStories training (~34 EFLOP) is ~650-1,300 CU per run, not 400+; its ablation protocol (10k x 1024, ~3.5 EFLOP) ~70-135 CU per run. The user offered +500-1000 CU for a warranted paper-scale experiment; the Astra round-4 prompt asks for a verdict per tier.
- **2026-10-28 (user-reported, at `w2mgr_eval`'s launch): 98.94 CU left.** W2C stage 1 and the idle time since used 19.7 CU (118.61 -> 98.94), against ~21 estimated at run-time rates: the run-time rate holds. `w2mgr_eval` ~2-2.5 CU.
- **R062 (2026-10-28): W2C stage 1 done, 239 A100-min, ~21 CU at run-time rates (~26 at the ~1.25x burn). Expected
  balance ~95 CU (118.61 at W2C's start); ask the user for the panel number before planning spend.** Nothing queued.
- **2026-10-28 (user-reported): balance 118.61 CU; both W2C stage 1 notebooks running at 10.6 CU/h combined (5.3 CU/h
  per A100-40GB, as the repo's assumed 5.4).** W2M-G, M3-0, E1b and W2C's first minutes used 9.3 CU against 7.4 of
  logged run time: the per-hour rates did not change, so R056's ~2x gap came from runtimes left connected or other
  use, and the runtime-release cell plus the user's checks removed it. Plan CU figures return to run-time rates plus
  ~25%: W2C stage 1 is ~22 CU, not 44.
- **R057: the user bought 100 CU (2026-10-28): ~128 CU available (27.93 + 100); the brief's budget rises accordingly.** Ready to launch (PR 16): W2M-G ~5, M3-0 ~5.4, E1b ~6, then W2C stage 1 ~44, L1R ~10 if gated in (CU at R056's doubled burn; caps in GPU-minutes in `docs/colab_runs.md`).
- Colab: ~116 CU spent (p1, c4, w0 ~12.5; wave 1 ~46 incl. ~6.5 lost to restarts, R030; WMT14 pair ~11.9, R036;
  W1d ~3, R039; J2-F ~6.5, R042; W2M ~26, R049; D1 ~5.6, mech eval ~0.5, E1 ~4.3, R052-R054), **estimated from run
  time; actual burn is ~2x (R056)**: the panel shows **27.93 CU** left this month (75.6 used since 103.56, vs 36.9
  from run time). Until explained, double every CU estimate; state kill caps in A100-minutes. The brief's ~200-CU
  budget may already be reached: buying compute is the user's decision (recommended: 100 CU when W2C needs it).
- Plan (`docs/plan_2026-10-27.md`, proposed): W2M ~27 (registered); E1/E2 ~2; D1 ~1.5; W2C 14-28 if W2M's gate
  passes; J1d ~23 only if D1's text gate passes; coupled-sampler pilot <= 5 only if S0 passes; reserve 15. Likely
  path ~71, ~50 unallocated.
- CPU: ~13.5 CPU-h before session 5; session 5 ~1.2 h (Sudoku paired rerun, test suites).

## Next 3 actions
1. **User (Colab): `w2mgr_eval` is running** (launched 2026-10-28, ~23:00 UTC) (R064: PR 18 merged at `17a4bf4` after Codex x2 and deep audit 13, A13-1
   fixed). L4 only, seeds 1-3, ~2-2.5 CU; same Drive as the `w2c1_*` notebooks. Then copy `results/colab/w2mgr_eval/`
   and run `python scripts/analyze_w2mgr.py`. The W2C claim (wording in R063) waits for its reading. Merge no code
   while it runs.
2. **User: the E1 criterion-(c) claim** (`docs/claim_e1_2026-10-28.md`, R061): claim it as scoped, as supporting
   evidence only, or hold.
3. **User: next compute (~95 CU expected).** Options: W2M-G localization on the seed 1-3 checkpoints (eval-only, ~2 CU
   per seed on L4; `scripts/w2mg_eval.py` pins seed 0, small code change + registration); the 0%-clean factorial; Phase
   3 synthesis and write-up. **Do not run `l1r_L_ts_m2r`** (R058: its gate failed).
