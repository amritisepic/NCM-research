# Run log (append-only)

One entry per run: hypothesis, config hash, estimated vs actual compute, results, decision.
Compute units: CPU-h = wall-clock hours on the 4-core cloud container (Xeon w/ AMX; ~0.5 TFLOPS fp32,
~2-3 TFLOPS bf16 matmul peak); CU = Colab compute units. Results files are in `results/`.

---

## R000 — environment and data checks (2026-10-23)
- **What:** fetched upstream code (HazyResearch/ncm @ c702742), released checkpoints, Sudoku test sets,
  TinyStories BPE data; verified pickles contain no code (restricted unpickler).
- **Sudoku test-set audit** (`src/ncmr/sudoku.py`, exact solution counter): all 6,000 test puzzles have a
  unique solution; easy/medium/hard reuse the *same* 2,000 solution grids with 41/46/51 blanks.
  Fraction solvable by naked+hidden singles alone (no search): easy 99.9%, medium 98.8%, hard 85.5%;
  mean parallel propagation rounds 3.7 / 5.1 / 7.5 (max 9 / 12 / 18).
- **Compute:** <0.1 CPU-h. **Decision:** difficulty = blank count; use propagation rounds as a per-puzzle
  difficulty covariate in all Sudoku analyses.

## R001 — Sudoku reproduction, first attempts (aborted)
- Two launches killed: (1) per-level decode bug in my script (decoded native-length latents, which cannot
  form a board); fixed by decoding full-resolution truncated latents (`ncmr.ncm_tools.zhat_snapshots`);
  (2) 10x slowdown from running two 4-thread jobs on 4 cores. **Rule adopted: one heavy CPU job at a time.**
- 128-board smoke test (hard): 85.2% solved, NFE 12, 40 s on CPU fp32 eager.

## R002 — released TinyStories tokenizer: per-level reconstruction by token class
- **Hypothesis:** the tokenizer is lossy on names/numerals/rare tokens (prompt's premise).
- **Setup:** `scripts/analyze_tokenizer.py`, 2000 val openings (story + PAD form), spaCy POS classes,
  rare = bottom 2% of word mass (200k train stories). Config: released `tinystories/config.yaml`.
- **Results** (`results/phase0_tokenizer_analysis.json`): full-depth token accuracy 99.98%
  (error 184 ppm overall; names 918 ppm, rare words 4588 ppm, numerals 0, function words 51 ppm).
  Truncated (full-resolution) decodes: L2 22.2%, L4 55.6%, L6 82.4%, L7 94.1%, L9 98.2%, L10 99.0%.
  Rare words lag most (L7 76.0%). EOS reconstructed 100%, **PAD 0%** (decoder emits "Once upon" after
  EOS) => NCM trained on packed story-start windows, not padded stories. Re-encoding the reconstruction
  reproduces 98.3-99.6% of codes per level. Encoder is a lookup table: 17.56M trainable params
  + 18.87M EMA codebook entries = 36.43M total (vs our 10M cap).
- **Decision:** the "lossy tokenizer" premise is mostly false on TinyStories at full depth (errors are
  10-25x elevated for names/rare words but absolute rates are <0.5%). The NCM–AR gap must come from
  generating codes. Switched our data pipeline to packed story-start windows.
- **Compute:** ~0.3 CPU-h (incl. a contended run).

## R003 — H2: are coarse codes bags of words? (released TinyStories tokenizer)
- **Pre-registered prediction:** if coarse codes are window bags-of-words, shuffling tokens inside the
  windows of level k leaves level-k codes mostly unchanged, while a 1-token meaning-preserving edit
  changes the affected window's code more often.
- **Setup:** `scripts/probe_invariance.py`, 1000 val windows, seed 0.
- **Results** (`results/phase0_invariance_probe.json`, % codes unchanged): shuffle within 16-token
  windows: L1 73.3% (L0 0.0%: level 0 uses a learned, position-dependent downsampler); shuffle within
  8-token windows: L1 73.5%, L2 89.3%; one synonym swap: L1 96.7%, L2 97.8% (one window of 8/16
  affected => ~27% / ~35% of affected windows change); one random-token swap: L1 93.5%, L2 95.9%.
- **Reading:** scrambling every window's word order is *less* visible at level 2 than one synonym swap
  in one window. Coarse "concepts" behave like bags of words. Caveat: level-0 changes confound level 1;
  a per-window, level-0-teacher-forced version is the clean follow-up.
- **Compute:** ~0.1 CPU-h.

## R004 — Phase-0 reproduction: released Sudoku-hard checkpoint (+ H1 serial-depth analysis)
- **Setup:** `scripts/repro_sudoku.py --level hard` (upstream generate, eager, fp32, greedy, batch 128),
  2000 released test boards; config = released `sudoku/hard/config.yaml`.
- **Est. compute:** ~141 TFLOP, ~10 CPU-min. **Actual:** 716 s generation wall-clock (12 min), 0.2 CPU-h.
- **Result** (`results/phase0_sudoku_repro_hard.json`): **86.30% solved** (paper 86.1; independent repro
  86.3) — reproduced within noise. Cell acc 98.35%, NFE 12.
- **Per-level (full-resolution truncated decodes):** cell accuracy 14.6% (L0) … 76.0% (L9), 81.9% (L10),
  98.4% (L11); 0% of boards solved or verifiable before the last level => level-skipping early exit
  (e.g. upstream PR #12) can never fire on this checkpoint.
- **H1 (pre-registered: success falls with propagation depth) — supported**
  (`results/phase0_sudoku_depth_analysis.json`): solved 97.8% (0-4 rounds, n=46), 99.0% (5-6, n=517),
  95.6% (7-8, n=705), 84.8% (9-10, n=322), 63.9% (>=11, n=119), 50.2% (needs search, n=291).
  Spearman(rounds, failure | singles-solvable) = 0.30, p = 6e-36. Failures are predictable: AUC 0.87
  from the decoder's minimum cell confidence, 0.72 from max code entropy at the last level.
- **Decision:** adopt "serial depth" as the organizing hypothesis for the Sudoku track (C10, C4). Any
  adaptive-compute rule for Sudoku must change the tokenizer/level schedule (C5) or *add* compute where
  confidence is low; the released hierarchy cannot exit early.

## R005 — Sudoku training data (ours)
- `scripts/gen_sudoku.py --level hard --n 100000 --seed 0`: random complete grid (randomized backtracking)
  + uniqueness-preserving hole digging to 51 blanks; 0 collisions with test grids; 362 s on 2 workers.
  Stored at `assets/sudoku_train/hard.npz` (last 2000 = our val split).
- Difficulty matches the released hard test set: 86.5% singles-solvable (test 85.5%), mean propagation
  rounds 7.47 (test 7.49) on a 2000-puzzle sample.

## R006 — C4 implementation check (Recurrent NCM)
- `RecurrentNCM(carry=False)` on the released Sudoku-hard checkpoint reproduces upstream `generate`
  exactly (identical greedy codes on 8 puzzles) and the parallel teacher-forced logits to 9e-5.
- Smoke tests of tokenizer / ncm / rncm training and --final evaluation pass (tiny configs).

## R007 — Sudoku easy/medium reproduction (released checkpoints)
- easy **99.4%** (paper 99.5), medium **96.4%** (paper 96.5), hard 86.3% (paper 86.1, R004). All within
  noise of the paper and the independent reproduction. NFE 12. Files: `results/phase0_sudoku_repro_*.json`.
- One earlier launch hit the 30-min background limit (easy at 1664/2000); relaunched with a 2-h limit.

## R008 — released TinyStories generator: teacher-forced per-level code NLL
- `scripts/released_generator_eval.py nll --n 1000` (val windows, packed). `results/phase0_released_generator_nll.json`.
- NLL nats per window, levels 1..11 (lengths 8..128): 55.8, 80.8, 84.9, 78.0, 66.2, 64.1, 29.7, 84.4,
  131.8, 161.4, 33.5; **total 870.5 nats/window = 6.8 nats/token** (+ level 0 from a unigram).
  Per code: 6.98, 5.05, 3.54, 2.44, 1.65, 1.23, **0.46 (64 codes)**, 1.11, 1.43, 1.49, **0.26 (128)**.
- **Reading:** codes are a deterministic function of the text, so H(codes) = H(text) (~1-1.5 nats/token
  for TinyStories AR models, to be measured by our AR baseline). The generator is therefore far from the
  code distribution, and the bound log p(x) >= log p(x|r*) + log p(r*) is loose (many code hierarchies
  decode to the same text). The token-straddling levels (76/92/108) carry 43% of the NLL while adding
  ~5 points of truncated-decode accuracy. Motivates B7 (equivalence-class targets) and a token-aligned
  schedule test; P1 will split the rest into within-level TC vs representation cost.
- **Paper read (docs/2610.14162v1.pdf):** AR baseline on TinyStories (same architecture and updates)
  Gen-PPL 8.89 vs NCM 15.41; conditioning corruption is NCM's largest ablation effect (~48 -> ~19);
  more levels keep helping. Plan revised in `docs/phase1b_reevaluation.md`.

## R009 — repository hygiene (public repo)
- Rewrote branch history (force-push with lease) so no commit carries a personal name/email or a
  session link; trees unchanged. Later commits omit session links.
- Colab time estimates revised upward: p1_tc_pilot ~1.5-3.5 h on a T4, c4_sudoku_pilot ~2-4 h
  (sequential level passes are kernel-launch bound; quantization runs in fp32). Actuals to follow.

## R010 — released TinyStories NCM: full evaluation suite (CPU)
- 5 seed groups x 256 samples (1280), paper protocol (level 0 from the stored unigram, T = 1, 11 NFE).
  `results/phase0_released_ncm_tinystories_eval.json`.
- Gen-PPL GPT-2 large **16.10 ± 0.59** (paper 15.41), TinyStories-33M 11.95 ± 0.59, entropy 3.971 (paper
  3.98), MAUVE 0.878 ± 0.007 (vs 1280 human val windows), rep-2/3/4 0.101/0.022/0.006, distinct-2/3
  0.194/0.459. Generation 623 s on 4 CPU cores (0.49 s/sample).

## R011 — invariance probe, full run (n = 1000; released tokenizer)
- Unconditional (all levels re-encoded): within-16-token shuffles leave 73% of L1 codes unchanged, within-8
  shuffles leave 74% of L1 and 89% of L2 unchanged; a one-token synonym swap leaves 94-98% of codes at
  L0-L7 and 89-92% at L8-L10 unchanged. With coarser levels forced (conditional analysis, R003 method):
  L1 99.94% unchanged under w16 shuffles; L1 99.95% / L2 99.94% under w8 shuffles; synonym swap leaves the
  affected window unchanged at L1 84.5%, L2 74.8%, L3 53.4%, L4 26.2%. `results/phase0_invariance_probe.json`.
- Reading: coarse codes are window bags of words (order-free), confirmed at n = 1000.

## R012 — serial-depth law on all three Sudoku sets (released checkpoints)
- Solve rate by constraint-propagation rounds (0-4 / 5-6 / 7-8 / 9-10 / >= 11 / needs search):
  hard 97.8 / 99.0 / 95.6 / 84.8 / 63.9 / 50.2; medium 98.8 / 97.8 / 89.8 / 82.9 / 50.0 (n = 4) / 58.3;
  easy 99.9 / 98.0 / 71.4 (n = 14). Decoder min-cell confidence flags failures (AUC 0.85-0.87).
  `results/phase0_sudoku_depth_analysis.json`.

## R013 — B2: exact Jacobi (fixed-point) decoding across levels, training-free
- Sudoku-hard (500 puzzles, greedy): fixed point = sequential output in 100%; **6.39 passes vs 12
  (-46.7% NFE)** at zero quality change (85.2% solved, as sequential on this subset); 1.87x the FLOPs
  of sequential decoding; the correct board first appears at pass 2.04 on average. `results/b2_jacobi_levels_hard.json`.
- TinyStories (n = 4, sampling with fixed Gumbel noise): 12 passes vs 11, no saving. WMT14 (n = 6, greedy):
  11.33 passes; first pass with the final output at 9.8. Text gets no training-free saving.
- Accounting fix (after these runs): WMT14's sequential NFE is 11 (level 0 comes from the prefix pass), not
  10 as the smoke file says; `scripts/jacobi_levels.py` now counts it.

## R014 — B1 pre-check: what errors does the released generator make, and can re-quantized labels repair them?
- `scripts/repair_capacity.py --n 256` (CPU, 11 min), `results/b1_repair_capacity.json`. The decoded text of
  a corrupted hierarchy is the ceiling a perfect generator reaches under each label type.
- **Decoder barely needs any single coarse level:** replacing all codes of one level (0-10) by random codes,
  the rest canonical, leaves 99.2-99.97% of tokens correct. Replacing all levels but the finest by random
  codes leaves **42.6%**; by the generator's own one-step samples (given the clean coarser levels; 58% of
  codes change) leaves **98.0%**.
- **One-step disagreement with the encoder's codes** (sampled, T = 1) per level 0..11: 100 (unigram),
  98, 91, 81, 67, 55, 48, **21 (L7, 64 codes)**, **45, 53, 55 (L8-L10, token-straddling)**, 14 (L11). Greedy:
  -, 96, 82, 68, 53, 41, 35, 14, 32, 40, 41, 9.
- **Rank of the generator's sampled code** in the encoder's distance ordering: median 1055/1118/454 at
  L1-L3 (different content), 0-1 from L5 on; top-16 share 44% (L3) -> 97% (L7) -> 81-88% (L8-L10) -> 99% (L11).
- **Re-quantized labels repair only partly:** NCM's uniform recipe (mean rate 0.5; 31% of codes changed)
  -> ceiling 92.9% with clean labels vs 96.9% re-quantized; on-policy recipe (21% changed) -> 99.40% vs
  99.84%. Re-quantization changes 46-58% of labels.
- **Reading:** (1) the paper's uniform corruption is far harsher than the generator's real errors (uniform
  codes destroy 57% of tokens; the model's own one-step choices destroy 2%), so on-policy corruption
  (B1a) is a much closer match to inference; (2) at fine levels the model's "errors" are mostly
  near-equivalent codes (nuisance), at coarse levels they are content choices; (3) re-quantized labels
  (B1b) have a modest ceiling effect. Priors updated in `docs/phase1b_scaleup.md`.

## R015 — budget raised to ~200 CU (user); scale-up plan and trainer
- Plan: `docs/phase1b_scaleup.md` (tiers S/L/M, waves 0-3). Scale-limit change (released 88.6M / 36.4M
  models for tiers L/M) flagged for the user's confirmation.
- New code: `src/ncmr/corruption.py` (hierarchies: clean = upstream exactly, unit-tested; uniform /
  on-policy corruption; re-quantized and soft labels), `scripts/train_gen.py` (fine-tune or from scratch on
  the released tokenizer; multi-pass on-policy errors; sample-based eval during training; calibration and
  eval-only modes), `scripts/tf_levels_diag.py`, GPU support in `scripts/jacobi_levels.py`, `--seed` /
  `--calib_steps` / snapshots in `scripts/train.py`, duplicate-word metric in `src/ncmr/eval_text.py`.
  CPU smoke tests pass (fine-tune with on-policy + re-quantized + soft labels; scratch; calibration; AR).
- Soft-label temperature: tau = 0.25 puts 77% (72-87% by level) of the weight on the canonical code.
- Smoke-test warning: 2 Adam steps at lr 1e-4 without warmup on batch 2 raised val NLL on 4 windows from
  1075 to 1459 nats; tier-L fine-tunes therefore use lr 5e-5 with 200 warmup steps (all arms alike).

## R016 — linear probes on the released tokenizer's levels (n = 4000 val windows; complete)
- `scripts/probe_levels.py --n 4000`, run in 3 chunks after the CPU queue hit its 2-h limit (levels 0-3 from
  the interrupted run's log, 3 decimals; the probes are deterministic). `results/phase0_level_probes.json`.
  Logistic regression, 5-fold CV; "cum" = mean-pooled cumulative reconstruction up to level k, "code" = mean
  of level k's own code vectors. Chance: main character 0.36 (majority of 8 names), dialogue / content /
  order 0.50 (balanced), token@64 0.19.
- Cumulative view: dialogue 0.54 (L0) -> 0.93 (L1) -> 0.95; main character 0.32 -> 0.56 (L1) -> 0.85 (L4) ->
  0.89 (L5-L11, flat); content words 0.53 -> 0.76 (L1) -> 0.89 (L6+, flat); token@64 0.09-0.12 at every
  level (below the majority rate: pooled features carry no position; the probe overfits); order 0.66-0.74
  at every level, highest at L1-L2.
- **Own-code view collapses at the token-straddling levels:** main character / dialogue / content from
  level k's own codes: L7 (64 codes, 2 tokens each) 0.85 / 0.84 / 0.84; **L8 0.40 / 0.68 / 0.69, L9
  0.32 / 0.60 / 0.58, L10 0.34 / 0.57 / 0.56 (near chance)**; L11 (128 codes, 1 token each) 0.80 / 0.89 / 0.84.
- **Reading:** global story attributes are in place by level 1-5 and add nothing after level 6. The three
  straddling levels hold codes with almost no linearly readable content of their own, are the hardest for
  the generator to predict (R014: 45-55% one-step disagreement) and carry 43% of its NLL (R008). Being a
  residual level does not explain it (L11 is residual and content-bearing); the non-integer
  tokens-per-code ratio (1.68, 1.39, 1.19) is the distinguishing property. Raises the prior of the
  token-aligned schedule test (bet 7 in phase1b_scaleup.md) from 25% to 35%.

## R017 — decisions and arrivals (2026-10-23)
- User approved the scale-limit change (tiers L/M use the released 88.6M generator / 36.4M tokenizer) and
  launched `w0_calib_diag` on an L4. Project brief condensed into `docs/brief.md` (goals, success bar, rules).
- P1 and C4 Colab results committed by the user under `results/colab/` (T4; all runs present). Training wall
  time: P1 2.9 h (ncm_s 28 min, ts_fact_s / ts_ar_s 56-57 min each, ar_s 15 min, tok_s 14 min, m-scale
  calibration 6 min), C4 4.8 h (tokenizer 28 min, NCM 124 min, RNCM 136 min): ~10 CU incl. setup, at the top of
  the 4.5-9 CU estimate. Analysis against the pre-registrations: next session.

## R018 — P1 analysis: within-level TC share of the NCM–AR gap (tok_s, Colab T4, seed 0)
- `scripts/analyze_p1.py --runs results/colab/p1_tc_pilot` -> `results/p1_tc_decomposition.json`. **Scope: our
  8.4M tok_s tokenizer and 5.5M-body generators, not the released tokenizer/generator.** Nats per 128-token
  val window (1000 windows), final checkpoint (6000 steps).
- NLL (incl. level 0 under the unigram, 4.51): NCM 1232.7; two-stream factorized 1232.8; two-stream within-level
  AR 1290.1; AR tokens 173.6 FLOP-matched (6000 x 128 windows, 4.1 vs 4.0 PFLOP) / 212.4 data-matched (step
  1500). Gap NCM - AR = **1059.1** (FLOP-matched), 1020.3 (data-matched).
- **TC = NLL_fact - NLL_ar = -57.3 nats (-5.4% of the gap).** Per level 1..11: +1.1, +1.9, +1.0, +0.3, -2.0, -2.7,
  -2.6, **-21.0, -14.7, -17.3** (token-straddling L8-L10), -1.2. Trend over the 12 evals: -142 -> -57, rising
  monotonically, still negative at the end; all four NLL curves still falling (NCM -11 nats in the last 500 steps).
- Registered sanity checks: NLL_ts_fact within 5% of NLL_ncm (0.01%: pass); tokenizer full-depth acc 99.17%
  (pass); truncated-decode curve monotone (7.6% -> 99.2%: pass).
- **Verdict H3: inconclusive (instrument failure).** The literal rule (TC < 10% of the gap -> falsified) is met,
  but I do not apply it: true TC is >= 0 (more conditioning cannot raise entropy), so a negative value is not
  an estimate of TC but evidence that the within-level-AR probe is the weaker learner. Likely cause, found in
  the code: its previous-code embeddings use PyTorch's default N(0, 1) init (norm ~16 at d = 256) and are added
  to the code stream's latent input (norm 7-15 on the released tokenizer, measured), so the AR probe starts with
  its own-position conditioning buried in noise, while the factorized probe's code stream starts identical to
  NCM's input. The probe is therefore not a superset of the factorized one at equal optimisation.
- **Decision (phase1b_scaleup.md §7):** neither branch fires; C2/C7 are not killed by P1 and C2 stays conditional
  on the wave-0 diagnostic. Re-test registered as P1b (`preregistration.md`): the same AR probe with zero-initialized
  previous-code embeddings (at step 0 it equals the factorized probe), reusing tok_s and ts_fact_s; ~1 CU.
- **Reading (an inequality, not a test of H3):** codes are a deterministic function of the text, so
  H(codes) <= H(text) <= 173.6 nats (the AR model's val cross-entropy). Every code model here is therefore at
  least 1059 (NCM) / 1116 (within-level AR) nats above the entropy of its own targets: >= 86% of NCM's NLL is
  model excess, not information. The P1 "representation term" (1116.5) is mostly this learnability cost, not
  an information cost of the representation. Whatever TC turns out to be, it can at most be a share of a gap that
  is dominated by how hard canonical codes are to predict (consistent with R008/R014 on the released model:
  near-equivalent codes at fine levels). Small scale, one seed, curves not converged.

## R019 — C4 analysis: Recurrent NCM (carried state) on Sudoku-hard (Colab T4, seed 0)
- `results/colab/c4_sudoku_pilot/`. Tokenizer: val token acc 100%, board exact 100%. Generators 5.66M, 20k steps
  x 64, 12 NFE, greedy.
- Solve rate, val (1000 generated hard) / released hard test (2000): NCM **2.1 / 3.00%**, RNCM **0.4 / 1.05%**.
  By propagation rounds (test; <=6 n=563, 7-8 n=705, 9-10 n=322, >=11 n=119, needs search n=291):
  NCM 6.7 / 2.4 / 0.6 / 0.8 / 0.7; RNCM 3.4 / 0.3 / 0.0 / 0.0 / 0.0. Rounds >= 9 (n = 441): NCM 3/441, RNCM 0/441.
- (b) - (a): **-1.7 points val, -1.95 test, -0.68 on >= 9 rounds.** Final teacher-forced train CE per code: NCM
  0.159, RNCM 0.127 (carry fits the training data better and decodes worse).
- **Verdict H4: falsified** as registered ((b) - (a) < +1 point overall). Validity limit: both arms are at the
  floor (the released 25.5M model solves 86.3%; ours saw 1.28M puzzle exposures vs the paper's 51.2M and a 5x
  smaller body), so the test had almost no power at the depths H4 is about. The registered rule does not
  condition on baseline competence; I apply it as written.
- **Decision (§7):** carry is killed for Sudoku and does not join the tier-L Sudoku arms.
- Untested reading: lower teacher-forced loss with worse greedy decoding fits carry amplifying exposure bias
  (the carried state is built from clean levels in training, from the model's own levels at inference).

## R020 — wave 0 (`w0_calib_diag`, Colab A100-40GB; user-run)
- **Released reference with the wave-1 protocol** (`L_ts_released`, eval only): clean val NLL **861.8**
  nats/window (R008: 870.5 on CPU fp32 with a different window packing; -1.0%), Gen-PPL GPT-2 large **16.35 ± 0.51**
  over 4 x 256 (R010: 16.10 ± 0.59), TinyStories-33M 12.01, entropy 3.976, MAUVE 0.888 ± 0.021, rep-2 0.100,
  11 NFE. **W1 sanity check passed** (~870 / ~16.1).
- **Throughput calibration: not recoverable from the upload.** The three calibration runs wrote `calib.json`,
  which the notebook's collect cell did not copy (only their configs arrived). Fixed in `make_notebooks.py`
  (collects `calib.json`); the files are still on Drive under `ncm-research/runs/calib_{L,M,M_ar}/`.
- **Where quality is lost** (`diag_tf_levels_released_ts.json`, n = 512, seed 0, one group: SE ~0.4 Gen-PPL):
  real codes at levels 0..j, generator samples the rest. Gen-PPL GPT-2 large / TinyStories-33M by j:
  -1 (unconditional) 16.00 / 11.71; 0: 15.88; 1: 15.39; 2: 14.47; 3: 12.36; 4: 10.60; 5: 9.51; 6: 9.15; 7: 8.96;
  8: 8.81; 9: 8.75; 10: 8.67; 11 (tokenizer reconstruction of real text) 8.62 / 3.49. Entropy flat (3.94-3.97).
  Share of the 7.38-point gap lost when sampling each band (given the real coarser levels): level 0 2%,
  **levels 1-3 (8-24 codes) 48%, levels 4-5 (32-40 codes) 39%**, level 6 5%, levels 7-11 5% (token-straddling
  L8-L10: ~0.2 points, 3%). MAUVE: 0.82 (j = -1) -> 0.87 (real level 0) -> 0.91-0.97 from j = 1 on.
- **Reading:** quality is lost in the coarse and middle levels (the plan), not in the fine levels. The
  token-straddling levels hold 43% of the NLL (R008) and the least content (R016) but cost ~3% of the sample
  quality: NLL share and quality share dissociate. §7 rule ("most of the loss when levels 1-3 are sampled"
  -> C2 and tokenizer bets gain priority over B1): 48% < 50%, so it **does not fire as written** (borderline);
  levels 1-5 together hold 87%. Decision: **bet 7 (token-aligned schedule) demoted** — its quality upside on
  this model is bounded by ~0.2-0.3 Gen-PPL; B1 stays first (its corruption acts on exactly these levels);
  C2 gets no promotion from either trigger (P1 inconclusive, w0 below threshold).
- **Exact Jacobi across levels (training-free), full sets:** Sudoku-hard (2000): exact stop 6.41 passes
  (-46.6% NFE, 1.87x FLOPs), output = sequential in 100%, 86.15% solved; verifier stop ("check": first board
  satisfying the rules and clues) **2.37 passes, 90.3% solved** (+4.15: for ~4% of puzzles an intermediate pass
  holds the solution and the fixed point is wrong); stable-output stop 2.92 passes, 88.2%. TinyStories (512,
  sampled): 11.94 passes vs 11 (no saving). WMT14 full test (3003, greedy): **11.07 passes vs 11**, fixed point
  = sequential in 100%, BLEU (lowercased sacreBLEU) **22.63**; stable-output stop 10.59 passes, 22.64. Target for
  criterion (d) text: <= 8.25 passes at >= 21.63 BLEU. (`b2_jacobi_levels_hard.json` and
  `b2_jacobi_wmt14_n6.json` in the upload are stale copies of the repo's older files, not w0 output.)
- Criterion (d), Sudoku half: met training-free by exact Jacobi (-46.6% NFE at zero quality change), but it
  needs a text task too and the FLOPs rise 1.87x; stated, not claimed.
- Compute: A100 at ~5.4 CU/h; wall time per command not recorded in the upload. Charged as ~2.5 CU (estimate).

## R021 — Sudoku repair-level ceiling (CPU pre-check for the tier-L repair arm)
- `scripts/sudoku_repair_ceiling.py --n 250` (our generated val puzzles), `results/sudoku_repair_ceiling_val.json`,
  1.6 CPU-min. Released hard model solves **88.0%** of our generated hard puzzles (released test: 86.3%), so our
  generator's distribution is close enough for fine-tuning.
- Oracle with extra full-resolution residual levels on the finest codebook (codes = nearest to the true residual):
  solved after 0..8 levels 88.0, 88.8, 88.8, 90.4, 93.2, 95.2, 95.6, 95.6, 95.6 -> fixes 7% of failures with one
  level, 20% with three, saturates at 63% after six. A repair level built from the finest codebook is a weak
  correction channel (16 codes meant for tiny residuals). Decision: the repair arm uses a representationally
  complete pass instead (re-predict the whole hierarchy conditioned on the current board; `preregistration.md` S1).

## R022 — throughput -> step counts; Sudoku and WMT14 tier-L builds (2026-10-23)
- **Calibration (w0, A100-40GB, user-uploaded `calib_*__calib.json`):** tier L fine-tune 2.23 s/step at 256 windows
  = **39.2 useful TFLOPS** (13.7 GB peak at micro-batch 32); tier M NCM from scratch 1.11 s/step = 24.7 TFLOPS; tier M
  AR 0.133 s/step = 44.2 TFLOPS. The plan assumed ~3x the L4 rate on an A100 (75-117 TFLOPS) -> 33-52%, below the
  60% rule. **Wave-1 steps cut to 45%** (W1a: 1800 / 1350 / 1080; ~6 CU per arm on an A100). Notebooks use
  micro-batch 64 on >= 40 GB GPUs (same optimizer step).
- **Tier M is over the per-run cap at this throughput:** `M_ncm` 15k steps ~ 4.6 A100-h ~ 25 CU per seed, `M_ar`
  70k steps ~ 14 CU. Both on hold; re-plan after wave 1 (a run > 15 CU needs the user's approval).
- **Sudoku tier L built** (`scripts/train_sudoku_L.py`, `src/ncmr/jacobi_train.py`, `configs/scale/L_sdk_*.yaml`;
  pre-registered S1/S1a): arms ctrl / uniform / onpolicy / repair (board-feedback parallel passes trained on the
  model's own in-progress Jacobi decodes). Checks: the shared Jacobi decoder's fixed point equals sequential
  decoding on 100/100 val puzzles (6.32 passes); clean hierarchy = upstream inputs exactly; all four arms train,
  evaluate, resume (incl. the repair pool) and run eval-only on CPU. Cost re-estimate ~5 CU per 5000-step arm ->
  halved to 2500 steps (S1a). Carry not included (R019).
- **WMT14 target format found from the released generator's teacher-forced NLL** (48 newstest2014 pairs, nats/seq):
  English ids + EOS + PAD padding 392.5; PAD padding 459.7; packed text 1014.2; EOS padding 2706.2. The tokenizer
  reconstructs the PAD tail poorly (45% of all 64 positions) but decoding cuts at the first EOS.
- **WMT14 tier L built** (`scripts/prep_wmt14.py`, `scripts/train_wmt14_L.py`, `configs/scale/L_wmt_*.yaml`;
  pre-registered J1): control (teacher forcing on references, 3000 x 128) vs self-consistency on the model's own
  greedy Jacobi trajectories (2000 steps, equal FLOPs). CPU smoke test: both losses, trajectory sampling, fp32 eval;
  fixed point = sequential output in 100% of the smoke sentences. Data: 400k pairs from HF `wmt/wmt14`, val =
  newstest2013 (3000; 2953 within the 63-id target limit for NLL).
- Notebooks: `w2b_L_sdk_*` (4), `w2c_L_wmt_*` (2), `p1b_tc_rerun`; seed-independent commands now run once per
  notebook. CPU: ~0.7 h this session.

## R023 — external hostile review: corrections to R008, R018, R019, R020 (2026-10-23)
- Review (GPT, of commit `1c7185c`) saved verbatim in `docs/reviews/gpt_hostile_review_2026-10-23.md`; triage of every
  finding in `docs/reviews/triage_2026-10-23.md`. All its recomputed numbers were checked against `results/**.json`
  and agree. The codex audit's numeric recomputation (231 checks, R024) found the same R020 slips plus two minor ones.
- **R008 correction:** codes are a deterministic function of the text, so H(codes) <= H(text); equality would need
  invertibility (full-depth reconstruction is ~99%, so near-equality is plausible, not shown).
- **R018 corrections.** (1) The reading "≥ 86% of NCM's NLL is model excess ... mostly learnability cost, not
  information" is **withdrawn**. The excess over H(codes) (point estimate 1059.1 / 1232.7 = 85.9%, small models,
  one seed) contains the within-level factorization cost (TC) as well as approximation and optimization error; the
  bound cannot split them. Counter-example: 8 copies of one fair bit have 1 bit of joint entropy and 8 bits of optimal
  factorized cross-entropy, an 87.5% excess that is entirely TC. Consequence: the motivation of B7 / B1b ("canonical
  codes are hard to learn") loses this support; R014 (near-equivalent codes at fine levels) still stands. (2) The TC
  trend is -142.1, -148.9, then rising to -57.3, not monotone. (3) P1b's premise was false (the AR probe's attention
  mask differs from the factorized probe's, so they differ at step 0); P1b withdrawn before launch, P1c planned
  (`preregistration.md`).
- **R019 correction (scope):** "almost no power" was wrong: RNCM's 0/441 on >= 9 rounds bounds its deep solve rate at
  <= 0.68% (one-sided 95%), so a +10-point gain was detectable. The limit is the regime: both arms near floor, trained
  from scratch at 5.7M parameters and 1.28M exposures. Verdict stands as registered, scoped to "carry hurts under this
  recipe"; it says nothing about carry in a competent model. Lower training CE with worse decoding fits overfitting or
  optimization differences as well as exposure bias. Carry stays out of S1; revisit only if S1 shows that a competent
  model's remaining failures concentrate at high propagation depth.
- **R020 corrections:** of the 7.381-point gap, levels 1-5 hold **86.3%** (not 87%: 47.8 + 38.5), level 6 4.9%,
  levels 7-11 **7.3%** (not 5%), token-straddling L8-L10 **0.29 Gen-PPL = 4.0%** (not ~0.2 / 3%); levels 6-11 12%
  (STATE said 10%). MAUVE for j >= 1: 0.91-0.96. Command JSONs do hold `wall_s` (only setup / billed time is missing).
  **Interpretation narrowed:** the curve is an oracle intervention (real coarse codes supplied), not an attribution of
  the loss to levels; it says nothing about fine levels after a wrong coarse plan or about a retrained level schedule.
  Bet 7 stays demoted only for the stated reason (small fine-level cost under real coarse codes). The C2 trigger
  (48% vs 50%) is inside the noise of a one-group diagnostic and is treated as uninformative; C2 waits for P1c.
- **Criterion (d) accounting:** verifier halting is a legitimate adaptive rule (it uses only rules and clues). Logical
  NFE is not realized GPU time (batched decoding keeps stopped examples), and the w0 "FLOPs ratios" are position
  counts without attention. Any (d) claim now needs batch-1 wall-clock and an attention-inclusive cost (S1b, J1a).
- **Plan changes (all registered before any run):** W1b (16 sample groups, fp32 eval, screen 0.35 against each
  arm's registered comparators, step-matched A1 curve, conditional controls A1m / A6s), S1b (feedback zero / shuffle
  ablation, H10 credit rule), J1a (capped-Jacobi curves at 6 / 8 / 9; J2 must beat capped J0 and J1 at equal or more
  passes). Literature added to `phase1_literature_and_candidates.md` (scheduled sampling, GLAT, Infinity, CLLMs,
  Jacobi decoding for MT, RRN / HRM / TRM): B1b is close to Infinity's re-quantized self-correction; J2 is CLLM-style
  consistency across NCM levels. No prior work predicts which arm wins.

## R024 — codex code audit of the W1 trainer: fixes and tests (2026-10-23)
- Audit (`docs/reviews/codex_audit.md`, `codex_audit_numbers.json`) of `scripts/train_gen.py` and
  `src/ncmr/corruption.py` at the W1 state, numeric recomputation of R018-R021 (`tests/recompute_r018_r021.py`), and
  CPU tests on the released Sudoku-hard model. Triage in `docs/reviews/triage_2026-10-23.md`.
- **Two bugs would have invalidated W1:** (F2) `policy_codes` re-entered the caller's `torch.autocast` object; on exit
  autocast stayed on, so every later forward of the on-policy arms (val NLL, sampling, the GPT-2 large scorer) ran in
  bf16 while the uniform arms ran in fp32. Only `train_gen.py` nested the context; P1, C4, w0, Sudoku and WMT14 code
  never did, so no logged result is affected. (F3) two-pass policy codes conditioned pass 2 on the canonical level 0
  while the hierarchy carried a unigram level 0.
- **Fixed:** F2 (no nested context in `policy_codes`; `Trainer.ctx` now returns a fresh autocast context per use),
  F3 (one unigram level 0 for every pass), F1 (corruption RNG checkpointed and restored), F5 (`--eval_only` refuses a
  directory with a checkpoint; resume refuses a different init / model / corruption / train config), F6 (sample
  metrics inside `torch.random.fork_rng`), F7 (A6's soft-label gap frozen per level from 512 val windows; partition
  dependence 0.05 -> <= 3e-6), F8 (sampling and scoring in fp32 with TF32 off; A0 re-evaluated in the control
  notebook). F4 (FLOP proxy omits tokenizer / hierarchy work) accepted as a description: the omitted work is ~1% of a
  step and attention pricing is identical across arms; the proxy stays the matching unit, labelled as such.
- Also: per-level corruption rates logged (`corrupted_frac_per_level`); WMT14 eval adds capped-Jacobi BLEU at 6 / 8 / 9,
  length ratio and per-sentence passes (J1a); Sudoku eval adds feedback-zero / shuffled decoding for the repair arm
  (S1b); Windows-safe `os.devnull` in `models.py`; P1b notebook removed; W1 configs and notebooks regenerated.
- **Tests (CPU, `tests/`):** 34 passed: clean `build_hierarchy` equals upstream encoding and inputs (released Sudoku
  tokenizer + small text tokenizers), `jacobi_decode`'s fixed point equals `net.generate` greedy on the first 20
  released hard puzzles, and regression tests for F2 (also at the `Trainer` level), F3 and F7 (strict xfails in the
  audit, now passing). Jacobi cap extraction checked with a stub network; the Sudoku repair arm's eval (new feedback
  modes) ran end to end on val on CPU. Not run locally: a W1 training step and the WMT14 eval (released TinyStories /
  WMT14 checkpoints not downloaded). CPU: ~0.5 h.

## R025 — CPU smoke test of the amended S1 / J1 code; repo hygiene (2026-10-23)
- Merged the R023/R024 commit; `pytest tests` 34 passed on this container. Checked the F2 fix in `train_gen.py`
  (the policy forward no longer enters a second autocast) and that the Sudoku / WMT14 trainers never nest it.
- `train_wmt14_L.py` (J1a eval: capped Jacobi at 6 / 8 / 9, length ratio, per-sentence passes) and
  `train_sudoku_L.py` (S1b: feedback zeroed / shuffled decodes) run end to end on CPU with the smoke configs:
  training steps, final val and test eval, fixed point = sequential output in 100% of smoke sentences. Neither
  trainer has had an independent code audit (the R024 audit covered `train_gen.py` and `corruption.py`).
- Hygiene: 4 commits in the public history carry a personal author name and email; flagged to the user (rewriting
  needs their approval, since it force-pushes the shared branch).

## R026 — public-history author rewrite (2026-10-23, user-approved)
- The 4 commits authored with a personal name and email were rewritten to the GitHub noreply identity
  (`git filter-branch --env-filter`, author and committer fields only; final tree identical) and both branches
  force-pushed with lease. All commit hashes changed. Old -> new for hashes cited in this log and in `docs/reviews/`:
  `1c7185c` -> `0f37c24`, `f2222ca` -> `f9f90ff`, `5cde67a` -> `5a43383`, `d86eb78` -> `d0e26a4`, `cac8598` -> `e1384a9`, `b073e9a` -> `b397656`, `8054cb5` -> `7427b96`.
- Any local clone (e.g. the codex folder) must run `git fetch origin && git reset --hard origin/claude/ncm-research`
  after setting its identity to the noreply address.

## R027 — tier-M re-plan after an external design review (2026-10-23)
- Review saved in `docs/reviews/gpt_tierM_review_2026-10-23.md` (local file links replaced by repo paths), triage in
  `docs/reviews/triage_tierM_2026-10-23.md`. All its costs recomputed here and agree (24.93 / 13.95 CU per seed for
  the old plan, 116.6 CU for 3 baseline pairs; design C 7.65 / 4.28 CU, 27,335 FLOP-matched AR steps).
- **Adopted: design C (registered as M1)**: d512 x 6 NCM from scratch (6000 x 256, ~7.7 CU) and FLOP-matched AR
  (27,335 x 256, ~4.3 CU), 16 x 256-sample eval, level-0 prior from training windows (new `level0: train` path in
  `train_gen.py`, CPU smoke-tested). Old `w2_M_ncm` / `w2_M_ar` notebooks removed (configs kept for the record).
- **Gate added beyond the review:** the seed-0 pair (~12 CU) decides whether seeds 1-2 (~24 CU) run:
  G0 >= max(2.4, 4 x W1's measured control training SD).
- **Changed from the review:** its kill rule "AR improves >= 0.35 in its final quarter" would fire from cosine
  annealing alone; replaced by a conditional 1.5x-steps AR (seed 0) that any criterion-(b) claim must also beat.
- Budget view (A100, estimates): spent ~12.5; queued wave 1 + Sudoku + WMT14 ~54; M1 seed-0 pair ~12, seeds 1-2 ~24,
  candidate ~23, AR-c15 ~6.4; wave-1b / S1 / J1 seeds ~20-40. Total ~150-175 of ~200 CU.

## R028 — second code audit (Sudoku / WMT14 trainers) and P1c step 1; fixes before any S1 / J1 run (2026-10-23)
- Audit by codex (`docs/reviews/codex_audit2.md`, branch `codex/audit2`, files merged here), triage in
  `docs/reviews/triage_audit2_2026-10-23.md`. Its suite reproduced here before any change (12 passed, 13 strict xfails).
- **Real defects, all fixed:** repair-pool checkpoint not atomic and not restorable without drawing data (A2-1);
  Sudoku corruption RNG reseeded on resume (A2-2); no run-identity check, so `--eval_only` could have evaluated a
  fine-tuned checkpoint as the released model (A2-3); **J2's teacher ran in bf16 under the training autocast, not
  the fp32 map that exact decoding uses (A2-4)**; degenerate trajectories (A2-5); arms with the same seed did not see
  the same data (A2-6); **J2's executed compute was up to 28% below the control's when teachers stopped early
  (A2-7)**; WMT prep could return fewer pairs or reuse a stale archive (A2-8); S1b tails and c* not recorded (A2-9);
  capped hypotheses discarded, so J1a's paired bootstrap was impossible (A2-10). Three of the audit's patches were
  modified (base-checkpoint hashing, A2-5 and A2-10 fail-closed behaviour; reasons in the triage).
- No result is affected: no S1 / J1 run had started. Registered as S1c / J1b (`preregistration.md`).
- **Checks after the fixes (CPU):** test suite 59 passed (25 audit-2 tests ported to the working tree; 14 of them fail
  on the pre-fix commit); Sudoku uniform / repair train, evaluate, stop on the time budget and resume exactly (pool
  restored); the manifest refuses a changed config and `--eval_only` on a trained directory; WMT14 consist trains with
  the fp32 full-budget teacher (degenerate count 0 in the smoke run), saves capped hypotheses for caps 1-12 on val and
  test, sequential = exact on all smoke sentences; prep reproduces the previous 20k smoke selection exactly and
  refuses a mismatched archive.
- **P1c step 1 (codex, CPU, 258 s):** the two-stream TC estimator recovers known TC on synthetic codes (independent
  0, four duplicated bits 2.079, Markov q = 0.10 / 0.25: 1.104 / 0.392 nats) within 3.1e-5 nats in all 12 seed-cases
  (`docs/reviews/p1c_synthetic.json`, `scripts/p1c_synthetic.py`). This validates the estimator's arithmetic and
  basic optimisation on one tiny level with constant context, not its behaviour on learned codes. P1c step 2 (the
  zero-init side-channel probe on real codes) remains to be built and registered.
- Cost note: J2's fp32 teacher adds wall-clock; the CU estimate for `w2c_L_wmt_consist` rises from ~4 to ~5.

## R029 — automated Codex review (2026-10-23)
- Code changes to `src/ scripts/ configs/ notebooks/ assets/` now reach `claude/ncm-research` (the branch Colab
  clones) only through a PR that Codex has reviewed on its current head, with every P0/P1 triaged in
  `docs/reviews/triage_pr<N>.md` (`scripts/review/codex_pr.py`; protocol in `CLAUDE.md`; review rules for Codex in
  `AGENTS.md`, drawn from the R024 / R028 defect classes). A Claude Code hook (`.claude/hooks/review_gate.sh`) blocks
  direct pushes of gated code to that branch and merges that bypass the gate; it fails closed if it cannot compare
  with the remote.
- Deep audits (failing strict-xfail tests, number recomputation) are required before any Colab run whose trainer,
  sampler, corruption, eval or FLOP code changed since the last audit, and before any success-bar claim: via an
  `@codex` brief on the PR (cloud) or `scripts/review/codex_local.sh audit` (laptop; gpt-6.1-sol, ultra effort).
- Not covered: pushes made outside Claude Code; GitHub review needs Codex code review enabled for the repo.
- R029 addendum (laptop session, 2026-10-23): the gate was tested and four holes fixed. (1) The hook read its
  input with `jq`, which this Windows laptop lacks; it then saw an empty command and allowed every push and merge.
  It now falls back to Python and fails closed if it cannot parse input that mentions a push or merge. It also
  checks the repo the command actually runs in (hook `cwd`, a leading `cd`, `git -C`) and the source of
  `SRC:claude/ncm-research` refspecs. (2) Any Codex issue comment after the head commit counted as a review, so
  a usage-limit or setup notice would have passed the gate; now only a "didn't find any major issues" comment or a
  PR review on the head counts, and limit / setup notices make `wait` exit 4. (3) Pagination split pages on "][",
  which also occurs in comment bodies (`x[0][1]`); it now pages by hand. (4) Any 6+ digit number in the triage file
  counted as a triaged id; the id must now share a line with accept / partial / reject. `merge` pins the head SHA
  the gate passed. Tests: `tests/test_review_gate.py` (12; 10 fail against the R029 scripts on this laptop).
- Codex has not reacted to the review request on PR #1 (posted 21:17 UTC, checked 22:18 UTC): code review is
  probably not enabled for the repository in Codex settings. Until it is, no gated code can merge.

## R030 — wave 1, registered analysis (W1 + W1a + W1b, seed 0; Colab A100-40GB; user-run)
- Runs: `results/colab/w1_L_ts_{ctrl,requant,onpol,b1,onpol2,soft}/` (A0 = `L_ts_released16` in the ctrl folder).
  Every number below is recomputed by `scripts/analyze_w1.py` -> `results/w1_analysis_s0.json` (PR 2, Codex-reviewed)
  and independently by Codex audit 3 (`docs/reviews/codex_audit3_numbers.json`); all aggregates agree.
- **Code version:** every training session of every arm (12 sessions) logs `corrupted_frac_per_level`, which the R024
  fix commit introduced; from R024 to the last commit before the wave, the trainer's only change is the tier-M
  level-0 branch, which W1 configs never enter. All W1 runs therefore trained with R024-equivalent code; none is void.
  Exact SHAs were not recorded (notebooks record them from now on, PR 3). Runtime: torch 2.11.0+cu130.
- Protocol: 16 x 256 samples (seeds 42, 0-14), sampling and scoring in fp32 with TF32 off; Gen-PPL = mean of the 16
  group values (GPT-2 large), SE = group SD / 4; MAUVE = mean over 5 quantization seeds of 4096 samples vs 4096 human
  val windows; clean val NLL in nats/window (computed under the training autocast in every arm; Codex audit 3, A3-1).

| arm | steps | Gen-PPL ± SE | MAUVE | clean val NLL | entropy |
|---|---|---|---|---|---|
| A0 released | 0 | 16.16 ± 0.11 | 0.827 | 861.8 | 3.970 |
| A1 control (uniform corruption) | 1800 | 16.60 ± 0.08 | 0.881 | 844.0 | 3.962 |
| A2 uniform + re-quantized | 1800 | 16.29 ± 0.09 | 0.915 | 749.3 | 3.963 |
| A3 one-step on-policy | 1350 | 41.44 ± 0.43 | 0.496 | 885.7 | 4.026 |
| A4 on-policy + re-quantized | 1350 | 36.81 ± 0.26 | 0.673 | 707.4 | 4.037 |
| A5 two-pass on-policy | 1080 | 32.17 ± 0.18 | 0.543 | 913.9 | 3.990 |
| A6 soft labels | 1800 | 17.46 ± 0.10 | 0.832 | 888.9 | 3.958 |

- **Sanity checks:** A0 NLL 861.8 (~870) and Gen-PPL 16.16 (~16.1): pass. A1 - A0 = +0.44 Gen-PPL, below the 1.0
  recipe-damage limit: the recipe is not declared damaging.
- **Screen** (gain = comparator minus arm; pass needs gain >= 0.35, |entropy gap| <= 0.05, MAUVE gap >= -0.01):
  A2 vs A1 +0.31 (fails the 0.35 bar); A3 vs A1 -24.83; A4 vs A1 -20.20; A5 vs A1 -15.56; A6 vs A1 -0.85: all fail.
  **A4 vs A3 +4.63** (entropy gap +0.011, MAUVE gap +0.177) and **A5 vs A3 +9.27** (-0.037, +0.047) **pass** when the
  entropy and MAUVE conditions are read against the same comparator, which is how W1b's sentence reads ("below any of
  its registered comparators, entropy within 0.05 nats and MAUVE not lower by > 0.01") and how Codex read it. Read
  against A1 as in W1's original wording, both fail (A4: entropy +0.075, MAUVE -0.208; A5: MAUVE -0.338).
- **Step-count confound:** A1's curve (256 samples, seed 10000) is 16.486 at 1080, 16.705 at 1350, 16.299 at 1800;
  neither matched point is >= 0.5 better than 1800, so the rule does not fire.
- **Conditional controls:** A1m (needs A3 to advance) and A6s (needs A6 to advance) are not triggered.
- **Verdicts:** H6 (A3 < A1) not supported. H7: A2 < A1 not supported; A4 < A3 passes the screen. H8 (A5 < A3)
  passes the screen. H9 (A6 < A1) not supported. Registered wording for the arms with no passing comparison: "B1a (A3),
  B1b's uniform half (A2 vs A1) and B7 (A6) not supported at tier L (a sampling-noise-only gain of >= 0.7 Gen-PPL would
  have passed with ~95% probability)"; they are deprioritized for wave 2, not declared false. The "no arm passes" clause
  does not apply as written, because A4 and A5 pass against A3; both passes are comparisons inside a collapsed regime
  (each arm is >= 15 Gen-PPL worse than A1).
- **Registered consequences:** A1 gets seeds 1-2. A4 and A5 advance to seeds 1-2 (a Phase-3 claim for H7's second half
  or H8 would also need A3 at seeds 1-2). Whether to spend that compute is put to the user (R031).
- Training-seed variance is unmeasured (seed 0 only); per W1b the screen is a selection step, not a test.
- **Compute (estimate from the logs):** surviving training 1080-1800 steps per arm at 2.1 s (uniform) / 3.0 s
  (on-policy) / 3.7 s (two-pass) per step; restarts discarded ~0.25 h (A1), ~0.4 h (A3) and ~0.5 h (A4) of work,
  ~6.5 CU in all. Total ~8.6 A100-hours incl. ~3 min setup per session and the evals, ~46 CU at w0's 5.4 CU/h
  (6.5-9.7 per notebook; ~6.5 holds only for the notebooks without restarts). Spent to date ~58 CU of ~200.

## R031 — wave 1, exploratory: integrity checks, what the collapse is, what is significant (2026-10-24)
Everything here is exploratory and changes none of R030's verdicts.
- **Defect hunt (two independent ones): no defect explains the on-policy collapse.** Codex audit 3 (run by the user in
  Codex desktop on `d501333`; report `docs/reviews/codex_audit3.md`, triage `docs/reviews/triage_audit3_2026-10-24.md`):
  the canonical-policy identity test passes bit for bit (released Sudoku tokenizer, clean and re-quantized labels); a
  small text model with W1's level schedule matches forced sequential inference at every level; offsets, level 0,
  prefix, temperature 1, `fnet is net` and no compile all as specified. My own reading of `policy_codes` ->
  `build_hierarchy` -> `compute_loss` and of the generator's training forward vs `generate` (both block-diagonal: each
  level attends to the prefix and its own input only) agrees. Same scoring path for every arm.
- **Protocol defects (audit 3):** A3-1 the policy and val-NLL forwards run under the training autocast with TF32 on
  (every arm alike, so it cannot make one arm collapse; the result field "fp32 sampling and scoring" is true for
  samples only); A3-2 policy sampling draws from the global RNG that dropout uses, and only the root module's mode is
  restored; A3-3 the resume key ignores `grad_accum` and a missing config or corruption RNG fails open; A3-4 the
  notebooks recorded no provenance (fixed in PR 3, merged after three Codex rounds, `docs/reviews/triage_pr3.md`).
  All four reproduce (the audit's 9 strict xfails fail on `d501333` with its own torch 2.11); A3-1..A3-3 are fixed in
  PR 5 (97 tests pass on torch 2.11 and 2.6), which awaits Codex review and, before any run uses it, a deep audit.
- **R014 corrections (audit 3, checked against `results/b1_repair_capacity.json`):** the sampled median ranks
  1055/1118/454 are levels 0-2 (L1-L3: 1118/454/37); "58% of codes change" counts disagreement with the re-quantized
  labels, and the canonical disagreement is ~41.6% of all slots (52.0% of the replaced coarse slots); recipe label
  changes are 57.5% / 45.5%, and L8-L10 top-16 shares 88.3 / 80.8 / 80.5%. No conclusion of R014 changes.
- **Latent hazard, verified not to affect W1:** on torch < 2.11 a no-grad forward inside an autocast region caches bf16
  weight casts without their autograd link, and the training forward in the same region reuses them, so only weights
  autocast does not cast (norms) get gradients. Audit 3 measured 24,832 of 25.5M parameters receiving gradients in an
  on-policy Sudoku step on torch 2.6 (this laptop's environment). PyTorch fixed it in 2.11 (issue #158232; the fix is
  in v2.11.0's `aten/src/ATen/autocast_mode.cpp` and absent from v2.6-v2.10). W1 ran torch 2.11.0+cu130, and its
  on-policy steps took 1.40x the control's time (a full backward predicts 1.33x; one without weight gradients ~1.0x).
  W1's on-policy arms therefore trained all parameters. Any future on-policy run must run the policy forward with
  autocast disabled (A3-1's patch) so the result does not depend on the torch version.
- **Resume traces:** A3 resumed once from step 250 (step-300 val NLL 912.50 before, 913.67 after); A4 resumed three
  times (step 600: 731.48 vs 731.53; step 900: 716.50 vs 716.28); A1 restarted from initialization once. GPU kernels
  are not bitwise deterministic and the on-policy arms amplify weight differences through their own samples. The
  collapse is in both A3 trajectories and precedes the rollback.

**What the data say that we did not expect** (W1b protocol unless stated; scripts `scripts/w1_sample_stats.py` and
`scripts/onpolicy_repro.py`, in PR 6 awaiting Codex review; outputs in `results/`):
1. On-policy training breaks sampling fast and then stops: A3's clean val NLL 861.8 -> 912.5 by step 300; its curve
   Gen-PPL is 37.3 at step 600 and 37.0-37.2 afterwards. All three on-policy variants collapsed (32-41 vs 16.6).
2. The corruption that was mildest in token terms was the most destructive in training: replacing coarse codes by the
   generator's one-step samples keeps 98% of decoded tokens, by uniform codes 43% (R014; audit 3 notes these are decoder
   ceilings with the finest level kept).
3. The collapse is local, not global: the pooled vocabulary of A3 / A5 is as close to the human windows as A1's
   (unigram Jensen-Shannon 0.0108 / 0.0073 vs 0.0094), but local order breaks: bigram gain over unigram (training-set
   bigram model, 20M tokens) 2.13 (A1) -> 1.71 (A3) nats/token, adjacent duplicate tokens 6.9 -> 17.7 per 1k, the
   deficit spread over the whole window (-0.33 to -0.45 in every position band). Reading 30 samples per arm (numpy
   seed 2026, `results/w1_samples_read_s2026.md`): A3's stories keep their openings, characters and plot; the errors
   are doubled, dropped, fused or garbled words ("faucet faucet", "OnceOnce was", "namedddy", "Whiskersrmometer").
4. Clean-NLL gains sit where sample quality is not: A2's -113 nats vs A0 is 88% in L8-L11 and A4's -154 is 98% there;
   R020's oracle curve gives those levels ~4-7% of the quality gap. A4 has the best clean NLL of all arms and the
   second-worst samples. Fourth observation of this split (R008 / R016 vs R020).
5. A3's clean-NLL damage is mid-level (L3-L6: +15 of +24 nats), while its training loss (2.16-2.20) and gradient norm
   (0.33) are below the control's (2.45-2.53, 0.55): the arm does not take larger steps.
6. A5 (two-pass) has the worst clean NLL (+52, mostly L8-L10) but better samples than A3 (32.2 vs 41.4).
7. Paired by sample seed (the 16 groups share seeds), A2 - A1 = -0.31 ± 0.08 Gen-PPL from sampling noise alone;
   training-seed variance is unmeasured. Suggestive, below the registered bar.
8. MAUVE scale: A0 scores 0.827 at 4096 vs 4096 (W1b) and 0.888 at 1024 vs 1024 (w0). MAUVE's default quantization
   uses ~n/10 clusters, so the two protocols are not on one scale; W1b's MAUVE SDs (0.006-0.031) are over quantization
   seeds only, not sampling noise. Within W1b every arm shares the protocol, so arm differences are comparable, but
   their sampling error is unknown. Cheapest check: four disjoint 1024-sample MAUVEs of an arm, in the next notebook.

**Explanations for the collapse and the cheapest test that separates them**
- (a) A code defect. None found in two hunts; the remaining GPU-only path (autocast cache) is excluded by version and
  timing. Test if needed: 600 on-policy steps with an fp32 policy and a separate policy RNG (A3-1 / A3-2 patches).
- (b) Clean labels that no longer fit the policy's coarse codes. A4's re-quantized labels do fit the used codes and
  removed only ~19% of A3's damage (36.8 vs 41.4, control 16.6): at most a small part.
- (b') **Undetectable corruption (leading).** Uniform random codes are easy to spot, so the model learns "repair codes
  that look random, trust codes that look plausible". The policy's own samples look exactly like the codes it produces
  at inference, so with targets the input does not determine (the canonical code of the true text, or the repair
  toward it), the best it can do is discount every plausible coarse code by roughly the corruption rate. At inference
  every coarse code is its own, so fine codes are drawn partly independently of the coarse plan. Fits items 3 (plan
  kept, local detail broken), 5 (clean NLL hurt where codes depend most on their context), A4 (the repair target is
  still unpredictable from a plausible-looking context) and A5 (second-pass replacements fit a wrong plan, so among
  canonical neighbours they are partly detectable: less damage). This is Huszár's (2015) improperness of scheduled
  sampling, made visible by residual quantization. Predictions: plausibility alone, at matched slots and rates, makes
  the model hedge on clean contexts; greedy decoding hides the hedge and sampling at temperature 1 (as W1 does)
  exposes it.
- (c) A feedback loop (the policy drifts with the model): the damage appears by step 300 and stops growing.
- (d) Milder corruption (0.18 vs 0.31 of codes) loses denoising robustness: would leave clean NLL intact; it rose by
  50 nats at step 300.
- Separating tests: (i) Sudoku on CPU (below): same slots and rates, the wrong code uniform vs drawn from the model's
  own distribution with the right code excluded, decoded greedily and at temperature 1; (ii) text: on-policy plus a per-position "replaced" flag (corruption
  made detectable) should not collapse; plausible-wrong codes at NCM's uniform rates should; uniform corruption at A3's
  lower per-level rates (A1m) should not; a frozen-policy arm separates (c).

**Sudoku CPU test of the mechanism (exploratory; `scripts/onpolicy_repro.py`, `results/onpolicy_repro/`).** Released
Sudoku-hard model, fp32 on CPU, batch 16, lr 1e-4 after 20 warmup steps, 200 steps, seed 0; the same 300 held-out
generated puzzles for every arm. Masks come from one generator: identical slots, NCM's rates (0.75 -> 0.25), 25.86% of
codes replaced in both arms; labels canonical. Only the wrong code differs: uniform among the 15 other codes
(`uniform_wrong`) or drawn from the model's own one-step distribution with the canonical code excluded
(`plausible_wrong`).
- Training loss per code 0.308 (uniform) vs 0.395 (plausible), mean over the run: the model undoes plausible wrong
  codes less well, i.e. it cannot tell them from right ones as easily.
- Clean teacher-forced NLL (nats per puzzle; 43.2 at step 0): 45.9 (uniform) vs 75.2 (plausible) at step 200; the
  plausible arm's +32 sits in L8-L10 (+25.8). Same slots, rates and targets: only plausibility changed, and it made the
  model spread probability off the canonical code on clean contexts (the "hedging" that (b') predicts).
- Greedy solve rate (88.3% at step 0), paired on the same 300 puzzles, change by step 200: uniform -4.7 ± 1.5 points,
  plausible -1.0 ± 1.5 (plausible minus uniform +3.7 ± 1.6). Argmax decoding hides the hedging entirely; uniform
  corruption, which the released Sudoku model never saw in training, costs a few points.
- Temperature-1 sequential sampling (seeded, same puzzles; 87.0% at step 0): uniform -4.3 ± 1.6, plausible -5.3 ± 1.7
  (plausible minus uniform -1.0 ± 2.0). Sampling exposes part of the plausible arm's hedge (greedy -1.0 vs sampled
  -5.3; uniform: -4.7 vs -4.3), but nothing collapses: the model stays confident (<= 0.51 nats per code). Predictive
  entropy on clean contexts (nats per puzzle, 54.8 at step 0): 57.8 (uniform) vs 95.2 (plausible), the rise at L7-L11.
- Re-quantized labels with plausible wrong codes (`plausible_wrong_requant`, W1 A4's analogue): clean NLL +6.7 (not
  +32), predictive entropy +21% (not +74%), greedy -3.3 ± 1.6 and temperature-1 -4.3 ± 1.6 points (as uniform).
  Re-quantization largely removes the hedge here, yet in text A4 still collapsed with the best clean NLL of all arms:
  clean-context hedging does not account for A4. Open.
- What this does and does not show: plausibility alone, at matched slots and rates, makes the model hedge on its own
  context (predictive entropy +74% vs +5%), even though Sudoku's target is fixed by the puzzle; greedy decoding hides
  the hedge and temperature-1 sampling exposes part of it. That is the first step of (b'), replicated across domains.
  Not shown: a collapse (this confident model loses ~5 points, not half its quality), that the hedge is what broke the
  text samples, any effect at tier L, or more than one seed. J2 decodes greedily, the regime in which the hedge was
  harmless here. The released model fails on few-clue prompts (4% valid grids at 18 of 30 clues), so a Sudoku
  setting whose targets the input does not determine was not testable. Verdict on (b'): suggestive, not established.

## R032 — Codex review back, provenance in notebooks, audit-3 fixes, usage limit (2026-10-24)
- **Codex code review works** (bot `chatgpt-codex-connector`, ~2-3 min per review). PR 2 (`scripts/analyze_w1.py`,
  the R030 recomputation): no findings, merged. PR 3 (notebook provenance, Codex audit 3's A3-4): two P1s in two rounds,
  both accepted with regression tests (`tests/test_notebook_provenance.py`; triage `docs/reviews/triage_pr3.md`), a
  third round to verify the fixes found nothing; merged. Round 3 went one past CLAUDE.md's two-round cap, to review
  accepted (not disputed) fixes on the new head; flagged to the user.
- **Codex usage limit reached ~02:35 UTC.** Waiting for it: PR 4 (deep-audit-4 brief for the WMT14 trainer, required
  before the J1 launch), PR 5 (fixes for audit 3's A3-1..A3-3; then a deep audit before any run uses them), PR 6
  (exploratory scripts of R031). Nothing gated merged after that; no notebook is declared ready.
- **Review tooling on this laptop** (edit-protected, for the user): `codex_pr.py wait` crashes decoding the bot's
  emoji with Windows' cp1252 (works with `PYTHONUTF8=1`) and exits early with a gate FAIL as soon as Codex posts its
  "review running" status comment; the gate itself (`status` / `merge`) reads the final "Didn't find any major issues"
  receipt correctly.
- **Notebook provenance (PR 3):** each command appends start / end records (git SHA, UTC time, GPU, checkpoint
  present, last logged step, exit code) to `<run_dir>/provenance.jsonl`; `_meta.json` carries them. A run directory
  started at another commit refuses to resume if `src/ scripts/ configs/ assets/` differ between any earlier session's
  commit and now, and a checkpoint without provenance (every wave-1 directory) is refused; `ALLOW_CODE_CHANGE = True`
  overrides on purpose.
- **Audit-3 fixes (PR 5, not merged):** `fp32_eval` boundary for the text policy forward and `val_nll`; own
  checkpointed RNG for on-policy draws; resume key includes `grad_accum`; missing config / RNG states fail closed; text
  run manifest (bundle, BPE and data SHA-256, torch / CUDA) and hashed Sudoku base checkpoint. The audit's tests were
  committed verbatim, then ported: 97 tests pass on torch 2.11.0+cpu and on 2.6. On torch 2.6 the old on-policy step
  leaves the Linear weights without gradients (`test_all_trainable_parameters_receive_onpolicy_gradients` fails on
  `d501333`); with `fp32_eval` it passes on both versions.

## R033 — decision: wave-1 seeds (user, 2026-10-24)
- A3, A4 and A5 get no seeds 1-2. A4 and A5 nominally advance under W1b (against the collapsed A3, R030), but those
  comparisons sit inside a collapsed regime and serve no success criterion; the registered selection consequence is not
  followed, by the user's decision. Saves ~39 CU (~52 with A3 as their comparator).
- A1's seeds 1-2 (registered: "A1 always gets seeds 1-2") are held until M1's gate or a new tier-L comparison needs the
  training-seed SD (~13 CU).

## R034 — W1d diagnostic built; audits 3 (completed), 4 and 5; MAUVE precision correction (2026-10-24)
- **W1d** (registered before any run; `docs/preregistration.md`): eval-only temperature frontier (T = 0.8 / 0.9 / 1.0,
  8 x 256 samples) and oracle-intervention curve (j = -1 / 3 / 5 / 7, 1024 val windows) for A0 / A1 / A2, to decide
  whether A2 (re-quantized labels) earns a confirmation. `scripts/diag_w1_frontier.py`, notebook `w1d_requant_diag`
  (A100, ~2.5 CU; the ~1 CU first estimate was too low for this precision). Its sampler is tested equal to
  `train_gen.py`'s; at T = 1 every group must reproduce wave 1's group for the same seed within 0.01 Gen-PPL or the run
  stops before anything else. PR 7: Codex round 1 raised four P1s (reproduction only reported, MAUVE features in bf16,
  notebook provenance publishing absolute Drive paths, registration not on the PR branch), all fixed or answered with
  mutation-checked tests (`docs/reviews/triage_pr7.md`); round 2 clean. Deep audit 5 (local Codex CLI; its sandbox could
  not run Python, so its tests were run here): no P0/P1; one P2 (A5-1, the reproduction helper could drop or collapse
  groups) reproduced and fixed (`docs/reviews/triage_audit5_2026-10-24.md`).
- **Correction to R030 (precision label):** wave 1's Gen-PPL scorers ran in fp32, but its MAUVE features were computed
  under bf16 autocast (the default of `mauve_features`), identically for every arm. "fp32 sampling and
  scoring" in the result files covers sampling and Gen-PPL only. W1d featurizes MAUVE in fp32 (new opt-in flag; default
  unchanged so later W1 / M1 runs stay comparable with wave 1).
- **Audit 3 completed** (Codex desktop, `codex/audit3-onpolicy` `1697171`): its CPU diagnostic (51 small-batch steps,
  greedy on-policy vs uniform on the released Sudoku model) found no rapid collapse, an inconclusive negative consistent
  with R031; final verification 18 passed / 8 strict xfails on `1bd3816` (A3-1..A3-3 open until PR 5 merges). The
  report and numbers on the branch were updated verbatim; PR 5 was reconciled with the completed tests (97 -> 98 pass,
  torch 2.11 and 2.6) and now also writes the run's name, not its absolute path, into `samples_final.json`.
- **Audit 4** (Codex desktop on `c91126a`, `codex/audit4` `665c24b`): "hold launch" for three P1s in the WMT14 trainer:
  validation NLL under the training autocast, base checkpoint identified by size, and dropped degenerate sources still
  charged at the full FLOP proxy (576 - 12d rows executed vs 576 reported); two P2s (an initial fixed point charged two
  passes; curve evaluations discarding per-sentence records). Suite reproduced (18 passed, 7 strict xfails); all five
  fixed in PR 8 with the audit's patches, plus checkpointed run-level degenerate totals (from the audit's draft); 103
  tests pass on torch 2.11 and 2.6 (`docs/reviews/triage_audit4_2026-10-24.md`). Registered as J1c before any J1 run.
- **Codex channels:** pull-request reviews work; deep audits requested as PR comments need a Codex cloud environment for
  the repository ("create an environment"), so audits run locally (`scripts/review/codex_local.sh`, or Codex desktop).
- Launch state at the time of writing: nothing is cleared yet. W1d waits for PR 7's last review and merge; the WMT14 pair
  waits for PR 8's review and merge.
- **Adversarial analysis of the on-policy explanation** (Codex desktop, run by the user; `docs/reviews/
  codex_adversarial_onpolicy.md`, triage `docs/reviews/triage_adversarial_onpolicy_2026-10-24.md`): (b') survives but is
  not established. New direct evidence on R031's saved Sudoku checkpoints: plausible-wrong training cuts the model's
  reliance on its true L6 code by 76% (finer-target NLL advantage) and 81% (normalized logit response) relative to
  uniform-wrong training. Accepted: ADV-1 (P1, a dropout-RNG confound in `scripts/onpolicy_repro.py`, same class as A3-2;
  fix in PR 6, paired rerun needed before calling R031's Sudoku contrast controlled); a competing explanation R031 missed
  (L1 proposals are sampled under the canonical L0 while 75% of windows then carry a unigram L0, so about half the L1
  positions condition on a parent the proposal never saw; inference has no such mismatch); and a wording correction to
  R031: the corrupted-context posterior depends on the clean and replaced joint-context densities, not on the nominal
  corruption rate. The tier-L mechanism arms will add a used-parent proposal ablation and follow the review's design
  notes (flag from the mask, not from correctness; the lower-rate uniform arm registered under its own name).

## Advisory literature report published (2026-10-24; documentation only)
- Saved `docs/literature_advice_2026-10-24.md`: six-question primary-source review, paper/settings/effect-size
  table, mechanism alternatives and discriminating tests, likelihood caveats, level interventions, Jacobi
  consistency evidence, a proposed compute budget and explicit falsification thresholds.
- Recomputed released TinyStories L8-L11 clean-NLL share (46.79%) and predicted-slot share (63.13%), and
  WMT14's position-count proxy at 8.25 full Jacobi passes (2.46 times sequential). These are descriptive
  calculations, not new model evaluations or attention-inclusive FLOP measurements.
- Preserved the report's R030-R033 evidence scope and added a publication note pointing to R034's newer
  Sudoku dropout-RNG confound, used-parent proposal mismatch, MAUVE precision correction and J1c amendment.
- Experimental proposals remain speculation; no registrations, training code or launch decisions changed.

## R035 — Session close: Codex at its usage limit, PR 6's review finding fixed, handoff written (2026-10-24; process)
- Codex's review of PR 7's head `f087a48` failed at the code-review usage limit (16:36 UTC). Per `CLAUDE.md`, no gated
  merge until it resets: W1d (`w1d_requant_diag`) is not cleared; the WMT14 pair is (PR 8 merged before the limit).
- PR 6: Codex's P1 (the exploratory scripts printed absolute output paths) accepted and fixed: a `shown()` helper prints
  repo-relative paths or file names only; `tests/test_exploratory_print_paths.py`, each print site mutation-checked
  (`docs/reviews/triage_pr6.md`). Its new head `8516fea` needs a review.
- Handoff for the next session: `docs/prompts/handoff_2026-10-24.md` (re-check the Codex work, the open PRs, next
  steps, and the literature report's frozen-teacher arm J2-F and mechanism 2x2 as decisions for the user).

## R036 — WMT14 pair, seed 0: J2 misses its screen; its pass saving comes partly from a collapse onto input-independent outputs (2026-10-24; registered J1 + J1a + J1b + J1c)
Runs `w2c_L_wmt_ctrl` (J0 re-evaluation, J1 control 3000 steps) and `w2c_L_wmt_consist` (J2, 2000 steps), A100-40GB,
torch 2.11, both at `c60f8b4` (PR 8 merged), one session each, no resumes. 132 A100-minutes, ~11.9 CU (planned ~9).
Results `results/colab/w2c_L_wmt_{ctrl,consist}/` (provenance `cmd` fields templated to `{R}`; the notebook's copies of
repository result files left out). Registered numbers recomputed by `scripts/analyze_j1.py` (PR 9, awaiting Codex)
into `results/j1_analysis_s0.json`; every number below is in that file or in the result files.

| | val passes | val BLEU | test passes | test BLEU | test length ratio |
|---|---|---|---|---|---|
| J0 released | 10.916 | 23.55 | 11.069 | 22.61 | 0.930 |
| J1 control | 11.527 | 22.82 | 11.622 | 21.81 | 0.930 |
| J2 consist | 9.357 | 18.53 | 8.890 | 16.02 | 0.750 |

**Registered readings**
- Sanity: J0 reproduces R020 (11.069 passes vs 11.07, BLEU 22.61 vs 22.63; no sentence converges on pass 1, so J1c's
  pass-1 rule changes nothing for J0). J1 loses 0.73 (val) / 0.80 (test) BLEU vs J0: under the 1-BLEU limit, the recipe
  is not damaging. J2: 32,000 trajectory sources, 0 degenerate (J1b's 1% rule satisfied); executed FLOP proxy equal for
  J1 and J2 (8.38e16). Exact differs from sequential for 1 J1 test sentence (1294) and 2 J2 val sentences (531, 1118);
  "exact" excludes them.
- **Screen (val): J2 fails both conditions** (passes 9.357 > 9.0; BLEU 18.53 < 22.55). J2 does not advance; J1 gets no
  seeds 1-2. **H12 kill condition met on test:** 19.2% NFE saving at a 6.59 BLEU loss (needs >= 25% at < 1). J2's passes
  are 2.73 below J1's, so that kill clause alone would not fire.
- J1a at cap ceil(8.89) = 9: J0 22.51 BLEU (8.99 passes), J1 21.56; J2 is 6.49 and 5.54 below them. Length ratio 0.18
  below J0's (limit 0.02). The paired bootstrap was not computed: the reference translations are not in the repository,
  and with differences of 5.5-6.5 BLEU it cannot change the reading. J0 at cap 8 loses 1.11 BLEU (not within 1), at 8.0
  passes (27% fewer than 11); at cap 9, 0.10 BLEU at 18% fewer. A fixed cap is not an adaptive rule (J1a), but it is the
  bar any consistency arm must beat.

**Exploratory (labelled; not registered analyses)**
- J2 decodes 8.7% of val and 16.2% of test sentences to one of two input-independent strings ("In 1999 , the-D-D-D-...
  ." for 428 test sentences, "Staff was very friendly and helpful." for 59), reached in ~3 passes on average. J0 and J1
  produce no string shared by >= 5 sentences. For 352 test sentences J2's first pass from the all-zero start already
  emits the first string. On the remaining sentences J2 averages 10.04 test passes (8.7% saving; J0 11.04) and emits 9%
  fewer words than J0; adjacent duplicated words rise 17.1 -> 25.6 per 1k (J1 21.0).
- Course of training (first 500 val sentences, the curve subset; J2's last curve point equals the final evaluation on
  those sentences): J2's passes fall to 9.81 by step 500 and drift to 9.64 by step 2000 (J1 11.34-11.52); the shared
  output share grows 0 -> 1.8 -> 1.6 -> 2.8%. The curve subset understates the collapse (2.8% vs 8.7% on all of val; its
  BLEU 24.19 vs 18.53).
- Reading: the self-consistency loss with the current model as its own teacher rewards fast convergence whatever the
  output; an input-independent output that is its own fixed point within a pass or two is a cheap solution, and once a
  source maps to it, the teacher's target for that source is that output. This is a moving-teacher feedback loop, the
  failure the literature report's frozen-teacher arm (J2-F) was proposed to separate. One seed; not established.
- Wall-clock: J2 took 1.37x J1's time (4202 s vs 3071 s, including evaluations) at an equal generator FLOP proxy.

**Decisions**
- J2 is dead for criterion (d) as registered. The user's decision (this session): register J2-F as a contingent arm that
  runs if J2 misses its screen; registered in `docs/preregistration.md` (J2-F), after this result, with that timing
  stated in the registration. J2's miss triggers it. Build and deep-audit it once Codex is back; then ~6.5 CU.
- Criterion (d), text half: no candidate is left besides J2-F.

**Review of the previous session's work (handoff section 2; an analyst agent's re-read, checked here before logging)**
- Codex triages re-read at their fix heads: audits 3, 4, 5 and PR 7's rows stand. Two corrections and two test gaps:
  - **Correction (R031, R034, `triage_audit3`):** "every arm alike, so it cannot make one arm collapse" is wrong for
    the policy forward. Wave 1's `policy_codes` (`scripts/train_gen.py`) runs inside `compute_loss`'s bf16 autocast
    with TF32 on, and only the on-policy arms have it. Precision is therefore an untested competitor for the on-policy
    collapse, not an excluded one (validation NLL, shared by all arms, is as stated). Logged as a re-check row in
    `docs/reviews/triage_audit3_2026-10-24.md` and `triage_adversarial_onpolicy_2026-10-24.md`.
  - **Correction (R031 item 5):** "the arm does not take larger steps" was inferred from lower loss and gradient norm;
    under AdamW the update size is not set by the gradient norm, and no update norms were saved. Withdrawn.
  - Test gaps: PR 5 has no test of the text `run_manifest.json` or of the missing-`policy_rng` resume branch (add before
    its review and deep audit); A4-3's padding has a row-count test but no test that the real rows' loss and gradient
    are unchanged by the padding (add with J2-F's trainer option, which reuses `consist_loss`).
- Riskier decisions: A4-3 padding (loss uses only the real rows; an all-degenerate step returns exactly `loss_tf` and
  still runs the registered rows; dropout draws for the real rows differ bitwise from an unpadded batch, not in
  distribution): agree. A4-4 `initial_fixed_ok`: changes only the pass count of sentences whose all-zero start is the
  fixed point, applied to the teacher and every J0 / J1 / J2 evaluation; measured effect on J0 here: none (no such
  sentence, 11.069 passes). Clearing the WMT14 pair on the R028 precedent: no objection found. Ported Codex tests:
  legitimate (marker removals, the one snapshot count, two structural assertions that the fixes made necessary); one
  more port this session (below).
- Wave-1 clean NLL under autocast: every R030 / R031 conclusion that rests on clean NLL involves differences of 24-154
  nats per window, against at most ~9 nats for precision and window packing combined (R008's 870.5 fp32 vs 861.8);
  none flips on scoring precision. R031 item 5's per-level split (+15 of +24 nats at L3-L6) is the least robust.
- R030's screen reading (comparator-relative entropy / MAUVE) was chosen after the data but reports both readings;
  moot after R033. R031's "replicated across domains" for the Sudoku contrast was too strong while ADV-1 was open:
  see the paired rerun below.
- Full suite on `claude/ncm-research`: 102 passed, 1 failed on both torch 2.6 and 2.11. The failure was caused by this
  session's results commit: the ported audit-4 numbers test compared the exact list of WMT-named result files, which
  the WMT14 results extend. Ported like its file count (the audited files must all still exist; mutation-checked by
  removing one); it now passes on both versions, the other 102 having passed in the full runs (`3a41304`).

**R031's Sudoku contrast, paired rerun with ADV-1 fixed** (exploratory; `scripts/onpolicy_repro.py` at PR 6's head
`8516fea`, CPU fp32, R031's settings: batch 16, lr 1e-4, 20 warmup, 200 steps, seed 0, the same 300 puzzles, 8 threads;
`results/onpolicy_repro/{uniform,plausible}_wrong_adv1_s0.json`). With the fix both arms draw replacements from their own
generator and share the dropout stream; the logged per-level corrupted fractions are identical in the two arms at every logged step (same masks).
- The uniform arm is unchanged to the printed digit (its draws already used their own generator): clean NLL 45.89,
  greedy 83.7%, T = 1 82.7%.
- **Hedging replicates:** clean teacher-forced NLL 43.2 -> 75.6 nats per puzzle (plausible) vs 45.9 (uniform); R031:
  75.2. The excess sits in L8-L10 as before (+24.6 nats). Predictive entropy per code, summed over levels: 2.09 -> 3.16
  (plausible; R031 3.15) vs 2.33 (uniform). Training loss 0.399 vs 0.308.
- **"Greedy decoding hides the hedge" does not replicate.** Paired change in greedy solve rate by step 200: uniform
  -4.7 ± 1.5, plausible -5.0 ± 1.5 (plausible minus uniform -0.3 ± 1.5; R031: +3.7 ± 1.6). At T = 1: uniform
  -4.3 ± 1.6, plausible -6.3 ± 1.6 (difference -2.0 ± 1.8; R031: -1.0 ± 2.0). R031's greedy advantage of the plausible
  arm came from the dropout-RNG confound. Corrected reading: at matched slots and rates, plausible wrong codes make
  the model hedge on clean contexts (clean NLL +30 nats vs +3), while solve rates fall by similar amounts in both arms
  (~5 points greedy, 4-6 sampled). One seed; the step from hedging to a collapse is still not shown.
- The first attempt ran the two arms in parallel; the uniform process exited with status 1 and no traceback after
  start-up. Run again alone, it finished normally; the plausible run from the parallel attempt is the one reported.
- CPU: ~1.0 h wall (two arms, 8 threads each), plus the two suite runs (~5 min).

## R037 — Codex back: PRs 5, 6, 7, 9 merged; J2-F built and deep-audited (2026-10-24; process)
- Codex's usage limit reset. Reviews: PR 7 (`f087a48`), PR 9 (`d2846c8`), PR 5 (`f007888`, after adding two tests for
  the text run manifest and the on-policy policy-RNG resume guard, each mutation-checked) and PR 6 (`8516fea`; its first
  request got no response, a second one was answered) all came back with no P0/P1; all four merged through the gate.
  **`w1d_requant_diag` is cleared** (deep audit 5 done; PR 5 does not touch W1d's imports). PR 5's text-trainer changes
  still need a deep audit before any text tier-L run uses them.
- **J2-F built** (PR 10, `claude/wip-j2f`): arm `consist_frozen` in `scripts/train_wmt14_L.py`, config, notebook
  `w2c_L_wmt_consist_frozen`, 13 code-gate tests (mutation-checked). Codex review of `3df998a`: no P0/P1.
- **Deep audit 6** (local Codex CLI, `codex/audit6`, brief `docs/prompts/codex_audit6_j2f.md`): two P1s, both reproduced
  (7 strict xfails) and fixed with the audit's patches: A6-1, a run directory without its first checkpoint accepted a
  different arm or base (inherited by every trainer with the same guard; fixed here for WMT14, `train_gen.py` and
  `train_sudoku_L.py` still have it); A6-2, the analysis could not select J2-F. Added a launch-time check that the frozen
  teacher reproduces J0's first 128 saved exact hypotheses and pass counts (gate 1 on the released weights). Triage
  `docs/reviews/triage_audit6_2026-10-24.md`. All R036 numbers recomputed by the audit without disagreement; one wording
  correction to R036: its aggregates are over all sentences ("exact excludes them" applies to the exactness claim only).
  Full suite 160 passed on torch 2.6 and 2.11. PR 10's round-2 review was requested at 23:59 UTC; merge pending.

## R038 — J2-F cleared: PR 10 merged, deep audit 7 clean; A6-1 guard for the other trainers in PR 11 (2026-10-25; process)
- PR 10 round 2 (`f50a27d`: audit 6's fixes, the launch-time teacher check, ported tests): Codex, no P0/P1; merged.
- The launch check changed `evaluate()` after audit 6, so the delta got its own deep audit (7, local Codex CLI,
  `codex/audit7`, brief `docs/prompts/codex_audit7_j2f_delta.md`): no P0/P1. Its sandbox could not run Python; its five
  regression probes and the WMT14 suites were run here: 72 passed on torch 2.6 and 2.11. Full suite 208 passed (torch
  2.6). Triage `docs/reviews/triage_audit7_2026-10-25.md`. **`w2c_L_wmt_consist_frozen` is cleared** (A100).
- The A6-1 guard (run identity checked only once a checkpoint exists) also sat in `scripts/train_gen.py` and
  `scripts/train_sudoku_L.py`; fixed with tests in PR 11 (review requested). No queued notebook uses those trainers.
- The user launched `w1d_requant_diag`.

## R039 — W1d: A2's Gen-PPL gain is a T = 1 effect; registered reading "trade-off", A2 not pursued (2026-10-26; registered W1d, exploratory)
- Run: `w1d_requant_diag`, A100-40GB, code `81a6979` (all three sessions), 2026-10-25 00:08-00:41 UTC, ~33 min
  (~3 CU with setup). Results `results/colab/w1d_requant_diag/` (the notebook's copies of four repo-level JSONs in
  `results/` were identical to them and are not duplicated). Numbers below recomputed from the result JSONs; paired
  differences are over the 8 sample seeds (frontier) or 4 window groups (oracle), mean ± SE, sampling noise only.
- **Integrity gate passes:** at T = 1 every group of A0, A1 and A2 reproduces its wave-1 Gen-PPL exactly (max |diff| 0.0,
  tol 0.01). The numbers are read.
- **Frontier** (Gen-PPL GPT-2 large; A0 / A1 / A2; A2 - A1 paired):
  | T | A0 | A1 | A2 | A2 - A1 | entropy A1 / A2 | MAUVE A1 / A2 |
  |---|---|---|---|---|---|---|
  | 0.8 | 13.40 | 13.60 | 13.82 | +0.22 ± 0.08 | 3.949 / 3.945 | 0.921 / 0.919 |
  | 0.9 | 14.34 | 14.79 | 14.84 | +0.05 ± 0.10 | 3.955 / 3.955 | 0.909 / 0.932 |
  | 1.0 | 16.02 | 16.56 | 16.23 | -0.33 ± 0.12 | 3.962 / 3.964 | 0.896 / 0.930 |
  Mean of the three paired differences -0.02 ± 0.08. TinyStories-33M Gen-PPL agrees in sign at every T (+0.42, +0.28,
  -0.24). Entropies are within 0.005 nats at every T; A2's MAUVE is not lower by more than 0.01 at any T (0.8: -0.002).
  A2 is worse than the released model at every T (+0.42, +0.50, +0.21).
- **Registered frontier reading: "trade-off"** (the Gen-PPL gain reverses at T = 0.8, and the mean over T is not 2 SE
  below 0). Not "A2 ahead": **W1c is not registered; A2 is reported as a likelihood / MAUVE shift without a Gen-PPL gain
  and is not pursued** (bet 4 dropped). The held A1 seeds 1-2 lose their W1c use; they remain the only source of A1's
  training SD for M1's gate.
- **Oracle curve** (levels 0..j real, rest sampled at T = 1; Gen-PPL GPT-2 large):
  | j | A0 | A1 | A2 | A2 - A1 | A1 - A0 |
  |---|---|---|---|---|---|
  | -1 | 15.96 | 16.66 | 16.52 | -0.14 ± 0.09 | +0.69 ± 0.11 |
  | 3 | 12.38 | 12.38 | 12.80 | +0.42 ± 0.17 | +0.00 ± 0.09 |
  | 5 | 9.52 | 9.43 | 9.54 | +0.11 ± 0.03 | -0.09 ± 0.01 |
  | 7 | 8.92 | 8.87 | 8.89 | +0.02 ± 0.01 | -0.05 ± 0.01 |
  Registered oracle reading: A2 - A1 does not survive to j = 5 (it reverses sign), so whatever gain A2 has arises while
  the model samples its own coarse levels. Caveat: the j = -1 gain itself is 1.4 SE, so this reading rests on a gain that
  this run barely resolves.
- **Exploratory (not registered):** (1) Given real levels 0..3, A2 renders levels 4+ *worse* than A1 (+0.42 ± 0.17); its
  only edge is when it conditions on its own sampled coarse codes. That fits re-quantized labels teaching the model to
  render consistently from wrong parents at some cost on right ones. One seed; a hypothesis, not a finding. (2) Wave 1's
  continued training (A1) loses 0.69 Gen-PPL to the released model with no oracle, but nothing once levels 0..3 are
  real and slightly gains at j >= 5: the control's damage sits in sampling levels 0-3. This argues against reading W1's
  shortfall to A0 as a fine-level effect. (3) The T = 1 advantage plus the T = 0.8 reversal at equal entropy looks like
  a change in how the sampled tail is shaped rather than a better model; untested.
- Compute: ~73 CU spent of ~200.

## R040 — W2M built and merged (PR 12); deep audit 8 briefed (2026-10-27; process)
- The W2M implementation left uncommitted in session 6 (`src/ncmr/w2m.py`, `build_hierarchy`'s `masks` / `canonical`,
  `corruption.mode: mech` in `scripts/train_gen.py`, `scripts/mech_eval_w2m.py`, eight configs, nine notebooks, tests)
  was tested (72 W2M tests; full suite 285 on CPU, torch 2.11) and opened as PR 12.
- Codex round 1, two P1s, both reproduced and accepted (`docs/reviews/triage_pr12.md`): (1) the flag / contamination
  footprint omitted the tokenizer's kernel-3 residual convolution, so flags were 0 at 305 positions the tokenizer
  actually changes (24-row test); this is the registration's definition, now implemented exactly, not an amendment.
  (2) measure (a) needed ~45 GiB at 512 windows; now batched, same value. Regression tests mutation-checked against
  `8b82b9c`. Round 2 (`2ede453`): no P0/P1. Merged (`31827b8`); full suite 290 pass.
- No wave-1 or WMT14 code path changed (`tests/test_w2m_old_arms.py`). **W2M notebooks are not cleared**: deep audit 8
  (brief `docs/prompts/codex_audit8_w2m.md`, audited `2ede453`, also covers PR 5's text-trainer changes) comes first.
  The Codex CLI is not installed on the laptop, so the audit runs in Codex desktop from that brief.
