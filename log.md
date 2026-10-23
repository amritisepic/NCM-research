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
