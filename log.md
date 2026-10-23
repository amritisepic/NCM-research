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
