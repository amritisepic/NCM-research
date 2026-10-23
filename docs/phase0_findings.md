# Phase 0 findings (released NCM checkpoints)

All numbers measured by us unless marked. Run IDs refer to `log.md`; raw results in `results/`.

## 1. Sudoku: reproduced, and difficulty = serial depth (R004)
- Released Sudoku-hard checkpoint, 2000 released test boards, greedy, fp32 on CPU: **86.30% solved**
  (paper 86.1%; independent reproduction 86.3%). NFE = 12 (one generator pass per level).
- The released test sets reuse the *same* 2000 solution grids across easy/medium/hard and differ only in
  the number of blanks (41/46/51). Every puzzle has a unique solution.
- **What predicts failure is constraint-propagation depth.** We solve each puzzle by repeated parallel
  "singles" (fill every cell that is forced by its row/column/box, all at once; count rounds). Solve rate
  by depth: 99.0% at 5-6 rounds, 95.6% at 7-8, 84.8% at 9-10, 63.9% at >=11, and 50.2% on the 14.5% of
  puzzles singles cannot finish (they need search). Rank correlation of depth with failure 0.30 (p ~ 1e-35).
  Plain reading: each NCM level is one parallel pass; puzzles that need many sequential inference steps
  exceed what 12 passes x 8 layers deliver.
- Failures are self-detectable: the decoder's least-confident cell flags failures with AUC 0.87.
- **No early exit is possible with this hierarchy:** decoding the truncated hierarchy gives 0% solved
  boards before the last level (cell accuracy 82% after level 10, 98% after level 11). A level-skipping
  exit (e.g. upstream PR #12) never fires.

## 2. TinyStories tokenizer: near-lossless, lookup encoder, bag-of-words coarse codes (R002, R003)
- Encoder = token + position embedding lookup, GroupNorm, then 12-level residual VQ (lengths 1..128,
  4096 codes each); decoder = a single transformer block. 36.4M parameters incl. 18.9M EMA codebook entries.
- Full-depth reconstruction is essentially exact: 184 errors per million tokens; names 918 ppm, rare
  words 4588 ppm (25x the average, but still 0.46%), numerals 0. => The NCM-vs-AR quality gap on
  TinyStories cannot come from tokenizer loss; it comes from generating the codes.
- Truncated decodes recover tokens gradually (level 2: 22%, level 4: 56%, level 7: 94%, level 10: 99%);
  rare words lag (76% at level 7).
- EOS is reconstructed 100%, PAD 0% (the decoder emits "Once upon" after an EOS) => training windows were
  packed story-start windows. We use the same format.
- **Coarse codes behave like bags of words.** Shuffling the word order inside every 16-token window leaves
  73% of level-1 codes unchanged; shuffling inside 8-token windows leaves 89% of level-2 codes unchanged.
  A single meaning-preserving synonym swap changes the affected window's code ~27-35% of the time. So
  coarse levels are more sensitive to which words occur than to how they are ordered. This follows from
  the architecture: a window average of context-free token embeddings is order-invariant (up to the
  normalization and a learned level-0 downsampler).

## 3. Cost structure (analytic)
- With NCM's level schedule the generator processes 641 positions per 128-token window, so at equal
  parameters NCM spends ~4.25x the training FLOPs of AR per window and ~4.3x the sampling FLOPs per
  sample. NCM's advantage is sequential depth (11 NFE vs 128), which matters on parallel hardware.
- Matching parameters, data and FLOPs simultaneously is therefore impossible; we match parameters and
  training FLOPs, and report the data-matched AR point from its learning curve.
