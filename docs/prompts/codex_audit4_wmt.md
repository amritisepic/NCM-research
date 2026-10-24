Deep audit 4 before wave 2c (`w2c_L_wmt_ctrl`, `w2c_L_wmt_consist`: WMT14 tier L, registered J1 + J1a + J1b). Follow
"Deep audit protocol" and "Code Review Rules" in AGENTS.md. Push only `docs/reviews/codex_audit4.md`,
`docs/reviews/codex_audit4_numbers.json` and `tests/test_audit4_*.py`; change no other file.

## Why
The WMT14 trainer was changed after the last audit that covered it (audit 2, R028 fixes), and audit 3 excluded it.
J2 is the experiment most likely to produce a success-bar claim (criterion d, text half), so it must be audited
before launch. Wave 1 (R030, R031) also showed that training on a model's own intermediate codes can destroy sampling
when the targets are not identifiable from what the model sees; J2 trains on the model's own Jacobi states.

## Subject
Commit `c91126a` (head of `claude/ncm-research`). Files: `scripts/train_wmt14_L.py`, `scripts/prep_wmt14.py`,
`src/ncmr/jacobi_train.py`, `src/ncmr/train.py`, `configs/scale/L_wmt_ctrl.yaml`, `configs/scale/L_wmt_consist.yaml`,
the `w2c_*` entries of `notebooks/make_notebooks.py`. Log entries whose numbers to recompute: R020 (WMT14 exact Jacobi
11.07 passes, BLEU 22.63, stable-output stop 10.59 / 22.64) and the WMT14 parts of R028, from `results/**.json`.

## Spec (what correct means)
- Registration: `docs/preregistration.md` J1, J1a, J1b. Both arms draw one common batch of 128 per step; J2 uses rows
  1-64 for teacher forcing and every 4th of rows 65-128 as trajectory sources; equal generator-proxy FLOPs
  (control 3000 x 384 = J2 2000 x 576 forward rows).
- J2 teacher: the fp32 greedy map (autocast off, TF32 off) run for exactly K + 1 = 12 passes, stopped sources holding
  their codes; training states sampled uniformly from the states strictly before the first state equal to the fixed
  point; target = the fixed point; degenerate sources (all-zero start already the fixed point) dropped and counted.
- Exact decoding: stop when no code changes; output must equal the model's sequential greedy output, checked per
  sentence; NFE = passes including the one that detects convergence; every cap 1-12 keeps sentence-ordered hypotheses
  (val and test) so J1a's paired bootstrap at ceil(J2's mean passes) is possible; length ratio reported.
- A resumed run must equal an uninterrupted one (model, optimizer, data RNG, every RNG the state sampling uses).
- Val = newstest2013 (selection), test = newstest2014, scored once at the end; nothing tunes on test.

## Questions
1. Does `consist_loss` implement J1b exactly (state indexing: input of pass t+1 = output of pass t; uniform sampling
   strictly before the fixed point; K + 1 passes; degenerate handling; the row split)? Write a test with a stub network
   whose Jacobi trajectory is known.
2. Precision: is any no-grad forward run inside the training autocast region with autocast *enabled*? On torch < 2.11,
   autocast caches weight casts made under no-grad without their autograd link and the training forward in the same
   region reuses them (audit 3 measured gradients reaching only ~0.1% of a Sudoku model's parameters on torch 2.6).
   Check that every parameter receives a gradient in a J2 step, on the installed torch and by reasoning for older
   versions. Is `val_nll` scored under the training autocast (as audit 3 found for the text trainer, A3-1)?
3. RNG and resume: the state sampling uses `torch.rand` on the global generator, which dropout also uses. Does a
   resumed J2 run draw the same states as an uninterrupted one? Is anything in the checkpoint missing?
4. Identifiability (wave-1 lesson): is the J2 target a deterministic function of what the model sees at that step
   (source prefix plus input state), for every position, including padding, EOS and degenerate cases? Is there any
   position where the label is not determined by the input, so the model would learn to discount its own codes?
5. Evaluation: does exact Jacobi equal sequential greedy for every sentence (fp32, TF32 off); are the capped
   hypotheses, per-sentence passes, length ratio and BLEU (the w0 protocol) computed and saved as J1a / J1b require?
   Does `--eval_only` refuse a trained run directory, and does the manifest refuse a changed config or data archive?
6. FLOP / NFE accounting: does the executed work match the registered proxy (no early-stopped teacher counted at full
   budget), and is any J2 cost (teacher passes in fp32) omitted from the wall-clock that J1 reports?

## Numbers
Recompute R020's WMT14 numbers and R028's WMT14 smoke checks from `results/**.json`; list disagreements. Save to
`docs/reviews/codex_audit4_numbers.json`.

## Out of scope
Style, naming, performance tuning, the TinyStories and Sudoku trainers, tier M.
