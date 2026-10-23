# Triage of the tier-M design review (log R027)

Source: `gpt_tierM_review_2026-10-23.md`. Every cost was recomputed from the calibration JSONs and
`src/ncmr/flops.py`: 24.93 / 13.95 CU per seed, 116.6 CU for 3 x (NCM + AR), design C 7.65 / 4.28 CU with 27,335
FLOP-matched AR steps (NCM = 4.56x AR per window at d512 x 6). All agree.

| # | Finding | Verdict | Action |
|---|---|---|---|
| T1 | The 15k / 70k plan costs 116.6 CU for the baselines alone. | **accept** | `w2_M_ncm` / `w2_M_ar` removed; superseded by design C. |
| T2 | Design C (d512 x 6, 6k NCM steps, FLOP-matched AR) fits; costs are projections, recalibrate. | **accept** | `M_ncm_c.yaml`, `M_ar_c.yaml`; the NCM notebook runs a 30-step calibration first. Registered as M1. |
| T3 | No measured <= 25M gap exists; with a training SD of 0.59, half closure needs G > 2.36; a point estimate of 50% is not enough. | **accept** | M1 gate: seeds 1-2 of the baselines run only if the seed-0 gap G0 >= max(2.4, 4 x the pooled training SD measured by W1's control seeds). Half closure must be supported by the 95% interval, not the point estimate. This adds a cheaper stage than the review's "all three pairs first": a ~12 CU seed-0 pair decides whether the other ~24 CU are spent. |
| T4 | Training-seed SD is unknown; sampling SD is not training SD. | **accept** | W1's control seeds 1-2 give the first training-seed SD (tier L); M1's seeds 1-2 give tier M's. 16 x 256 samples per arm keep sampling noise at ~0.21 per difference. |
| T5 | Re-estimate the level-0 prior from training data. | **accept** | `train_gen.py`: `level0: train` estimates it from 262k training windows with its own RNG (data order unchanged); TV distance to the released prior is logged. |
| T6 | Save AR's data-exposure checkpoint; it is not an independently optimized short-run baseline. | **accept** | Snapshot at 6000 (data-matched), reported as secondary only. |
| T7 | Kill if AR improves >= 0.35 Gen-PPL in its final quarter. | **partial** | Under a cosine schedule much of the final-quarter gain is learning-rate annealing, so this would fire for a well-trained AR. Replaced: the 75% snapshot is reported as a curve point only. Before any criterion-(b) claim, a conditional AR at 1.5x the FLOP-matched steps (seed 0, ~6.4 CU, `w2_Mc_ar15`) is run, and half closure must also hold against its gap. |
| T8 | Shared released tokenizer -> claim is conditional on the tokenizer, not a full from-scratch system claim. | **accept** | Stated in M1. |
| T9 | FLOP proxy omits tokenizer/hierarchy work and undercounts dense masked attention. | **accept** | Same wording as W1b: proxy labelled, measured wall-clock per arm reported. |
| T10 | GPT-2 large primary, TinyStories-33M always reported; entropy-matched comparison; EOS, repetition, length, MAUVE. | **already covered** | W1b protocol (16 groups, entropy-matching temperature chosen on separate seeds) applies to M1. |
| T11 | Success also needs <= 42 NFE at 128 tokens. | **accept** | In M1 (NCM uses 11; any candidate must state its NFE and decoder work). |
| T12 | Parameter counts: body 25.2M; AR 29.4M with embeddings; C's AR ~23.1M; tokenizer 36.4M not included. | **accept** | M1 states parameter counts that way. |
