# Phase 1b at ~200 CU: the scaled plan (2026-10-23)

Supersedes the run plan in `phase1b_reevaluation.md` (the bets B1-B7 are defined there). Budget raised from
100 to ~200 Colab compute units (CU) by the user. Rules unchanged: pre-registration, one axis at a time,
>= 3 seeds before any claim, ask before any single run > 15 CU.

## 1. What 200 CU buys, and what it does not
Colab burn rates (third-party measurements, March 2026; the Colab resource panel is authoritative): T4 1.19,
L4 1.71, A100-40GB 5.40, A100-80GB 7.52 CU/h. So 200 CU is about 117 L4-hours or 37 A100-hours, roughly
10-16 EFLOP of useful compute at plausible utilization (wave 0 measures the real number). L4 and A100 cost
about the same per FLOP; A100 is ~3x faster in wall-clock; T4 is about half as cost-efficient (no bf16).

| Run | FLOPs | CU (L4, est.) | Fits? |
|---|---|---|---|
| Paper's TinyStories generator from scratch (88.6M, 50k x 2048 windows) | ~34 EFLOP | 400+ | no |
| One arm of the paper's own ablation protocol (88.6M, 10k x 1024), 1 seed | ~3.5 EFLOP | ~40 | not worth it: no seeds, no other work |
| **Tier L:** fine-tune the released 88.6M generator, 4000 x 256 windows | 0.35 EFLOP | 4.5-7 | yes, 3 seeds per arm |
| **Tier M:** 25M generator from scratch on the released tokenizer, 15k x 256 | 0.41 EFLOP | 6-10 | yes, 3 seeds per arm |
| Sudoku: fine-tune the released 25.5M generator, 5000 x 256 | 0.08 EFLOP | ~1.3 | yes |

So "going big" means: **test training objectives at the paper's scale, on the paper's own trained models**,
and **train properly matched, multi-seed from-scratch baselines at <= 25M** for the AR comparison (criterion
b). It does not mean a bigger model trained from scratch: that would give one unreplicated run.

## 2. Scale-limit change (needs your confirmation)
Tier L fine-tunes the released 88.6M generator; tiers L and M use the released 36.4M tokenizer (frozen). The
brief capped tokenizer <= 10M and generator <= 25M parameters. I read "go big" as approval. If you meant
budget only, tier L drops out and tier M switches to our 8.4M tokenizer (same code, `--tokenizer`).

## 3. New evidence that re-ranks the bets (R014, CPU, released TinyStories model)
- The tokenizer's decoder barely needs any single level: replace all codes of any one level by random codes,
  keep the rest, and 99.2-99.97% of tokens survive. Replace *every* level but the finest: random codes keep
  42.6% of tokens; the generator's own one-step samples keep **98.0%** (although 58% of codes change).
- The generator's samples disagree with the encoder's codes at 91-98% of positions in levels 1-2 (it picks
  different content), ~21% at L7 (64 codes = 2 tokens each), **45-55% at the token-straddling L8-L10**, 14%
  at the token level. At fine levels its choices are near-equivalent codes (80-97% within the 16 nearest).
- Re-quantized labels repair corruption only partly: reconstruction ceiling 92.9% -> 96.9% under NCM's
  uniform recipe, 99.40% -> 99.84% under on-policy corruption.

What this means:
1. **NCM's uniform corruption trains on errors the model never makes.** Uniform codes destroy 57% of
   tokens; the model's own one-step errors destroy 2%. On-policy corruption (the paper's stated future work)
   is the closer match, and the real inference errors are *compounded* ones (level k built on sampled,
   not true, coarser levels), so a two-pass on-policy arm is added.
2. Per-code noise at fine levels is mostly harmless. If samples are still much worse than AR, the damage
   must come from **incoherent coarse plans** (independent parallel choices of content within a level) and
   from compounding. Wave 0's teacher-forced-levels diagnostic measures exactly where quality is lost.
3. B1b (re-quantized labels) and B7 (soft labels) lose prior: the ceiling gain is modest, and soft labels
   spread probability over codes the model already treats as near-equivalent.

## 4. Tiers
| Tier | Model | Purpose | Cost/run |
|---|---|---|---|
| S | our 8.4M tokenizer + 5.5M generator, from scratch | tokenizer bets (B3, B4, C6, token-aligned schedule) | <= 1 CU |
| L | released checkpoints (TinyStories 88.6M, Sudoku 25.5M, WMT14 120M), fine-tuned | training objectives at paper scale; control = same fine-tune with the paper's recipe, same FLOPs | 1.3-7 CU |
| M | released tokenizer + 25M generator from scratch; AR with the same body, FLOP-matched | criterion (b) and from-scratch confirmation of tier-L winners | 6-10 CU |

A tier-L win says the objective improves a strong, converged model; a tier-M win says it trains better
models from scratch. A claim needs both, or is stated as only one.

## 5. Bets, re-ranked (prior = chance of a claim that survives Phase 3)
| # | Bet | Test | Kill condition | Prior (was) |
|---|---|---|---|---|
| 1 | **B1a on-policy corruption** (one-step and two-pass) | tier L factorial, then M | no arm beats the control by >= 0.5 Gen-PPL at matched entropy | 35% (45% for B1 as a whole) |
| 2 | B2-trained: consistency fine-tune so exact Jacobi converges early on **text** (criterion d) | tier L on WMT14 (greedy, BLEU) | < 25% fewer passes at < 1 BLEU loss | 25% (20%) |
| 3 | B6/C5 Sudoku: full-resolution repair passes with halting (+ C4 carry if its pilot passes) | tier L Sudoku fine-tune | hard < +3 points at <= 12 mean NFE | 30% |
| 4 | C2 red-black within-level sub-steps (attacks incoherent plans) | tier L, only if P1 finds TC >= 30% of the gap or wave 0 locates the loss at coarse levels | no gain over an NFE-matched baseline | 30% (40%, conditional) |
| 5 | B1b re-quantized labels | in the same factorial | A2 not better than A1 | 20% (part of B1) |
| 6 | B7 soft labels (tau 0.25, top 16) | one arm in the factorial | A6 not better than A1 | 20% (30%) |
| 7 | Token-aligned level schedule (are the straddling levels the problem?) | tier S pair | no Gen-PPL gain at equal K | 25% |
| 8 | B3/B4 tokenizers (high risk) | tier S | as in phase1b_reevaluation.md | 15% |
| 9 | B5 exact AR sampling from NCM drafts (hybrid, reported separately) | tier M, once AR-m exists | mean accepted run < 4 tokens | 30% |

## 6. Waves (notebooks in `notebooks/`, queue in `colab_runs.md`)
| Wave | Notebook(s) | What | CU |
|---|---|---|---|
| 0 | `w0_calib_diag` | released reference with the wave-1 protocol; throughput calibration (L, M, AR); where quality is lost (real coarse levels 0..j + sampled rest); exact Jacobi on full WMT14 test, TinyStories, Sudoku-hard 2000 | ~2.5 |
| 1 | `w1_L_ts_{ctrl,requant,onpol,b1,onpol2,soft}` | tier-L factorial, seed 0 (pre-registered: `preregistration.md` W1) | 27-42 |
| 1b | same notebooks, SEEDS = [1, 2] | control + arms that pass the screen | 18-42 |
| 2 | `w2_M_ncm`, `w2_M_ar` (+ winner) | from-scratch matched baselines, 3 seeds each; winner arm if wave 1 has one | 36-90 |
| 2b | Sudoku tier L (to build after C4 reports) | control / on-policy / repair passes, 3 seeds | ~12 |
| 2c | WMT14 consistency fine-tune (to build) | B2-trained for criterion (d) | 10-15 |
| 3 | tier S tokenizer bets; C2 if triggered; B5 | | 10-25 |
| | reserve for re-runs and failures | | ~20 |

Expected total 150-200 CU, depending on what dies. No single run is planned above 15 CU. Wave 0 comes
first: if measured throughput is < 60% of the estimate, step counts are cut before wave 1 (not the seeds).
Wave 2 baselines (NCM-m, AR-m) do not depend on wave 1 and can run in parallel if your plan allows
several sessions.

## 7. What changes the plan
- P1: TC >= 30% of the NCM-AR gap -> C2 moves to wave 2 (tier L). TC < 10% -> C2 and C7 dead.
- C4: support -> carry joins the Sudoku tier-L arms; failure -> killed for Sudoku.
- Wave 1: no arm passes the screen -> B1/B7 killed at tier L, wave 2 runs only the baselines, and the freed
  budget goes to B2-trained (criterion d) and the tokenizer bets.
- Wave 0 diagnostic: if most of the quality loss happens when levels 1-3 are sampled (incoherent plans),
  C2 and the tokenizer bets gain priority over B1.
