<!-- Paste everything below this comment into Codex (desktop app, or `scripts/review/codex_local.sh audit audit8
docs/prompts/codex_audit8_w2m.md` from a checkout of the audited commit). Work in a new branch `codex/audit8` made
from the audited commit, and commit only the audit doc and its tests there. -->
Deep audit. Follow '## Deep audit protocol' and '## Code Review Rules' in AGENTS.md. Audited commit: 2ede453bac101319b2871a873d275e45850d280e
(head of `claude/wip-w2m`). Audit id: audit8. This repository is PUBLIC. Write no absolute local paths, usernames,
emails, session links or credentials into any file or into your final message; refer to files by repo-relative path.
Work on a new branch `codex/audit8` made from the audited commit; commit there and push it.
If your sandbox cannot run Python, say so in the audit doc and still write the tests; they will be run for you.

Deep audit 8 before the W2M notebooks `w2m_L_ts_m0` ... `w2m_L_ts_m7` and `w2m_mech_eval` (registered "W2M" in
`docs/preregistration.md`). Follow "Deep audit protocol" and "Code Review Rules" in AGENTS.md. Push only
`docs/reviews/codex_audit8.md` and `tests/test_audit8_*.py`; change no other file.

## Why
Wave 1's on-policy arms collapsed (Gen-PPL 32-41 vs 16.6; R030, R031). W2M's eight 600-step tier-L arms test why
(plausible-wrong vs uniform-wrong replacements, truthful vs shuffled flags, used-parent proposals, A3's realized rates).
~27 CU, the largest spend since wave 1, and it is a candidate for criterion (c). It changes the text trainer, the
corruption code and the sampler path, so CLAUDE.md's Tier-2 rule applies before launch. It also covers PR 5's
text-trainer changes (fp32 policy forward, separate policy RNG), which no deep audit has seen. Tier-1 review: the W2M PR.

## Subject
Head of branch `claude/wip-w2m`. Files: `src/ncmr/w2m.py` (new), `src/ncmr/corruption.py` (`build_hierarchy`'s
`masks=` / `canonical=` arguments), `scripts/train_gen.py` (`corruption.mode: mech`, `compute_loss_mech`, the plan /
micro-batch slicing, checkpoint `mech` state, `sample_eval.at`), `scripts/mech_eval_w2m.py` (new),
`configs/scale/L_ts_m{0..7}_*.yaml`, `notebooks/make_notebooks.py` and `notebooks/w2m_*.ipynb`, tests
`tests/test_w2m_arms.py`, `tests/test_w2m_old_arms.py`, `tests/test_w2m_train_loop.py`, `tests/w2m_support.py`,
`tests/w2m_ref_corruption.py`, `tests/test_w2m_pr12.py`, and the PR 12 triage `docs/reviews/triage_pr12.md` (two P1s fixed). References: wave 1's arms in the same trainer (audited in audits 1-3) and their seed-0
results `results/colab/w1_L_ts_*/`. Log entries to recompute: R039 (from `results/colab/w1d_requant_diag/`).

## Spec (what correct means)
The registration "W2M" in `docs/preregistration.md` is the spec. What it leaves implicit:
- **Equal data and masks.** For one seed, every arm draws the same batches in the same order and, except M7, masks
  exactly the same slots. Masks and the 25% clean rows come from one mask stream that no proposal, replacement, flag
  shuffle, dropout or evaluation draw advances. With `grad_accum > 1` the plan is drawn once per optimizer step and
  sliced per micro-batch, so the result must not depend on `grad_accum`.
- **Replacements.** M0 / M7: uniform (M0 may hit the canonical code, as in W1; M7's per-level rate is
  d_k / (1 - 1/V_k) on corrupted rows). M1 / M6: the current model's one-step samples at T = 1 given the clean coarser
  levels (M6: given the level-0 codes actually used), level 0 from the unigram; not excluded. M2 / M4 / M5: a frozen
  copy of the released generator, canonical excluded (level 0: unigram with canonical excluded). M3: uniform excluding
  canonical. Labels are canonical in every arm.
- **Precision.** Every proposal forward (current policy and frozen copy) runs in fp32 with TF32 off and autocast
  disabled, in eval mode with no grad, and leaves the training model's mode, the dropout RNG and the global RNG
  untouched, even though `compute_loss_mech` is called inside the trainer's bf16 autocast context.
- **Frozen copy.** Copied from the released weights before any checkpoint restore; never updated; identical after a
  resume; shares no parameter, buffer or cached tensor with the student.
- **Flags (M4, M5).** Per input position, the share of its full-length span whose cumulative reconstruction contains a
  replaced ancestor, propagated with the tokenizer's own up-sampling and the area down-sampling that builds the inputs;
  a function of the sampled mask only. They enter as `f * w`, `w` a zero-initialised trainable vector (step 0 equals the
  released model's logits). Inference, every curve / final evaluation, val NLL and Gen-PPL sampling use zero flags. M5
  permutes each row's flags within each level with its own RNG.
- **Resume.** Mask, replacement and flag-shuffle RNGs, `w`, and the realized-disagreement accumulators are in the one
  atomic checkpoint; a resumed run equals an uninterrupted one; a checkpoint without that state is refused; the run
  manifest refuses a different arm or config (audit 6's A6-1 pattern, PR 11).
- **Accounting.** The FLOP proxy adds the proposal forwards each arm actually executes (labelled a proxy; arms are
  matched on updates, not FLOPs). Realized per-level disagreement is logged for all rows and for corrupted rows.
- **Evaluation.** Curve at steps 100 / 300 / 600 (256 samples, val NLL); final at 600 with W1b's protocol (16 x 256
  fp32 samples, seeds 42 and 0-14). Mechanism eval: same windows, perturbations, corrupted contexts and probe split for
  all nine models, built once from the released generator; (a) KL(clean || perturbed) over levels > j; (b) probe on
  last hidden states, train windows 0-383, AUC on 384-511; (c) true-flag NLL minus zero-flag NLL (negative = flags used).
  Val windows are the even-index split only.

## Questions
1. Does any proposal forward run under bf16 autocast, or leak autocast / TF32 / mode state into the student's forward
   (R024's defect class)? Check `MechArm.proposals`, the frozen copy, `torch.compile` (`mech.fwd = fnet`), and the
   order in which `trainer.ctx` is entered.
2. Is the mask stream really independent of everything else (proposal sampling, flag shuffles, M7's different rates,
   `grad_accum`, evaluation cadence)? Is it identical across M0-M6 for seed 0, and does M7 share the clean rows?
3. Are M2 / M3 / M4 / M5 replacements always wrong at every masked slot, including level 0 and ties; is M0 / M1 / M6
   not excluded; does M6 condition on the used level 0 and M1 on the canonical one, exactly as registered?
4. Are the flags exactly the propagated sampled mask (compare to a direct computation through the tokenizer's
   reconstruction), and is there any path where non-zero flags or `w` reach evaluation, sampling or val NLL? Is `w`
   excluded from weight decay as the tests claim, and does that match what the registration implies?
5. Does a resumed run equal an uninterrupted one for every arm (all new RNGs, `w`, frozen copy, accumulators)? Is the
   frozen copy taken before `maybe_resume`?
6. Do wave 1's arms (configs other than `L_ts_m*`) still run bit-identically through the changed `build_hierarchy` and
   training loop?
7. Does `scripts/mech_eval_w2m.py` implement (a)-(c) as registered (KL direction, levels > j, finer inputs rebuilt,
   probe split, zero flags for (b), flag-use sign), with identical inputs for all nine models, in fp32 with TF32 off?
8. Do the notebooks run each arm with its own config and seed 0, record provenance, refuse resumes under changed code,
   and save everything the readings need (per-seed Gen-PPL for the paired D(X), realized disagreement, bigram gain,
   adjacent duplicates, per-level clean val NLL, the mechanism JSON)?
9. Do the 43 W2M tests exercise production code, and would each fail if its property broke? Name any registered
   code-gate property that is untested.
10. Recompute R039's tables (paired A2 - A1 per T and per oracle j, mean ± SE over groups) from
    `results/colab/w1d_requant_diag/w1d_A*__result.json`; list any disagreement.

## Out of scope
Style, naming, performance tuning, arms and notebooks other than W2M's, and the scientific design of W2M itself
(registered; raise design doubts in one short section at the end, not as findings).
