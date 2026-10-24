Adversarial analysis (not a code audit): try to break our explanation of the wave-1 on-policy collapse. Follow
"Code Review Rules" in AGENTS.md for any code finding. Push only `docs/reviews/codex_adversarial_onpolicy.md` and, if
you write tests or experiments, `tests/test_adv_onpolicy_*.py`; change no other file. CPU only, under ~30 CPU-minutes.

## Claim to attack
Log R031 (and R030 for the registered numbers). Every on-policy arm of wave 1 collapsed (Gen-PPL 32-41 vs 16.6 for the
uniform-corruption control). Our leading explanation, (b'): uniform random replacement codes are easy to recognise, so
the model learns to repair codes that look random and trust codes that look plausible; the model's own samples look
exactly like the codes it produces at inference, so with targets the input does not determine it learns to discount
every plausible coarse code by roughly the corruption rate, and at inference fine codes stop fitting its own coarse
plan. R031 lists the evidence (local-only damage in the samples, clean-NLL damage at L3-L6, A4's re-quantized labels
removing only ~19% of A3's damage, A5's smaller damage) and a CPU Sudoku test (`scripts/onpolicy_repro.py`, results in
`results/onpolicy_repro/`).

## What we want from you
1. The strongest alternative explanations that fit the same data at least as well, each with what it predicts that
   (b') does not. Candidates we have only partly excluded: a defect not covered by audit 3 (read `docs/reviews/
   codex_audit3.md`), loss of robustness to off-manifold codes from milder corruption, a policy feedback loop, an
   interaction of the corruption mask with level 0 (the policy's level 1 is sampled given the canonical level 0, while
   the used hierarchy carries a unigram level 0 at 75% of windows), and anything specific to residual quantization.
2. Check the Sudoku CPU test for confounds: are `uniform_wrong` and `plausible_wrong` matched in slots, rates, data
   order and optimizer state except for the choice of the wrong code? Does excluding the canonical code bias the
   plausible codes in a way that explains the result without (b')? Is 200 steps at batch 16 and lr 1e-4 long enough,
   and is the step-0 reference shared?
3. Design the single cheapest test that would falsify (b'), runnable on CPU with the released Sudoku model (hash-pinned
   assets in `tests/conftest.py`), and run it if it fits the budget. Report numbers with the seeds used.
4. Say plainly whether (b') survives, and what result in the planned tier-L arms (on-policy + "replaced" flag;
   plausible-wrong codes at uniform rates; uniform corruption at A3's lower rates) would overturn it.
