<!-- Run with `scripts/review/codex_local.sh audit audit15 docs/prompts/codex_audit15_builds.md` from a clean checkout of a
scratch branch that contains the three build branches (W2M-GX, PD, E0). Codex commits only the audit doc and tests. -->
Deep audit 15 before launching W2M-GX, PD and E0 (registered at the end of `docs/preregistration.md`). Follow "Deep
audit protocol" and "Code Review Rules" in `AGENTS.md`. Push only `docs/reviews/codex_audit15.md` and
`tests/test_audit15_*.py`; change no other file. The code was written by a Codex build session: audit it as you would
code by anyone else, and do not trust its PR descriptions or tests without running them.

## Subject
- The audited commit is HEAD of the scratch branch.
- Diff against `origin/claude/ncm-research`. List the files from `git diff --stat`. Expect at least:
  - `scripts/w2mg_eval.py`, `src/ncmr/w2mg.py`, `scripts/analyze_w2mgx.py`;
  - `src/ncmr/w2m.py`, `scripts/train_gen.py`, the generator model file;
  - `scripts/pd_coverage.py`, `scripts/analyze_pd.py`, `scripts/analyze_e0.py`;
  - `configs/scale/L_ts_pd_*.yaml`, `configs/scale/L_ts_e0_*.yaml`;
  - `notebooks/make_notebooks.py` and the generated notebooks.
- No results exist yet.

## Spec
The three registrations are the spec, with W2M-G's protocol (for W2M-GX) and W2M's protocol (for PD and E0)
unchanged otherwise.

## Questions
1. **W2M-GX: is the per-level model assignment exact?**
   - Do the existing two-model cut cells equal the new path bit for bit, with the same noise, on CPU?
   - Are the cross-seed cells' noise streams identical to W2M-GR's for the same replay seed?
   - Is each checkpoint's SHA-256 recorded and checked?
   - Are kappa, pi_6(s, t), sigma_6(s, t), e_up and e_down computed exactly as registered, including delta_bar?
   - Do the readings match the registered thresholds, inclusive / strict as written?
   - Does admission refuse wrong GPUs, replay seeds, group counts, cuts and checkpoint mappings?
   - Does the collector now keep the per-window arrays?
2. **PD: matched comparison.** Run an executed trace of the production training loop for P, D and U at seed 0 on a tiny
   model, as audit 12 did.
   - Batches, masks, clean rows and every pre-existing RNG state must be identical across arms.
   - D's levels 0, 1 and 6-11 must equal P's.
   - At levels 2-5, D's replacement multiset per (batch, level, position) must equal P's.
   - No slot may get its canonical code, and coverage must be computed as registered.
   - Is the derangement stream in the atomic checkpoint, and does a resumed D equal an uninterrupted D?
   - Do U's 175 steps equal the first 175 of M3's protocol?
   - Does `analyze_pd.py` implement r, validity, rescue, kill and inconclusive exactly?
3. **E0: is the erasure exactly what is registered?**
   - Is the zero vector applied to the masked L2-L5 code-embedding contributions before upsampling / convolution, and
     nowhere else?
   - Are targets, masks, rates and data identical to U0's? Is U0 M0's recipe with `clean_row_frac: 0.0` and the
     upstream replacement convention?
   - Is inference unchanged: identical forwards for identical weights?
   - Are FLOP / NFE proxies unchanged and labelled?
   - Resume equality.
4. **Precision and evaluation.** Do all three keep fp32 sampling and scoring with TF32 off, and no autocast leak from
   training into the final evaluation (R024)? Is the evaluation identical across arms? Is val the even-index split?
5. **Notebooks.**
   - GPU guards as registered: L4 whole-word for W2M-GX; A100 with `--micro_batch 64` for PD and E0.
   - Caps as registered.
   - Provenance and refuse-to-resume-under-changed-code.
   - The runtime-release cell.
   - Drive paths that cannot collide with existing runs.
6. **Anything that makes a registered reading unreadable or wrong**, such as a missing per-item output or a quantity
   computed on the wrong bank or split.

## Out of scope
Style and performance; code unchanged from `origin/claude/ncm-research` except where a question needs it.
