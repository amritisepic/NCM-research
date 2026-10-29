<!-- Handoff prompt for the Codex desktop app (agent mode, network on for git / gh). Open the repository, start from a fresh
worktree of `origin/claude/ncm-research`, and paste everything below this comment. -->
You are the lead agent for building three registered experiments in the NCM research repository. You plan, spawn
sub-agents, check their work and report. Read first: `AGENTS.md` (Code Review Rules; the repository is **public**, so
no absolute local paths, usernames, emails or session links anywhere), `STATE.md`, `docs/brief.md`, and the last three
sections of `docs/preregistration.md`: **W2M-GX**, **PD** and **E0**. Those registrations are the spec.

## Rules for every agent
- **The spec.** Implement exactly the registration. Never edit `docs/preregistration.md`, `docs/brief.md`, `AGENTS.md`,
  `CLAUDE.md`, `STATE.md` or `log.md`, and never choose thresholds, arms, seeds or step counts. If a registration is
  ambiguous, infeasible or contradicts the code, stop that build and put the question at the top of its PR
  description.
- **Branches.** Work only on `claude/wip-<topic>` branches. **Never push to `claude/ncm-research` and never merge**:
  Colab notebooks clone that branch, and the user's other agent merges after the review gate and the deep audit.
- **Tests.** CPU only, using `tests/conftest.py`, every wait bounded.
  - One test per registered reading, at its boundaries.
  - For trainer changes, an executed trace on a tiny model. It must show that arms with the same seed see identical
    batches, masks, clean rows and RNG states, as `tests/test_audit12_matched.py` does.
  - A resume-equals-uninterrupted test for every new piece of training state or RNG stream.
  - A test that the run directory refuses a different config, arm or base checkpoint.
- **Mutation checks.** For each reading test: break the behaviour, confirm the test fails, revert. List the mutations
  in the PR body.
- **Full suite** before opening a PR:
  `python -m pytest tests --ignore=tests/_deps -q` with `PYTHONPATH` = `tests/_deps`, `tests` and `src`. On Windows
  the separator is `;`. Some tests need git-ignored assets that a fresh worktree lacks: `tests/_assets/sudoku-hard`,
  `artifacts/tinystories` and `data/tinystories`. Copy or link them from the main checkout. Report anything skipped
  and why.
- **Notebooks.** Add specs to `notebooks/make_notebooks.py` and regenerate. Each spec needs its registered GPU guard,
  a cap in GPU-minutes, provenance and the runtime-release cell. Cost estimates are at run-time rates.
- **PRs.** Open one PR per build:
  - `python scripts/review/codex_pr.py open --head claude/wip-<topic> --title "..." --body-file <f>`.
  - The body gives the registration name, the files, the tests' summary line and the mutation list.
  - Then request review: `python scripts/review/codex_pr.py request <PR> --focus "<what changed, what to check>"`.
  - On Windows set `PYTHONUTF8=1` for `codex_pr.py`.
  - Poll `python scripts/review/codex_pr.py status <PR>` every 60 s until it no longer says "no Codex review on head".
    `wait` returns early on the bot's "running" comment.
- **Triage.** Put the review's P0 / P1 comments in `docs/reviews/triage_pr<N>.md`, in the table format of
  `docs/reviews/triage_2026-10-23.md`: reproduce first, then a regression test, then the fix. Allow at most two review
  rounds per PR; then stop.

## Agents to spawn
Each agent gets its own worktree and this brief's rules plus its section below. Give no agent another agent's
reasoning.
1. **Builder W2M-GX**: branch `claude/wip-w2mgx` from `origin/claude/ncm-research`. Start at once.
2. **Builder PD**: branch `claude/wip-pd` from `origin/claude/ncm-research`. Start at once, in parallel with 1.
3. **Builder E0**: branch `claude/wip-e0`, **stacked on `claude/wip-pd`**, since both touch the corruption path. Start
   when PD's code is committed. Its PR targets `claude/ncm-research`, and its body says it is stacked on PD's PR.
4. **Auditor**: a fresh agent with **none of the builders' context**. Start it after all three PRs have passed their
   review rounds.
   - Make a scratch branch `claude/wip-audit15-base` that merges `claude/wip-w2mgx` into `claude/wip-e0`. Push it for
     the record; it is never merged.
   - Give the auditor only `AGENTS.md` and `docs/prompts/codex_audit15_builds.md`. It follows "Deep audit protocol"
     and commits only `docs/reviews/codex_audit15.md` and `tests/test_audit15_*.py`, on branch `codex/audit15`, which
     it pushes.
   - If the auditor cannot run in a separate session, stop and say so; do not audit in the builders' context.

## Builder W2M-GX
- **Code to extend:**
  - `scripts/w2mg_eval.py`: `cell_specs`, `protocol_block`, the cached Gumbel sampler, `--arms`.
  - `src/ncmr/w2mg.py`: shares, jackknife, readings.
  - `scripts/analyze_w2mgr.py`: its admission checks after A13-1. Copy that strictness.
- **Cells.** Give the sampler a per-level model assignment: one checkpoint for each of levels 1-11. Crossed cells at
  any cut, single-level swaps and cross-seed cells then all use one mechanism.
- **Equivalence test.** Show on tiny CPU models that the per-level path reproduces the existing two-model cut cells
  bit for bit, with the same noise.
- **Checkpoints.** Load M2_s and M3_s for all three seeds in one run, record each checkpoint's SHA-256, and keep the
  noise identical to W2M-GR's for the same replay seed.
- **Pure cells.** Prefer re-sampling them (the registration's 3-SE sanity rule) over reusing W2M-GR's.
- **Analyzer.** New `scripts/analyze_w2mgx.py` writes `results/w2mgx_analysis.json`. Test it on:
  - synthetic cells with known kappa, pi_6(s, t) and single-level effects;
  - every reading boundary;
  - admission refusing a wrong GPU, replay seeds, groups, cut or checkpoint mapping.
- **Notebook** `w2mgx_eval`:
  - L4 whole-word guard, as `w2mgr_eval`;
  - reads W2C's seed 1-3 `final.pt` from the Drive `runs/` folder;
  - one run (cap 3.5 L4-hours) or one per seed (cap 75 L4-minutes each). Explain the choice.
  - Make its collector also copy the per-window JS / rho arrays. Audit 14 found the current collector drops them.

## Builder PD
- **Code:**
  - `src/ncmr/w2m.py`: M2 / M3 replacement, mask stream, clean rows, `realized`.
  - `scripts/train_gen.py`: the `mech` path, checkpoint / resume, final evaluation.
  - Configs `configs/scale/L_ts_m2_frozen_pw.yaml` and `L_ts_m3_uniform_wrong.yaml`.
- **New arm `pd`.** Take M2's proposals, then apply a seeded within-batch derangement across rows at levels 2-5, per
  level and position, over masked slots.
  - A slot whose canonical code equals the incoming code keeps its own proposal.
  - The derangement RNG is a new stream: in the atomic checkpoint, restored on resume, consumed by nothing else.
  - Log coverage per level in `metrics.jsonl` and `result.json`.
- **Configs.** `configs/scale/L_ts_pd_p.yaml` (M2), `L_ts_pd_d.yaml` (pd), `L_ts_pd_u.yaml` (M3). All 175 steps, seed
  0, W2M's protocol otherwise.
- **Tests:**
  - P and D share batches, masks, clean rows and RNG states.
  - D's levels 0, 1 and 6-11 equal P's.
  - At levels 2-5, D's replacement multiset per (batch, level, position) equals P's.
  - No slot gets its canonical code.
  - Coverage is computed as registered.
  - Resume equals uninterrupted.
- **Coverage script.** `scripts/pd_coverage.py`: the first 20 steps on real data and masks, on CPU. The registration
  needs >= 0.80 before launch. Say if the released assets were absent.
- **Analyzer.** `scripts/analyze_pd.py` writes `results/pd_analysis.json`: L_X, r under both scorers, paired SEs, and
  the valid / rescue / kill / inconclusive readings, with boundary tests.
- **Notebooks** `pd_L_ts_p`, `pd_L_ts_d`, `pd_L_ts_u`: A100 guard with `--micro_batch 64`, as the W2C notebooks. Caps:
  P 25, D 25, U 15 A100-minutes.

## Builder E0
- **Code:** the generator's input path, where each level's code embedding is upsampled / convolved into the next
  level's input (generator model in `src/ncmr/`), and the corruption path in `scripts/train_gen.py`.
- **New corruption option.**
  - At levels 2-5, a masked slot's code-embedding contribution is replaced by a fixed zero vector before the existing
    upsampling / convolution.
  - Elsewhere it is U0's uniform recipe.
  - No new parameters, flags or inference change; targets unchanged.
- **Configs.** `configs/scale/L_ts_e0_u0.yaml` (M0's recipe, `clean_row_frac: 0.0`, 400 steps) and `L_ts_e0_e0.yaml`.
- **Tests:**
  - U0 and E0 share batches, masks and RNG states.
  - E0's input at masked L2-L5 slots equals a direct computation with a zero contribution there.
  - Everything else is identical to U0's.
  - Eval-time forwards are identical for identical weights.
  - Resume equals uninterrupted.
- **Analyzer.** `scripts/analyze_e0.py` writes `results/e0_analysis.json`: the gate / kill / inconclusive readings,
  with boundary tests.
- **Notebooks** `e0_L_ts_u0`, `e0_L_ts_e0`: A100 guard, 30 A100-minutes each (55 total, as registered; state the
  split).

## Report (lead agent, at the end)
Per PR:
- number, head SHA, the full suite's summary line, the mutation list;
- review rounds and triage state;
- anything open or asked.

For the audit: the `codex/audit15` commit and its summary table. Write the report to `docs/reviews/build_report_15.md`
on `claude/wip-audit15-base` and push it.
