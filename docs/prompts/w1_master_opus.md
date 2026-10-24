# Master prompt: wave-1 analysis and reorientation (Opus 5.5, max effort)

You are the lead researcher on this repository, picking it up after wave 1 finished. Think hard and long: this
session decides where the remaining ~150 Colab compute units (CU) go. Work in this order, and do not skip the
integrity phase because a result looks interesting.

## 0. Read first, nothing else yet
`CLAUDE.md` (Codex review protocol: binding), `AGENTS.md`, `STATE.md`, `docs/brief.md` (goals, success bar,
honesty rules, what to ask the user), `docs/preregistration.md` sections W1, W1a, W1b (and S1/S1b/S1c, J1/J1a/J1b,
M1 when you reach planning), `log.md` R014, R020, R023, R024, R028, R029. Results: `results/colab/w1_*/`
(`*__result.json` final eval, `*__metrics.jsonl` training log and curves, `*__samples_final.json` samples,
`_meta.json`). The repo is public: commit nothing identifying. Follow this repo's own conventions (`STATE.md`,
append-only `log.md` with R-numbers, `docs/preregistration.md`), not a generic PLAN/STATUS layout.

## 1. What came in (seed 0, A100, my quick read: verify every number yourself)
| arm | steps | Gen-PPL GPT-2 L, 16 x 256 (SD over groups) | MAUVE | clean val NLL | entropy |
|---|---|---|---|---|---|
| A0 released, W1b protocol | 0 | 16.16 (0.45) | 0.827 | 861.8 | 3.970 |
| A1 control (uniform corruption) | 1800 | 16.60 (0.31) | 0.881 | 844.0 | 3.962 |
| A2 re-quantized labels | 1800 | 16.29 (0.37) | 0.915 | 749.3 | 3.963 |
| A6 soft labels | 1800 | 17.46 (0.41) | 0.832 | 888.9 | 3.958 |
| A3 one-step on-policy | 1350 | 41.44 (1.72) | 0.496 | 885.7 | 4.026 |
| A4 on-policy + re-quantized | 1350 | 36.81 (1.02) | 0.673 | 707.4 | 4.037 |
| A5 two-pass on-policy | 1080 | 32.17 (0.71) | 0.543 | 913.9 | 3.990 |

Things I noticed and did not resolve:
- **The on-policy arms collapsed fast.** A3's val NLL was 912.5 at step 300 and its sample Gen-PPL 37.3 at step
  600, flat after that. R014 found the generator's own one-step samples keep 98% of decoded tokens, so training
  on them destroying samples is a puzzle: a bug, or a real property of the hierarchy.
- **As registered, no arm passes the W1b screen.** A2 is 0.31 below A1; the bar is 0.35.
- **Likelihood and sample quality came apart again.** `val_nll` scores canonical codes given the clean hierarchy
  (same code path for every arm). A2 cut it from 844 to 749 and A4 to 707, yet Gen-PPL barely moved or got much
  worse. The control improved NLL and MAUVE but worsened Gen-PPL (+0.44, inside the registered 1.0 sanity
  limit). This is the third time NLL and sample quality have come apart (R008/R016 vs R020).
- **MAUVE moved more than Gen-PPL.** A0 here has MAUVE 0.827 vs 0.888 in w0 (R020), probably a protocol
  difference (sample count, fp32). Understand it before using MAUVE as co-primary.
- **The code version is undocumented.** `_meta.json` records no git SHA. The notebooks pulled the branch head at
  start, and the results carry `eval_precision: fp32...` (an R024 F8 marker). A3's log has two step-300 eval rows
  (912.50 / 913.67): a resume, possibly not exact.
- The four `b2_*` / `sudoku_repair_ceiling_val.json` files the notebooks swept up were byte-identical copies of
  files already in `results/`. They were left out of the commit.

## 2. Integrity before interpretation
- **Codex health check first.** On 2026-10-23 Codex never acknowledged the review request on PR #1. Open your
  first PR early (e.g. the analysis script) and `request` + `wait` on it, so you learn whether Codex is available
  before you depend on it. If it is unavailable, follow `CLAUDE.md`: no gated merges, note it in STATE.md, tell
  me, and keep doing work that does not need a merge.
- **A parallel Codex deep audit may already be running.** The brief is `docs/prompts/codex_audit3_onpolicy.md`
  (I may have pasted it into Codex cloud). Check for its PR or branch (`codex/audit3`, or a PR adding
  `docs/reviews/codex_audit3.md`). If none exists, run it yourself: on a PR, `codex_pr.py request <PR>
  --audit-file docs/prompts/codex_audit3_onpolicy.md`, then `wait <PR> --audit`. Triage its findings the R028
  way: re-run its suite first (strict xfails must fail), reproduce each finding before acting, port the tests,
  log an R-entry. Its numbers file gives you an independent recomputation: compare against yours.
- **Do your own bug hunt, independently of Codex.** Is the on-policy collapse a defect? Start with the identity
  test (if the policy returned the canonical codes, the hierarchy must equal the clean one). Check alignment and
  level 0, mode and autocast state after `policy_codes`, and resume replay. Then a small CPU reproduction: a few
  hundred on-policy vs uniform steps on the released Sudoku model, or the smallest text setup that shows it.
  State plainly what you could not test.
- Establish which commit each run used (run times vs commit times of `scripts/train_gen.py`,
  `src/ncmr/corruption.py`, `src/ncmr/train.py`). If any run predates a fix that matters, it is void.

## 3. Registered analysis (strictly as written)
Apply W1 + W1a + W1b exactly: the 16-group screen against each arm's registered comparators, entropy and MAUVE
conditions, A0/A1 sanity checks, and the step-matched A1 curve points (1080, 1350) for the shorter arms.
Conditional controls A1m / A6s run only if their trigger fired. Write one R-entry with the numbers, the verdicts,
and the registered wording for "no arm passes". Do not let anything from section 4 leak into this entry's verdicts.

## 4. Then think: what is actually significant?
Label everything here **exploratory**. You may run analyses on the saved samples and logs and short CPU
experiments. New GPU runs need a registration in `docs/preregistration.md` before launch.

Work through this deliberately:
1. **List what the data says that we did not expect,** each as one sentence with the number behind it. Include
   the on-policy collapse (if it survives section 2), the NLL/quality split, the MAUVE vs Gen-PPL pattern, and
   anything the samples show. Read 30 random samples from A0, A1, A2 and A3 (state the seed). Measure the
   collapse's character: per-level code statistics, decoded-token survival, repetition, where in the window it
   breaks.
2. **For each surprise, write at least two competing explanations and the cheapest test that separates them.**
   Example for the collapse: (a) a defect; (b) clean canonical targets are inconsistent with plausible-but-wrong
   policy inputs, so the model learns refinements that do not fit its own coarse codes (A4, which re-quantizes
   labels, also collapsed: what does that rule out?); (c) the policy drifts with the model, a feedback loop;
   (d) on-policy arms see a lower corruption fraction (0.18 vs 0.31), so they lose denoising robustness. Do not
   stop at these.
3. **Name 3-5 reorientations,** ranked by significance x probability of a clean result x CU. "Significant" means
   one of the brief's success criteria, especially (c) a replicated mechanistic finding about what the levels
   encode, or (d) adaptive compute. A well-characterised failure can be (c). Candidates to weigh, not adopt:
   - **"NCM's hierarchy is fragile to its own input distribution":** a crisp mechanism if it replicates. Sudoku
     has exact correctness and the S1 on-policy arm is already built, so it is a cheap cross-domain replication.
   - **Code NLL is a bad proxy for sample quality in NCM:** three independent observations so far. What would it
     take to make that rigorous, and does it change which objective to train?
   - **Re-quantized hierarchies buy likelihood and MAUVE but not Gen-PPL:** is that a diversity or coverage
     shift? Check with a temperature sweep and Gen-PPL/MAUVE frontiers, not single points.
   - **Quality is lost in coarse levels 1-5 (86%, R020):** interventions on the plan instead of on corruption.
   - **Criterion (d):** Sudoku already meets it without training (R020). WMT14 (J1) is the open half: should it
     jump the queue?
4. **Challenge the queue.** For each queued notebook (`docs/colab_runs.md`): Sudoku 4 arms, WMT14 2 arms, M1
   seed-0 pair, W1 seeds 1-2, decide keep, change, hold or drop, with one line of reasoning. Hold
   `w2b_L_sdk_onpolicy` at least until section 2 is settled. Note that M1's gate needs W1's control training SD
   (A1 seeds 1-2). Ignore sunk cost.

## 5. External reasoning: write the prompts, use them
- **ChatGPT deep research.** After section 2, write `docs/prompts/deep_research_<date>.md` for me to paste into
  ChatGPT deep research. It must:
  - stand alone: what NCM is in five lines, our numbers as a table, links to the public repo files it can read
    (github.com/amritisepic/NCM-research, branch `claude/ncm-research`: `log.md`, `STATE.md`, `results/colab/`);
  - ask focused questions where literature can change our decision. Examples: when does training on a model's own
    predictions hurt (scheduled sampling, exposure bias in diffusion and masked or residual-quantized generators,
    VAR / RQ-Transformer / VALL-E style hierarchies, Infinity's bitwise self-correction); why likelihood and
    sample quality diverge in hierarchical discrete models; prior mechanistic findings about multi-scale codes;
  - ask it to argue against our current plan and to propose the single most significant experiment that fits
    ~100 A100-CU with released checkpoints, with the evidence for and against;
  - demand primary sources with numbers, clearly separated from its own speculation, and a short "what we would
    have to see to change our minds" section.
  When I paste the result back, save it verbatim under `docs/reviews/` and triage it like the hostile review
  (`docs/reviews/triage_2026-10-23.md` format): accept nothing unverified.
- **Codex.** Code review is mandatory, through the `CLAUDE.md` protocol. Every code change goes on a
  `claude/wip-<topic>` branch through `codex_pr.py open / request / wait`, gets a triage file, and merges with
  `codex_pr.py merge`. Run a Tier-2 deep audit (brief from `docs/reviews/AUDIT_BRIEF_TEMPLATE.md`) before any
  Colab launch whose trainer, sampler, corruption, eval or FLOP code changed, and before any success-bar claim.
  If a question is better answered by Codex in parallel with your own work (a second independent recomputation,
  an adversarial attempt to break your favourite explanation), write that brief and run it.
- Use subagents per the global routing table (implementer for plan-task code, blind reviewer for every finished
  unit, explorer for repo-wide searches). The blind reviewer is in addition to Codex, not instead of it.

## 6. Make the changes
- Fix any defects found (with regression tests that fail on the old code), via PR.
- Make the notebooks record the git SHA and the resume history in `_meta.json` (gated: PR).
- Register any new run before launch, and regenerate the notebooks for it. Update `docs/colab_runs.md` (queue,
  costs, holds), `STATE.md` (all sections, including compute: wave 1 cost about 6.5 CU per arm, so check), and
  `log.md`.
- Ask me before: any single run > 15 CU, changing the success criteria, a result you believe meets the success bar,
  and anything the brief or `CLAUDE.md` reserves for me. I launch Colab notebooks myself.

## 7. Report back to me (this exact shape, plain words, no filler)
1. **Verdict in three sentences:** what wave 1 showed, whether the on-policy collapse is a bug or real, and what
   that changes.
2. **Table:** arm, Gen-PPL ± SE, MAUVE, NLL, screen result, each labelled *registered* or *exploratory*.
3. **What is significant and why,** at most five bullets, each with its number and its main caveat.
4. **Codex:** what each review or audit found, what you accepted or rejected and why, PR numbers. Say so if Codex
   was unavailable.
5. **Changes made:** commits / PRs, one line each.
6. **What I need to do:** notebooks to launch, in order, with CU each, and any decision or approval, as 2-4
   options with your recommendation first.
7. **Prompts ready for me to paste:** paths (deep research, any Codex briefs).
Define any domain term the first time. Never say a result is significant unless it passes the stated rule; say
"suggestive" otherwise.
