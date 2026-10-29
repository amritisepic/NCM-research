# Second reviewer model: options (2026-10-28)

Goal: an independent second opinion, from a different model family, on Codex-built PRs. Codex currently builds, reviews
and audits. Expected use is a few reviews a week, each a 500-3000-line diff plus its registration, ideally agentic (it
reads files and runs pytest). Status: research only. Nothing is installed or bought.

All facts were read 2026-10-28, mostly from official pages through a summarizing fetch. Benchmarks are mostly
vendor-reported. No independent code-*review* benchmark covers these models.

## Models
| Model | Price per 1M tokens (in / out) | Coding evidence | Data terms | Notes |
|---|---|---|---|---|
| Kimi K3 / K3.1 Pro (Moonshot) | ~$3 / $15 (third-party); K3.1 Pro $2 / $10 on OpenRouter | Terminal-Bench 2.1 88.3 (vendor harness); DeepSWE ~67 vs Fable 5 70 | Training on API data unclear (sources conflict); Singapore entity | 1M context; strongest vendor numbers here |
| DeepSeek V4 Pro | $0.66 / $1.98 off-peak, 2x at peak | SWE-bench Verified 80.6 (self-reported), ~74 (one independent report) | May train on input (opt-out by email); stored in the PRC | Anthropic-compatible endpoint; very cheap |
| GLM-5.3 (Z.ai) | $1.40 / $4.40 | Vendor suites only | Prompts not stored for API use; training not stated; Singapore | The Coding Plan is limited to supported tools; use the API, not the plan |
| MiniMax M3.1-Flash | $0.15 / $0.60 | Independent Terminal-Bench 4.0 25.8%; high measured hallucination rate | Inputs may "improve services" | Weakest fit |
| Gemini 3.1 Pro / 4 Argon (Google API) | $2 / $12; $2 / $10 | Not gathered | Paid tier: no training on inputs | No free tier for these; Gemini 3.8 Flash has one (inputs used for training) |
| Others (Qwen3.8, grok-4.x, Mistral Medium 3.5 / Devstral) | $1-2 in | Not gathered | Mixed / unstated | Not shortlisted |

At these prices a review costs cents to a few dollars, even agentic, so cost does not decide. The repository is public,
so data exposure is low. Registrations and results are public too.

## Harness
**Recommended: OpenCode (MIT) in WSL with an OpenRouter key.**
- One key reaches every model above.
- Billing is at provider list price, plus a 5.5% fee when buying credits.
- Account-level privacy settings route only to providers that neither train on nor retain prompts
  (`data_collection: deny`, ZDR).
- `opencode run -m openrouter/<vendor>/<model>` runs headless.
- A `permission` block in `opencode.json` can deny edits and allow only `pytest*` and `git diff/log/show*` in bash.

Alternatives:
- Claude Code itself pointed at another provider via `ANTHROPIC_BASE_URL` (OpenRouter, DeepSeek, Z.ai). Billed by that
  provider; whether it touches Claude plan usage is not verified.
- Codex CLI with a custom provider.
- The vendors' own CLIs (Kimi Code, ZCode, Qwen Code).

Antigravity CLI (Gemini's free route) is out: its consumer terms do not support wrapping it in another agent.

## Calibrate before trusting
Each candidate gets one task with a known answer. Review PR 18's head before the fix (`a949044`) against the W2M-GR
registration. Deep audit 13 found a real P1 there (A13-1: the analyzer accepted unregistered results). Keep the
candidates that find it without a flood of false positives. Planned candidates are Kimi K3.1 Pro and DeepSeek V4 Pro,
optionally Gemini 3.1 Pro (paid API). Expected cost is under $5 in total.

## User-only steps
1. Create an OpenRouter account.
2. Buy a small credit (~$10).
3. Set the training / retention opt-outs.
4. Add the key in OpenCode with `/connect`. The agent never handles keys.

## Calibration result (2026-10-29)
**Setup.**
- OpenCode 1.19.5 through OpenRouter; account-level settings exclude providers that train on or retain prompts.
- Read-only permissions, verified by a probe: writes, shell (except the test runner), web and out-of-tree reads were
  refused.
- The snapshot is PR 18's pre-fix head `a949044`, with no git history and with the diff against its base `733ea77`.
- The prompt was blind (no hint of the bug), and the four models ran in parallel.
- Ground truth: A13-1 (deep audit 13). The analyzer's admission check accepted results whose protocol block (clean bank,
  sampler, CI), per-cell spec (cut, prefix, suffix, seeds) or group vectors depart from the registration. Codex's PR
  review of the same head missed it.

| Model | Found A13-1 | Other findings | Cost | Finished (launched together) |
|---|---|---|---|---|
| Kimi K3.1 Pro | **Yes** (GR-2: protocol / cell spec; GR-3: group vectors) | GR-4, GR-5: real, still in the merged analyzer; GR-1, GR-6: plausible, unverified. All real gaps were labelled P1, one level high | $2.04 | last (~2x the others) |
| GLM-5.3 | No | R1, R2: the same real gaps as GR-5 / GR-4, correctly as P2 | $0.33 | 2nd |
| Gemini 3.1 Pro | No; called the validation "rigid" | none | $1.47 | 3rd |
| DeepSeek V4 Pro | No; called the validation "exhaustive" | none | $0.27 | 1st |

**Real gaps the reviewers found that audits 13 and 14 did not.** Both are still open in `scripts/analyze_w2mgr.py`:
- The (i) / (iii) intervals are checked only for being finite. `lo > hi`, or a mean outside its interval, is admitted.
- The sanity reference is checked by its source filename, not against the W2C files' values.

R065's numbers are unaffected: audit 14 recomputed them independently.

**Reading.** One known-bug case is a coarse filter, not a ranking.
- Kimi K3.1 Pro is the only model that found the bug. It is the recommended second reviewer.
- GLM-5.3 is a cheap secondary pass: real findings, correct priorities, no false alarms.
- DeepSeek V4 Pro and Gemini 3.1 Pro gave confident false negatives here.

Before routing roles in a general workflow, build a 5-8-case benchmark from this repository's audited, fixed defects
(for example A11-1, A13-1, R024) and score each model on bugs found per dollar and false-alarm rate.

## Cheap-tier benchmark, partial (2026-10-29)
**Setup.**
- Seven known-bug cases from this repository's audits:
  - C0 A13-1;
  - C1 PR 3 resume guard;
  - C2 W2M flag halo;
  - C3 TF32 / bf16 MAUVE;
  - C4 D1 gate tolerance / FLOPs;
  - C5 E1b cap reset;
  - C6 A11-1.
- Each case is a history-free snapshot of the pre-fix commit, with its PR diff. Large PRs are restricted to the relevant
  files, and `docs/reviews/` is excluded.
- Same blind prompt, read-only OpenCode, the four models in parallel per case.
- OpenRouter credit ran out during C2: it reserves credit for in-flight requests. C3-C5 did not run; C2 and C6 are
  partial.

"Found" means the reviewer named the defect's key element, at any priority.

| Model | C0 | C1 | C2 | C6 | Found / completed | Bench cost (completed runs) |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash | no (2 real P2 gaps) | **yes, both parts** (P1 + P2) | no | **yes** (as P2) | 2 / 4 | $0.37 |
| DeepSeek V4.1 Flash | no (1 real P2 gap) | **yes** (P1) | failed | failed | 1 / 2 | $0.21 |
| GLM-5.3 | no (docs nit only) | **yes** (as P2) | failed | failed | 1 / 2 | $0.57 |
| DeepSeek V4 Pro | no (1 real P2 gap) | no | no | failed | 0 / 3 | $1.03 |

The calibration run is not in the table: on C0, Kimi K3.1 Pro found the bug; GLM-5.3 and DeepSeek V4 Pro did not.

**Reading (small n).**
- No cheap model found C0 (A13-1) or C2 (the halo) in any run. Kimi K3.1 Pro remains the only model that found C0.
- GLM-5.3-Flash is the best value so far: two of four found, about $0.09 per review, and repeated real minor findings.
  It tends to rate real defects one level low.
- DeepSeek V4 Pro found none of its three cases.
- GLM-5.3's output varied between runs of the same case: two real gaps in the calibration run, one docs nit here.
- Several models independently found the two open gaps in `scripts/analyze_w2mgr.py` (CI bounds not checked; sanity
  reference not checked against the W2C files).

**To finish:** about $4 more credit for C2-C6. Run the models sequentially, not in parallel, so that in-flight
reservations do not exhaust the balance.
