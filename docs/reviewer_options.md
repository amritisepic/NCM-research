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
