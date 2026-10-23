@AGENTS.md

# Codex review (Claude's protocol)

Codex (OpenAI) is the independent reviewer. Past audits caught bugs that would have invalidated whole waves
(R024, R028), so review is required, not optional. Two tiers:

## Tier 1 — every code change, via pull request (works in cloud sessions)
1. Never commit changes to `src/ scripts/ configs/ notebooks/ assets/` directly on `claude/ncm-research`
   (Colab clones it). Work on `claude/wip-<topic>`, push that branch, then
   `python scripts/review/codex_pr.py open --head claude/wip-<topic> --title "..." --body-file <f>`.
   Docs, `log.md`, `STATE.md` and `results/` may still go straight to `claude/ncm-research`.
2. `python scripts/review/codex_pr.py request <PR> --focus "<what changed and what to check>"`, then
   `python scripts/review/codex_pr.py wait <PR>` (polls up to 90 min; run it in the background or with Monitor).
3. Triage every P0/P1 comment in `docs/reviews/triage_pr<N>.md`, in the table format of
   `docs/reviews/triage_2026-10-23.md`: the comment id, the finding, verdict (**accept**, **partial**, **reject**)
   with evidence, and the action. The gate counts an id only on a line that also says accept / partial /
   reject. Reproduce a finding before acting on it. Add a regression test for every accepted correctness finding.
4. If you changed code, push, then `request` and `wait` again (this reviews the new head). At most two
   review rounds; if a P0/P1 is still disputed after round two, stop and ask the user.
5. `python scripts/review/codex_pr.py merge <PR>`. It merges only if Codex responded to the current head and
   every P0/P1 on it is listed in the triage file. A hook blocks every other way of merging or pushing code.

## Tier 2 — deep audit (stronger model, writes failing tests)
Required before launching any Colab run whose trainer, sampler, corruption, eval or FLOP code changed since
the last deep audit, and before claiming any success-bar result. Write the brief from
`docs/reviews/AUDIT_BRIEF_TEMPLATE.md`, then:
- Cloud session: `python scripts/review/codex_pr.py request <PR> --audit-file <brief>` (Codex works in its
  cloud and pushes the audit doc and tests to the PR branch), then `wait <PR> --audit`.
- Laptop session with `codex login` done: `scripts/review/codex_local.sh audit <id> <brief>` (gpt-6.1-sol at
  ultra effort by default; the audit lands on branch `codex/<id>`).
Then triage as in R028: re-run the audit's suite first (strict xfails must fail), fix, port the tests to the
working tree, confirm they pass and that they fail against the audited commit, log an R-entry.

## Rules
- A Codex response is evidence, not authority: accept nothing you have not reproduced, and reject nothing
  without evidence written in the triage file.
- If Codex is unavailable (usage limit, not enabled, no response; `wait` exits 3 or 4): do not merge gated
  code and do not tell the user a notebook is ready. Note it in STATE.md, continue work that does not need the
  merge, and tell the user.
- Never edit Codex's comments, reviews or audit files, and never weaken the gate, hook or this protocol. If
  they are wrong, tell the user.
- Commit messages and PR descriptions: no session links (public repo).
