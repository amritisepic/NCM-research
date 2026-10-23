#!/usr/bin/env bash
# Claude Code PreToolUse hook (matcher: Bash). Exit 2 blocks the call and shows stderr to Claude.
# 1. Blocks `git push` that would put gated-path changes on the run branch (Colab clones it) without a PR.
# 2. Blocks PR merges that bypass scripts/review/codex_pr.py merge (which enforces the Codex gate).
set -uo pipefail
cmd="$(jq -r '.tool_input.command // ""')"
RUN_BRANCH="claude/ncm-research"
GATED_RE='^(src|scripts|configs|notebooks|assets)/'

block() { echo "BLOCKED by review gate: $*" >&2; echo "See CLAUDE.md, 'Codex review'." >&2; exit 2; }

# --- merges must go through the gated merge command ---------------------------------
if [[ "$cmd" =~ gh[[:space:]]+pr[[:space:]]+merge ]] || \
   { [[ "$cmd" =~ gh[[:space:]]+api ]] && [[ "$cmd" =~ /pulls/[0-9]+/merge ]]; }; then
  block "merge PRs with: python scripts/review/codex_pr.py merge <PR>"
fi

# --- pushes to the run branch ----------------------------------------------------
if [[ "$cmd" =~ git[[:space:]]+push ]]; then
  cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
  cur="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo)"
  targets_run=0
  [[ "$cmd" == *"$RUN_BRANCH"* ]] && targets_run=1
  # bare `git push` / `git push origin` / `git push origin HEAD` push the current branch
  if [[ "$cur" == "$RUN_BRANCH" ]]; then
    rest="${cmd#*git push}"; rest="${rest%%[;&|]*}"
    read -r -a toks <<< "$rest"
    args=(); for t in "${toks[@]}"; do [[ "$t" == -* ]] || args+=("$t"); done
    if (( ${#args[@]} < 2 )) || [[ "${args[1]}" == HEAD ]]; then targets_run=1; fi
  fi
  if (( targets_run )); then
    git fetch -q origin "$RUN_BRANCH" 2>/dev/null || true
    if ! all="$(git diff --name-only "origin/$RUN_BRANCH...HEAD" 2>/dev/null)"; then
      block "cannot compare HEAD with origin/$RUN_BRANCH (fetch failed or no merge base); refusing to push to it."
    fi
    changed="$(printf '%s\n' "$all" | grep -E "$GATED_RE" | grep -v '^scripts/review/' || true)"
    if [[ -n "$changed" ]]; then
      block "this push would put unreviewed code on $RUN_BRANCH:
$(printf '%s\n' "$changed" | head -n 15)
Push to a claude/wip-<topic> branch and open a PR: python scripts/review/codex_pr.py open --head <branch> --title ..."
    fi
  fi
fi
exit 0
