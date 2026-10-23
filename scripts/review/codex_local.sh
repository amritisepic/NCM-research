#!/usr/bin/env bash
# Run Codex locally (laptop sessions, where `codex login` has been done). Cloud sessions use codex_pr.py.
#
#   scripts/review/codex_local.sh review <id> <base-ref> <paths...>   read-only diff review -> docs/reviews/codex_<id>.json
#   scripts/review/codex_local.sh audit  <id> <brief.md>              deep audit in a worktree on branch codex/<id>
#
# Model and effort (override with env): CODEX_MODEL (default gpt-6.1-sol), CODEX_EFFORT (default ultra).
# Exit codes: 0 done (review: approve), 1 review found P0/P1, 2 usage, 3 Codex unavailable (limit/auth/network).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
MODEL="${CODEX_MODEL:-gpt-6.1-sol}"; EFFORT="${CODEX_EFFORT:-ultra}"
mode="${1:-}"; id="${2:-}"
[[ -n "$mode" && -n "$id" ]] || { sed -n 2,8p "$0"; exit 2; }
command -v codex >/dev/null || { echo "codex CLI not installed" >&2; exit 3; }
codex login status >/dev/null 2>&1 || { echo "codex not logged in (run: codex login)" >&2; exit 3; }
mkdir -p docs/reviews .codex-review
COMMON=(--color never -m "$MODEL" -c "model_reasoning_effort=\"$EFFORT\"")
HYGIENE="This repository is PUBLIC. Write no absolute local paths, usernames, emails, session links or credentials into any file or into your final message; refer to files by repo-relative path."

case "$mode" in
review)
  base="${3:?base ref}"; shift 3; (( $# )) || { echo "give paths to review" >&2; exit 2; }
  git diff --quiet HEAD -- "$@" || { echo "commit first: reviews pin a SHA" >&2; exit 2; }
  out="docs/reviews/codex_$id.json"; [[ -e "$out" ]] && { echo "$out exists (append-only); use a new id" >&2; exit 2; }
  sha="$(git rev-parse HEAD)"
  {
    echo "Review the change below as an independent reviewer. Follow the '## Code Review Rules' in AGENTS.md."
    echo "Report only P0/P1 issues (P0 = results invalid; P1 = blocks a trustworthy registered comparison or reproducible continuation) plus P2 edge cases you are sure of."
    echo "Every finding needs repo-relative file:line, evidence, and a test that would expose it. $HYGIENE"
    echo "Commit $sha, base $base."
    echo '```diff'; git diff -U8 "$base" HEAD -- "$@"; echo '```'
  } > ".codex-review/$id.prompt.md"
  if ! codex exec --sandbox read-only --ephemeral "${COMMON[@]}" \
        --output-schema scripts/review/review_schema.json -o ".codex-review/$id.raw.json" - \
        < ".codex-review/$id.prompt.md" > ".codex-review/$id.log" 2>&1; then
    tail -n 5 ".codex-review/$id.log" >&2; exit 3
  fi
  jq --arg sha "$sha" --arg base "$base" --arg model "$MODEL/$EFFORT" \
     '. + {meta:{reviewed_sha:$sha, base:$base, model:$model}}' ".codex-review/$id.raw.json" > "$out"
  jq -r '.findings[] | "- [\(.priority)/\(.confidence)] \(.file):\(.line) \(.issue)"' "$out"
  jq -e '[.findings[] | select(.priority=="P0" or .priority=="P1")] | length == 0' "$out" >/dev/null && exit 0 || exit 1
  ;;
audit)
  brief="${3:?audit brief file}"; [[ -f "$brief" ]] || { echo "no brief $brief" >&2; exit 2; }
  git diff --quiet HEAD || { echo "commit first: the audit pins HEAD" >&2; exit 2; }
  wt="$ROOT/../ncm-codex-$id"; br="codex/$id"
  git worktree add -q -b "$br" "$wt" HEAD
  {
    echo "Deep audit. Follow '## Deep audit protocol' and '## Code Review Rules' in AGENTS.md."
    echo "Audited commit: $(git rev-parse HEAD). Audit id: $id. $HYGIENE"
    echo; cat "$brief"
  } > ".codex-review/$id.prompt.md"
  if ! codex exec --sandbox workspace-write -C "$wt" "${COMMON[@]}" -o ".codex-review/$id.final.md" - \
        < ".codex-review/$id.prompt.md" > ".codex-review/$id.log" 2>&1; then
    tail -n 5 ".codex-review/$id.log" >&2; echo "worktree left at $wt for inspection" >&2; exit 3
  fi
  ( cd "$wt" && git add -A docs/reviews tests && git commit -qm "Codex audit $id" || true )
  echo "Audit committed on branch $br (worktree $wt). Triage it per CLAUDE.md, then: git worktree remove $wt"
  ;;
*) sed -n 2,8p "$0"; exit 2 ;;
esac
