#!/usr/bin/env sh
# Apply branch protection to main.
#
#   ./scripts/apply-branch-protection.sh [owner/repo] [branch]
#
# Requires the GitHub CLI, authenticated as someone with admin on the repository:
#   gh auth login
#
# Idempotent: re-running applies the same settings.
#
# The required checks below are the CI jobs that actually run on a pull request.
# This matters more than it looks: a required check whose name does not match a
# real job, or which is skipped on pull_request, leaves every pull request
# waiting forever for a result that will never arrive. "Live Stack Tests" and
# the deploy job are deliberately NOT required here for that reason - they carry
# `if: github.event_name != 'pull_request'`.

set -eu

REPO="${1:-Infinity-AI-Africa-Limited/smartbankAI-ml}"
BRANCH="${2:-main}"

if ! command -v gh >/dev/null 2>&1; then
  echo "The GitHub CLI is required. Install it, then run: gh auth login" >&2
  exit 2
fi

echo "Applying protection to ${REPO}@${BRANCH}"

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${REPO}/branches/${BRANCH}/protection" \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Lint, Audit & Unit Tests",
      "Build Images"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false,
  "required_conversation_resolution": true,
  "block_creations": false
}
JSON

echo "Applied. Verify with: ./scripts/verify-branch-protection.sh ${REPO} ${BRANCH}"
