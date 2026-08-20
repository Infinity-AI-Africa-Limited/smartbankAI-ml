#!/usr/bin/env sh
# SmartBank AI — Service Health Check
#
# Queries the orchestrator, which reports the health of every agent it can
# reach. It deliberately does not probe agent ports directly: agents are private
# by design and are not published to the host, so a direct probe reports failure
# for a stack that is entirely healthy.
#
# Usage: ./scripts/health_check.sh [orchestrator_url]

set -u

BASE="${1:-http://localhost:8001}"

echo "──────────────────────────────────────────────"
echo "  SmartBank AI — Health Check"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Orchestrator: ${BASE}"
echo "──────────────────────────────────────────────"

BODY="$(curl -s --max-time 10 "${BASE}/health")"
if [ -z "${BODY}" ]; then
  echo "  ✗  orchestrator is unreachable at ${BASE}"
  echo "──────────────────────────────────────────────"
  exit 1
fi

STATUS="$(printf '%s' "${BODY}" | grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')"
AGENTS="$(printf '%s' "${BODY}" | sed -n 's/.*"agents"[[:space:]]*:[[:space:]]*{\([^}]*\)}.*/\1/p')"

if [ -z "${STATUS}" ] || [ -z "${AGENTS}" ]; then
  echo "  ✗  orchestrator returned an unrecognised health response"
  echo "──────────────────────────────────────────────"
  exit 1
fi

FAIL=0
TOTAL=1
if [ "${STATUS}" = "ok" ]; then
  echo "  ✓  orchestrator — ${STATUS}"
else
  echo "  ✗  orchestrator — ${STATUS}"
  FAIL=$((FAIL + 1))
fi

# "name":"state","name":"state" → one pair per line, then split on the colon.
# The `|| [ -n "$pair" ]` guard matters: the final element carries no trailing
# newline, so a bare `read` drops the last agent from the listing.
printf '%s' "${AGENTS}" | tr ',' '\n' | while IFS= read -r pair || [ -n "${pair}" ]; do
  [ -n "${pair}" ] || continue
  NAME="$(printf '%s' "${pair}" | sed 's/^[[:space:]]*"\([^"]*\)".*/\1/')"
  STATE="$(printf '%s' "${pair}" | sed 's/.*:[[:space:]]*"\([^"]*\)".*/\1/')"
  if [ "${STATE}" = "ok" ]; then
    echo "  ✓  ${NAME} — ${STATE}"
  else
    echo "  ✗  ${NAME} — ${STATE}"
  fi
done

# The subshell above cannot mutate FAIL, so recount here for the exit status.
UNHEALTHY="$(printf '%s' "${AGENTS}" | tr ',' '\n' | grep -c -v '"ok"' || true)"
AGENT_COUNT="$(printf '%s' "${AGENTS}" | tr ',' '\n' | grep -c '.' || true)"
TOTAL=$((AGENT_COUNT + 1))
FAIL=$((FAIL + UNHEALTHY))

echo "──────────────────────────────────────────────"
echo "  Passed: $((TOTAL - FAIL)) / ${TOTAL}"
echo "──────────────────────────────────────────────"

[ "${FAIL}" -eq 0 ] || exit 1
exit 0
