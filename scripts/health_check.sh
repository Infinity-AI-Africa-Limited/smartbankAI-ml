#!/bin/bash
# SmartBank AI — Service Health Check Script
# Usage: ./scripts/health_check.sh [base_url]

BASE=${1:-"http://localhost"}
PASS=0
FAIL=0

declare -A SERVICES=(
  ["orchestrator"]="8001"
  ["fraud-detection"]="8002"
  ["credit-risk"]="8003"
  ["aml-compliance"]="8004"
  ["personalization"]="8005"
  ["predictive-analytics"]="8006"
  ["conversational-ai"]="8007"
  ["smart-dashboard"]="8008"
  ["data-aggregation"]="8009"
)

echo "──────────────────────────────────────────────"
echo "  SmartBank AI — Health Check"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "──────────────────────────────────────────────"

for SERVICE in "${!SERVICES[@]}"; do
  PORT="${SERVICES[$SERVICE]}"
  URL="${BASE}:${PORT}/health"
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$URL")
  if [ "$HTTP_CODE" = "200" ]; then
    echo "  ✓  $SERVICE (port $PORT)"
    ((PASS++))
  else
    echo "  ✗  $SERVICE (port $PORT) — HTTP $HTTP_CODE"
    ((FAIL++))
  fi
done

echo "──────────────────────────────────────────────"
echo "  Passed: $PASS / $((PASS + FAIL))"
echo "──────────────────────────────────────────────"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
