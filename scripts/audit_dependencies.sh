#!/usr/bin/env sh
set -eu

if ! command -v pip-audit >/dev/null 2>&1; then
  echo "pip-audit is required. Install it with: pip install pip-audit" >&2
  exit 2
fi

find agents infra orchestrator -name 'requirements*.txt' -print0 \
  | sort -z \
  | xargs -0 -n1 pip-audit -r
