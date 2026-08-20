#!/usr/bin/env python3
"""Verify that branch protection is in place and still matches the workflow.

    python3 scripts/verify_branch_protection.py [owner/repo] [branch]

Requires an authenticated GitHub CLI with admin read on the repository.

Two failure modes matter here, and the second is the quiet one:

  * protection was never applied, or was removed;
  * a CI job was renamed, so the required status check now names a job that no
    longer exists. GitHub waits forever for a check that will never report, or
    - worse, if the check was simply dropped - merges without ever running it.

The expected checks are therefore derived from the workflow itself rather than
hardcoded, so this cannot drift away from reality the way the Compose validator
and the health check did.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def jobs_that_run_on_pull_requests(workflow: dict) -> set[str]:
    """Job display names that will actually report a status on a pull request.

    A job carrying a condition that excludes pull_request, or one gated on
    workflow_dispatch, never reports - so it must never be a required check.
    """
    names: set[str] = set()
    for job_id, job in workflow["jobs"].items():
        condition = str(job.get("if", ""))
        if "pull_request" in condition and "!=" in condition:
            continue
        if "workflow_dispatch" in condition:
            continue
        names.add(job.get("name", job_id))
    return names


def gh_api(path: str) -> dict:
    try:
        result = subprocess.run(
            ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise SystemExit(
            "FAIL: the GitHub CLI is not installed. Install it, then run: gh auth login"
        ) from None
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Not Found" in stderr:
            raise SystemExit("FAIL: no branch protection is configured (or no admin access)")
        raise SystemExit(f"FAIL: gh api call failed: {stderr}")
    return json.loads(result.stdout)


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "Infinity-AI-Africa-Limited/smartbankAI-ml"
    branch = sys.argv[2] if len(sys.argv) > 2 else "main"

    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    expected = jobs_that_run_on_pull_requests(workflow)

    protection = gh_api(f"repos/{repo}/branches/{branch}/protection")

    failures: list[str] = []

    required = set(protection.get("required_status_checks", {}).get("contexts", []))
    if not required:
        failures.append("no required status checks are configured")

    unknown = required - expected
    if unknown:
        failures.append(
            "required checks name jobs that do not report on a pull request: "
            + ", ".join(sorted(unknown))
        )

    missing = expected - required
    if missing:
        failures.append("jobs run on pull requests but are not required: " + ", ".join(sorted(missing)))

    if not protection.get("enforce_admins", {}).get("enabled"):
        failures.append("administrators are exempt from protection")

    reviews = protection.get("required_pull_request_reviews")
    if not reviews:
        failures.append("pull request review is not required")
    elif reviews.get("required_approving_review_count", 0) < 1:
        failures.append("no approving review is required")

    if protection.get("allow_force_pushes", {}).get("enabled"):
        failures.append("force pushes are permitted")
    if protection.get("allow_deletions", {}).get("enabled"):
        failures.append("branch deletion is permitted")

    print(f"Branch protection for {repo}@{branch}")
    print(f"  required checks: {', '.join(sorted(required)) or 'none'}")
    print(f"  jobs reporting on pull requests: {', '.join(sorted(expected))}")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOK: protection is present and matches the workflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
