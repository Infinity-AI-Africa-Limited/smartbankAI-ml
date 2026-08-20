"""Assert every agent actually loaded its model, not merely that it answers.

Agents fall back to a stub when no artefact is present and still return HTTP
200 from /health, so a green health check proves the process is up and nothing
about whether it can score anything. The advisory paths would then return
fallbacks that read like results. This checks the model_loaded flag the agents
already publish.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# The orchestrator is deliberately absent: it routes and holds no model.
AGENT_PORTS = {
    "fraud-detection": 8002,
    "credit-risk": 8003,
    "aml-compliance": 8004,
    "personalization": 8005,
    "predictive-analytics": 8006,
    "conversational-ai": 8007,
    "smart-dashboard": 8008,
    "data-aggregation": 8009,
}


def fetch_health(url: str, timeout: float) -> dict:
    # Refuse anything but HTTP(S) at the point of use rather than suppressing the
    # audit: a file: or custom scheme here would read the runner's filesystem.
    if not url.startswith(("http://", "https://")):
        raise RuntimeError(f"refusing non-HTTP scheme in {url!r}")
    request = urllib.request.Request(url, headers={"X-Client-ID": "ci-model-gate"})  # noqa: S310 - scheme checked above
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - scheme checked above
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost", help="Scheme and host, without a port")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if not args.base_url.startswith(("http://", "https://")):
        print("--base-url must include a scheme", file=sys.stderr)
        return 2

    failures: list[str] = []
    for agent, port in sorted(AGENT_PORTS.items()):
        url = f"{args.base_url}:{port}/health"
        try:
            payload = fetch_health(url, args.timeout)
        except (urllib.error.URLError, RuntimeError, TimeoutError, json.JSONDecodeError) as error:
            failures.append(f"{agent}: health unreachable or unreadable at {url} ({error})")
            continue

        loaded = payload.get("model_loaded")
        if loaded is True:
            print(f"  loaded      {agent}")
        elif loaded is False:
            print(f"  STUB MODE   {agent}")
            failures.append(
                f"{agent}: reports model_loaded=false, so it is serving stub responses. "
                "Generate artefacts before promoting this stack."
            )
        else:
            failures.append(f"{agent}: /health omits model_loaded (got {loaded!r})")

    if failures:
        print("\nModel loading gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\nAll {len(AGENT_PORTS)} agents report a loaded model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
