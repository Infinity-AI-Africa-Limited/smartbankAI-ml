# Dependency Hardening Sources

The dependency-hardening branch used the following upstream references to select compatible upgrades. The final source of truth remains the package lock or requirement resolution executed in CI.

- [FastAPI version guidance](https://fastapi.tiangolo.com/deployment/versions/) recommends pinning a tested FastAPI release and allowing FastAPI to select its compatible Starlette range.
- [FastAPI release notes](https://fastapi.tiangolo.com/release-notes/) list `0.141.1` as the current tested release used by this branch.
- [prometheus-fastapi-instrumentator releases](https://github.com/trallnag/prometheus-fastapi-instrumentator/releases) documents the Starlette v1-compatible release line used to update instrumentation.

The audit command is `pip-audit -r <requirements-file> --format json`. It was run against every agent, base, and orchestrator requirements manifest before and after remediation.
