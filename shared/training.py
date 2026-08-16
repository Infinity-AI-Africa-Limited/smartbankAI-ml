"""Shared utilities for SmartBank AI development-only model training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYNTHETIC_DISCLAIMER = (
    "This artefact was trained only on synthetic development data. It is not approved for production "
    "credit, fraud, AML, regulatory, customer, or operational decisioning. A qualified independent "
    "model-risk review and bank-approved real-data validation are required before any production use."
)


def ensure_synthetic(frame: Any, dataset_name: str) -> None:
    if "synthetic_only" not in frame.columns or not bool(frame["synthetic_only"].all()):
        raise ValueError(f"{dataset_name} must be explicitly marked synthetic_only")


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_model_card(
    output_dir: str | Path,
    agent_name: str,
    model_name: str,
    model_version: str,
    dataset_name: str,
    features: list[str],
    metrics: dict[str, Any],
    limitations: list[str],
    intended_use: str,
) -> Path:
    output = ensure_output_dir(output_dir)
    contents = f"""# {agent_name} — Model Card

## Status

**Development-only synthetic model**. {SYNTHETIC_DISCLAIMER}

| Field | Value |
|---|---|
| Model | {model_name} |
| Version | {model_version} |
| Training dataset | `{dataset_name}` |
| Generated | {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} |
| Decision posture | Advisory only; human review required |

## Intended use

{intended_use}

## Feature fields

{', '.join(f'`{feature}`' for feature in features)}

## Offline synthetic evaluation

```json
{json.dumps(metrics, indent=2, default=str)}
```

## Limitations and required controls

""" + "\n".join(f"- {item}" for item in limitations) + "\n"
    path = output / "model_card.md"
    path.write_text(contents, encoding="utf-8")
    return path
