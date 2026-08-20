"""Gate model artefacts before they can be shipped anywhere.

Two separate questions get asked here, and conflating them is how synthetic
development models reach a bank.

Integrity: does every artefact match the SHA-256 manifest the agent loaders
verify at start-up? A mismatch means the file on disk is not the file that was
trained, which for a pickle is equivalent to arbitrary code execution.

Provenance: does every artefact carry a model card, and does that card say what
the model may be used for? Cards emitted by shared.training declare
development-only synthetic status. `--require-approved` refuses to pass while
any card still carries that status, which is what keeps a synthetic scorecard
out of a production credit decision.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "agents"
MANIFEST_NAME = "artefacts.sha256"
CARD_NAME = "model_card.md"

SYNTHETIC_MARKER = "Development-only synthetic model"
ADVISORY_MARKER = "Advisory only; human review required"


class GateFailure(Exception):
    """Raised with a message an engineer can act on without reading this file."""


def model_dirs() -> list[Path]:
    return sorted(p for p in AGENTS_DIR.glob("*/models") if p.is_dir())


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_manifest(manifest: Path) -> dict[str, str]:
    """Parse `sha256sum` output: '<digest>  <name>', two spaces, name may start './'."""
    entries: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise GateFailure(f"{manifest.relative_to(ROOT)}: cannot parse line {raw!r}")
        checksum, name = parts[0], parts[1].strip().lstrip("*")
        if name.startswith("./"):
            name = name[2:]
        entries[name] = checksum
    return entries


def check_integrity(model_dir: Path) -> None:
    manifest = model_dir / MANIFEST_NAME
    if not manifest.exists():
        raise GateFailure(
            f"{model_dir.relative_to(ROOT)} has no {MANIFEST_NAME}. "
            "Agent loaders refuse to deserialise without it; run scripts/build_synthetic_models.sh."
        )

    expected = parse_manifest(manifest)
    if not expected:
        raise GateFailure(f"{manifest.relative_to(ROOT)} is empty")

    for name, checksum in sorted(expected.items()):
        if name == MANIFEST_NAME:
            continue
        artefact = model_dir / name
        if not artefact.exists():
            raise GateFailure(f"{model_dir.relative_to(ROOT)}/{name} is listed in the manifest but missing")
        actual = digest(artefact)
        if actual != checksum:
            raise GateFailure(
                f"{model_dir.relative_to(ROOT)}/{name} does not match the manifest "
                f"(expected {checksum[:12]}…, found {actual[:12]}…)"
            )

    # An artefact absent from the manifest would be loaded unverified, so the
    # manifest has to be exhaustive rather than merely correct.
    tracked = {name for name in expected if name != MANIFEST_NAME}
    on_disk = {p.name for p in model_dir.iterdir() if p.is_file() and p.name != MANIFEST_NAME}
    unlisted = sorted(on_disk - tracked - {CARD_NAME})
    if unlisted:
        raise GateFailure(
            f"{model_dir.relative_to(ROOT)} holds artefacts absent from {MANIFEST_NAME}: {', '.join(unlisted)}"
        )


def card_status(model_dir: Path) -> str:
    card = model_dir / CARD_NAME
    if not card.exists():
        raise GateFailure(
            f"{model_dir.relative_to(ROOT)} has no {CARD_NAME}. "
            "Every artefact ships with a card stating intended use and limitations."
        )
    contents = card.read_text(encoding="utf-8")
    if ADVISORY_MARKER not in contents:
        raise GateFailure(
            f"{card.relative_to(ROOT)} does not record the advisory posture "
            f"({ADVISORY_MARKER!r}). Model output may never be presented as an autonomous decision."
        )
    return "synthetic" if SYNTHETIC_MARKER in contents else "approved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-approved",
        action="store_true",
        help="Fail if any artefact is still a development-only synthetic model. Use before production.",
    )
    args = parser.parse_args()

    directories = model_dirs()
    if not directories:
        print(
            "No agents/*/models directories found. Build artefacts before verifying them: "
            "scripts/build_synthetic_models.sh",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []
    statuses: dict[str, str] = {}

    for model_dir in directories:
        agent = model_dir.parent.name
        try:
            check_integrity(model_dir)
            statuses[agent] = card_status(model_dir)
        except GateFailure as failure:
            failures.append(str(failure))

    for agent in sorted(statuses):
        print(f"  {statuses[agent]:>9}  {agent}")

    if failures:
        print("\nModel artefact gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    synthetic = sorted(a for a, s in statuses.items() if s == "synthetic")
    if args.require_approved and synthetic:
        print(
            "\nRefusing to promote development-only synthetic models: "
            + ", ".join(synthetic)
            + "\nReplace them with artefacts trained on bank-approved data and signed off by an "
            "independent model-risk review, then regenerate the model cards.",
            file=sys.stderr,
        )
        return 1

    if synthetic:
        print(f"\n{len(synthetic)} synthetic development artefact(s) verified. Not for production decisions.")
    else:
        print(f"\n{len(statuses)} artefact(s) verified as approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
