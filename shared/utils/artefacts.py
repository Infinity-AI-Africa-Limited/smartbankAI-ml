"""Integrity-verified loading of serialised model artefacts.

Deserialising a pickle executes arbitrary code, so write access to the model
volume is equivalent to remote code execution inside the agent. Every artefact
load therefore goes through :func:`load_verified_artefact`, which checks the
file against a SHA-256 manifest published alongside it.

The manifest is ``artefacts.sha256`` in ``MODEL_DIR``, in the format produced by
``sha256sum``::

    <64-hex-digest>  <filename>

Until a signed model registry is in place this manifest is the integrity
boundary. It is written by ``scripts/build_synthetic_models.sh`` for local
development builds and must be produced by the release pipeline for any
artefact promoted to staging or production.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

from shared.utils.config import get_settings

logger = logging.getLogger(__name__)

MANIFEST_NAME = "artefacts.sha256"
_CHUNK = 1024 * 1024


class ArtefactVerificationError(RuntimeError):
    """Raised when an artefact cannot be verified against the manifest."""


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(manifest_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        expected, name = parts[0], parts[-1]
        entries[Path(name).name] = expected.lower()
    return entries


def verify_artefact(path: Path) -> None:
    """Verify ``path`` against the manifest, or raise :class:`ArtefactVerificationError`.

    An unverified load is permitted only when ``SMARTBANK_ALLOW_UNVERIFIED_ARTEFACTS``
    is explicitly set and the service is not running in production. The escape
    hatch exists so a developer can iterate on a freshly trained artefact; it can
    never be used to ship one.
    """
    settings = get_settings()
    manifest_path = Path(settings.model_dir) / MANIFEST_NAME

    if not manifest_path.exists():
        message = f"No {MANIFEST_NAME} manifest in {settings.model_dir}; cannot verify {path.name}"
        if settings.allow_unverified_artefacts and settings.environment != "production":
            logger.warning("%s — loading anyway because SMARTBANK_ALLOW_UNVERIFIED_ARTEFACTS is set", message)
            return
        raise ArtefactVerificationError(message)

    entries = _read_manifest(manifest_path)
    expected = entries.get(path.name)
    if expected is None:
        raise ArtefactVerificationError(f"{path.name} is not listed in {MANIFEST_NAME}")

    actual = _digest(path)
    if actual != expected:
        raise ArtefactVerificationError(
            f"{path.name} failed integrity verification: expected {expected}, got {actual}"
        )


def load_verified_artefact(path: Path | str) -> Any:
    """Verify and then deserialise a model artefact."""
    path = Path(path)
    verify_artefact(path)
    with path.open("rb") as handle:
        return pickle.load(handle)
