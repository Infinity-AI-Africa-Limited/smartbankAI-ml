"""The artefact gate only earns its place if it fails when it should."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_gate(monkeypatch, agents_dir: Path):
    """Import the script fresh and point it at a throwaway agents tree."""
    spec = importlib.util.spec_from_file_location(
        "verify_model_artefacts", ROOT / "scripts" / "verify_model_artefacts.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_model_artefacts"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", agents_dir.parent)
    monkeypatch.setattr(module, "AGENTS_DIR", agents_dir)
    return module


CARD = """# fraud_detection — Model Card

## Status

**Development-only synthetic model**. Not approved for production decisioning.

| Field | Value |
|---|---|
| Decision posture | Advisory only; human review required |
"""

APPROVED_CARD = CARD.replace("**Development-only synthetic model**. ", "")


def build_agent(agents_dir: Path, name: str = "fraud_detection", card: str = CARD) -> Path:
    model_dir = agents_dir / name / "models"
    model_dir.mkdir(parents=True)
    artefact = model_dir / "model.pkl"
    artefact.write_bytes(b"synthetic-weights")
    checksum = hashlib.sha256(artefact.read_bytes()).hexdigest()
    (model_dir / "artefacts.sha256").write_text(f"{checksum}  ./model.pkl\n", encoding="utf-8")
    (model_dir / "model_card.md").write_text(card, encoding="utf-8")
    return model_dir


def test_accepts_a_consistent_synthetic_artefact(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    build_agent(agents)
    gate = load_gate(monkeypatch, agents)
    model_dir = agents / "fraud_detection" / "models"
    gate.check_integrity(model_dir)
    assert gate.card_status(model_dir) == "synthetic"


def test_rejects_an_artefact_that_no_longer_matches_its_digest(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    model_dir = build_agent(agents)
    gate = load_gate(monkeypatch, agents)
    # Swapping the file is the pickle-substitution case the loader guards against.
    (model_dir / "model.pkl").write_bytes(b"tampered-weights")
    with pytest.raises(gate.GateFailure, match="does not match the manifest"):
        gate.check_integrity(model_dir)


def test_rejects_an_artefact_missing_from_the_manifest(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    model_dir = build_agent(agents)
    gate = load_gate(monkeypatch, agents)
    # An unlisted artefact would be loaded without verification.
    (model_dir / "extra_model.pkl").write_bytes(b"unlisted")
    with pytest.raises(gate.GateFailure, match="absent from artefacts.sha256"):
        gate.check_integrity(model_dir)


def test_rejects_a_missing_manifest(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    model_dir = build_agent(agents)
    gate = load_gate(monkeypatch, agents)
    (model_dir / "artefacts.sha256").unlink()
    with pytest.raises(gate.GateFailure, match="has no artefacts.sha256"):
        gate.check_integrity(model_dir)


def test_rejects_a_missing_model_card(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    model_dir = build_agent(agents)
    gate = load_gate(monkeypatch, agents)
    (model_dir / "model_card.md").unlink()
    with pytest.raises(gate.GateFailure, match="has no model_card.md"):
        gate.card_status(model_dir)


def test_rejects_a_card_that_drops_the_advisory_posture(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    model_dir = build_agent(agents, card="# card\n\n**Development-only synthetic model**.\n")
    gate = load_gate(monkeypatch, agents)
    with pytest.raises(gate.GateFailure, match="advisory posture"):
        gate.card_status(model_dir)


def test_distinguishes_an_approved_card_from_a_synthetic_one(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    build_agent(agents, card=APPROVED_CARD)
    gate = load_gate(monkeypatch, agents)
    assert gate.card_status(agents / "fraud_detection" / "models") == "approved"


def test_require_approved_refuses_synthetic_artefacts(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    build_agent(agents)
    gate = load_gate(monkeypatch, agents)
    monkeypatch.setattr(sys, "argv", ["verify_model_artefacts.py", "--require-approved"])
    assert gate.main() == 1


def test_require_approved_passes_once_cards_are_approved(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    build_agent(agents, card=APPROVED_CARD)
    gate = load_gate(monkeypatch, agents)
    monkeypatch.setattr(sys, "argv", ["verify_model_artefacts.py", "--require-approved"])
    assert gate.main() == 0


def test_default_run_accepts_synthetic_artefacts(tmp_path, monkeypatch):
    # Development and CI must stay usable; only promotion is gated.
    agents = tmp_path / "agents"
    build_agent(agents)
    gate = load_gate(monkeypatch, agents)
    monkeypatch.setattr(sys, "argv", ["verify_model_artefacts.py"])
    assert gate.main() == 0
