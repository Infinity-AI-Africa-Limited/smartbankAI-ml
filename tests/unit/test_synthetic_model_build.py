"""Regression checks for the reproducible SmartBank synthetic model build."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "synthetic"


def test_synthetic_manifest_declares_development_only_provenance():
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic_only"] is True
    assert manifest["marker"] == "SYNTHETIC_DEVELOPMENT_DATA_ONLY"
    assert len(manifest["datasets"]) >= 10


def test_each_agent_has_a_generated_model_card_and_evaluation_or_knowledge_report():
    expected = {
        "fraud_detection": ("evaluation_report.json",),
        "credit_risk": ("evaluation_report.json",),
        "aml_compliance": ("evaluation_report.json",),
        "personalization": ("evaluation_report.json",),
        "predictive_analytics": ("evaluation_report.json",),
        "conversational_ai": ("retrieval_evaluation_report.json",),
        "smart_dashboard": ("evaluation_report.json",),
        "data_aggregation": ("normalisation_evaluation_report.json",),
    }
    for agent, reports in expected.items():
        model_dir = ROOT / "agents" / agent / "models"
        assert (model_dir / "model_card.md").exists(), agent
        assert all((model_dir / report).exists() for report in reports), agent
        assert "Development-only synthetic model" in (model_dir / "model_card.md").read_text(encoding="utf-8"), agent


def test_synthetic_quality_report_records_two_class_risk_and_credit_targets():
    report = json.loads((DATA_ROOT / "data_quality_report.json").read_text(encoding="utf-8"))
    assert 0 < report["fraud_rate"] < 0.20
    assert 0 < report["loan_repayment_rate"] < 1
    assert report["normalised_records"] if "normalised_records" in report else True
