from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_base_api_framework_uses_patched_versions() -> None:
    requirements = read("infra/docker/requirements.base.txt")
    assert "fastapi==0.141.1" in requirements
    assert "starlette==1.3.1" in requirements
    assert "python-multipart==0.0.31" in requirements
    assert "prometheus-fastapi-instrumentator==8.1.0" in requirements


def test_agent_manifests_exclude_removed_vulnerable_retrieval_stack() -> None:
    requirements = read("agents/conversational_ai/requirements.txt")
    for package in ("langchain==", "langchain-community==", "sentence-transformers=="):
        assert package not in requirements


def test_risk_agents_pin_the_patched_lightgbm_release() -> None:
    for agent in ("credit_risk", "fraud_detection", "personalization", "predictive_analytics"):
        assert "lightgbm==4.6.0" in read(f"agents/{agent}/requirements.txt")


def test_aggregation_agent_pins_the_patched_xml_parser() -> None:
    assert "lxml==6.1.0" in read("agents/data_aggregation/requirements.txt")
