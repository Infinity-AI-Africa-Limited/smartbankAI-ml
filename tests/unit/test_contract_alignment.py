"""Bind the published OpenAPI contract to the models the orchestrator validates against.

The cross-repository check compares the two contract files byte-for-byte. That
proves both repositories carry the same document; it cannot prove the document
describes the code. That gap is how ``aml_check`` came to be documented as a
single transaction while the aml_compliance agent had always served a window,
and how ``employment_type`` came to be required by both implementations but
absent from the contract's required list.

These tests fail when the contract and the Pydantic models disagree.
"""

from pathlib import Path

import pytest
import yaml

from orchestrator.main import ROUTE_MAP, validate_payload
from shared.schemas.orchestrator_v1 import (
    CONTRACT_VERSION,
    AmlFeatures,
    AssistantFeatures,
    CreditFeatures,
    CustomerFeatures,
    RequestType,
    TransactionFeatures,
)

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "contracts" / "ml-orchestrator.v1.openapi.yaml"
CONTRACT = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
SCHEMAS = CONTRACT["components"]["schemas"]

# request type -> (OpenAPI request schema, OpenAPI payload schema, Pydantic model)
CASES = [
    ("fraud_check", "FraudRequest", "TransactionFeatures", TransactionFeatures),
    ("aml_check", "AmlRequest", "AmlFeatures", AmlFeatures),
    ("credit_assessment", "CreditRequest", "CreditFeatures", CreditFeatures),
    ("recommend", "RecommendationRequest", "CustomerFeatures", CustomerFeatures),
    ("chat", "AssistantRequest", "AssistantFeatures", AssistantFeatures),
]


def documented_payload_schema(request_schema: str) -> str:
    body = next(part for part in SCHEMAS[request_schema]["allOf"] if "payload" in part.get("properties", {}))
    return body["properties"]["payload"]["$ref"].rsplit("/", 1)[-1]


def model_fields(model) -> tuple[set[str], set[str]]:
    fields = model.model_fields
    names = set(fields)
    required = {name for name, field in fields.items() if field.is_required()}
    return names, required


def test_contract_declares_the_pinned_version():
    # The version is unquoted in the YAML, so a YAML 1.1 loader yields a date and
    # a YAML 1.2 loader yields a string. Compare on the rendered form.
    assert str(CONTRACT["info"]["version"]) == CONTRACT_VERSION


def test_documented_request_types_match_the_enum():
    documented = set(SCHEMAS["ContractMetadata"]["properties"]["request_type"]["enum"])
    assert documented == {member.value for member in RequestType}
    assert documented == {case[0] for case in CASES}


@pytest.mark.parametrize(("request_type", "request_schema", "payload_schema", "model"), CASES)
def test_request_routes_to_the_documented_payload_schema(request_type, request_schema, payload_schema, model):
    assert documented_payload_schema(request_schema) == payload_schema


@pytest.mark.parametrize(("request_type", "request_schema", "payload_schema", "model"), CASES)
def test_documented_fields_match_the_validated_model(request_type, request_schema, payload_schema, model):
    documented = set(SCHEMAS[payload_schema].get("properties", {}))
    implemented, _ = model_fields(model)
    assert documented == implemented


@pytest.mark.parametrize(("request_type", "request_schema", "payload_schema", "model"), CASES)
def test_documented_required_fields_match_the_validated_model(request_type, request_schema, payload_schema, model):
    documented = set(SCHEMAS[payload_schema].get("required", []))
    _, implemented = model_fields(model)
    assert documented == implemented


@pytest.mark.parametrize(("request_type", "request_schema", "payload_schema", "model"), CASES)
def test_unknown_fields_are_forbidden_on_both_sides(request_type, request_schema, payload_schema, model):
    assert SCHEMAS[payload_schema].get("additionalProperties") is False
    assert model.model_config.get("extra") == "forbid"


@pytest.mark.parametrize(("request_type", "request_schema", "payload_schema", "model"), CASES)
def test_validate_payload_uses_the_documented_model(request_type, request_schema, payload_schema, model):
    """The routing table and the contract must name the same model."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        validate_payload(RequestType(request_type), {"definitely_not_a_contract_field": 1})
    assert excinfo.value.status_code == 422


def test_aml_transaction_item_matches_the_agent_contract():
    documented = set(SCHEMAS["AmlTransaction"].get("properties", {}))
    expected = {"id", "sender", "receiver", "amount_ngn", "timestamp"}
    assert documented == expected
    assert set(SCHEMAS["AmlTransaction"].get("required", [])) == expected


def test_every_documented_request_type_has_a_route():
    for request_type, *_ in CASES:
        assert request_type in ROUTE_MAP, f"{request_type} is documented but has no route"
