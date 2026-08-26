import pytest

from app.routing import MODEL_REGISTRY, route_protocol


def test_routes_each_supported_protocol_from_dicom_metadata() -> None:
    assert route_protocol({"body_part": "LSPINE"}).protocol == "spine"
    assert route_protocol({"protocol_name": "LEFT HIP"}).protocol == "hip"
    decision = route_protocol({"body_part": "WHOLE BODY"})
    assert decision.protocol == "total-body"
    assert decision.model.status == "ready"


def test_conflicting_metadata_abstains_instead_of_guessing() -> None:
    decision = route_protocol({"body_part": "SPINE HIP"})
    assert decision.protocol == "unsupported"
    assert decision.source == "ambiguous"
    assert decision.model.status == "unsupported"


def test_manual_override_is_auditable() -> None:
    decision = route_protocol({"body_part": "LSPINE"}, "whole-body")
    assert decision.protocol == "total-body"
    assert decision.source == "manual-override"
    assert decision.confidence == 1.0


def test_invalid_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown protocol override"):
        route_protocol({}, "brain")


def test_registry_does_not_claim_untrained_models_are_ready() -> None:
    assert MODEL_REGISTRY["total-body"].status == "ready"
    assert MODEL_REGISTRY["spine"].status == "planned"
    assert MODEL_REGISTRY["hip"].status == "planned"
