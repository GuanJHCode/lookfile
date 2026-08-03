"""Machine and blinded-human evidence remain strictly separated."""

from __future__ import annotations

import json

import pytest

from specstyle.calibration.evidence_io import canonical_json
from specstyle.errors import DomainError
from specstyle.evaluation.evidence import (
    import_blind_labels,
    validate_machine_ledger,
)

from tests.unit.evaluation._formal_fixtures import (
    blind_labels,
    label_approval_receipt,
    machine_ledger,
    protocol_document,
    sealed_protocol,
)


def _documents():
    protocol = protocol_document()
    sealed = sealed_protocol(protocol)
    ledger = machine_ledger(protocol, sealed)
    return protocol, sealed, ledger


def test_machine_ledger_validates_complete_input_by_arm_matrix() -> None:
    protocol, sealed, ledger = _documents()
    receipt = json.loads(validate_machine_ledger(protocol, sealed, ledger))

    assert receipt["status"] == "AWAITING_BLIND_LABELS"
    assert receipt["input_count"] == 2
    assert receipt["record_count"] == 10
    assert receipt["missing_artifact_count"] == 1


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update({"profile": "preview"}),
        lambda value: value["records"].pop(),
        lambda value: value["records"].append(value["records"][0]),
        lambda value: value["records"][1]["attempts"][0].update({"seed": 999}),
        lambda value: value["records"][1]["attempts"][0].update(
            {"model_supply_sha256": "f" * 64}
        ),
        lambda value: value["records"][1]["attempts"][0].update(
            {"request_sha256": "e" * 64}
        ),
        lambda value: value["records"][1].update(
            {"final_qa_contract_sha256": "f" * 64}
        ),
        lambda value: value["records"][0].update(
            {"observed_at": "2026-08-03T09:59:59Z"}
        ),
    ),
)
def test_machine_ledger_rejects_unfair_incomplete_or_preview_evidence(mutate) -> None:
    protocol, sealed, ledger = _documents()
    value = json.loads(ledger)
    mutate(value)

    with pytest.raises(DomainError, match="machine evaluation ledger"):
        validate_machine_ledger(protocol, sealed, canonical_json(value))


def test_best_of_k_must_consume_full_preregistered_budget() -> None:
    protocol, sealed, ledger = _documents()
    value = json.loads(ledger)
    record = value["records"][2]
    record["attempts"].pop()
    record["generations_used"] = 2
    record["gpu_seconds"] = sum(item["gpu_seconds"] for item in record["attempts"])
    record["candidate_sha256"] = record["attempts"][-1]["artifact_sha256"]
    record["attempts"][-1]["qa_pass"] = True

    with pytest.raises(DomainError, match="machine evaluation ledger"):
        validate_machine_ledger(protocol, sealed, canonical_json(value))


def test_full_repair_cannot_change_seed_without_sampling_defect_reason() -> None:
    protocol, sealed, ledger = _documents()
    value = json.loads(ledger)
    value["records"][4]["attempts"][1]["seed"] = 102

    with pytest.raises(DomainError, match="machine evaluation ledger"):
        validate_machine_ledger(protocol, sealed, canonical_json(value))


def test_blind_label_import_has_no_arm_input_or_machine_fields() -> None:
    protocol, sealed, ledger = _documents()
    labels = blind_labels(protocol, sealed, ledger)
    approval = label_approval_receipt(protocol, sealed, ledger, labels)
    receipt = json.loads(
        import_blind_labels(protocol, sealed, ledger, labels, approval)
    )

    assert receipt["status"] == "BLIND_LABELS_COMPLETE"
    assert receipt["labeled_artifact_count"] == 9

    value = json.loads(labels)
    value["labels"][0]["arm"] = "E_full_specstyle"
    with pytest.raises(DomainError, match="blind evaluation labels"):
        import_blind_labels(protocol, sealed, ledger, canonical_json(value), approval)


def test_missing_blind_label_is_reported_without_mapping_it_to_false() -> None:
    protocol, sealed, ledger = _documents()
    labels = blind_labels(protocol, sealed, ledger, omit_last=True)
    approval = label_approval_receipt(protocol, sealed, ledger, labels)
    receipt = json.loads(
        import_blind_labels(protocol, sealed, ledger, labels, approval)
    )

    assert receipt["status"] == "BLIND_LABELS_INCOMPLETE"
    assert receipt["missing_blind_artifact_ids"] == ["blind-1-4"]


def test_synthetic_labels_are_permanently_test_only() -> None:
    protocol = protocol_document(evidence_class="TEST_ONLY")
    sealed = sealed_protocol(protocol)
    ledger = machine_ledger(protocol, sealed)
    labels = blind_labels(
        protocol,
        sealed,
        ledger,
        label_source="SYNTHETIC_TEST",
    )
    approval = label_approval_receipt(
        protocol,
        sealed,
        ledger,
        labels,
        label_source="SYNTHETIC_TEST",
    )
    receipt = json.loads(
        import_blind_labels(protocol, sealed, ledger, labels, approval)
    )

    assert receipt["evidence_class"] == "TEST_ONLY"
    assert receipt["formal_eligible"] is False


def test_label_approval_receipt_must_cross_bind_external_ledger() -> None:
    protocol, sealed, ledger = _documents()
    labels = blind_labels(protocol, sealed, ledger)
    approval = json.loads(label_approval_receipt(protocol, sealed, ledger, labels))
    approval["blind_labels_sha256"] = "f" * 64

    with pytest.raises(DomainError, match="label approval receipt"):
        import_blind_labels(protocol, sealed, ledger, labels, canonical_json(approval))
