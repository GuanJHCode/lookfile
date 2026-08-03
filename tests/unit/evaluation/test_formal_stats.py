"""Formal statistics use the full denominator and paired input resampling."""

from __future__ import annotations

import json

from specstyle.evaluation.stats import finalize_evaluation

from tests.unit.evaluation._formal_fixtures import (
    blind_labels,
    label_approval_receipt,
    machine_ledger,
    protocol_document,
    sealed_protocol,
)


def _documents(*, evidence_class: str = "FORMAL", omit_last: bool = False):
    protocol = protocol_document(evidence_class=evidence_class)
    sealed = sealed_protocol(protocol)
    ledger = machine_ledger(protocol, sealed)
    labels = blind_labels(
        protocol,
        sealed,
        ledger,
        label_source=(
            "EXTERNAL_HUMAN" if evidence_class == "FORMAL" else "SYNTHETIC_TEST"
        ),
        omit_last=omit_last,
    )
    approval = label_approval_receipt(
        protocol,
        sealed,
        ledger,
        labels,
        label_source=(
            "EXTERNAL_HUMAN" if evidence_class == "FORMAL" else "SYNTHETIC_TEST"
        ),
    )
    return protocol, sealed, ledger, labels, approval


def test_primary_metric_intersects_blind_usable_with_machine_approved() -> None:
    report = json.loads(finalize_evaluation(*_documents()))
    arms = {item["arm"]: item for item in report["arms"]}

    assert report["status"] == "FORMAL"
    assert arms["A_single_pass"]["human_usable_yield"] == 0.0
    assert arms["B_random_retry"]["human_usable_yield"] == 0.5
    assert arms["B_random_retry"]["false_reject_count"] == 1
    assert arms["E_full_specstyle"]["human_usable_yield"] == 0.5
    assert arms["E_full_specstyle"]["false_accept_count"] == 1
    assert arms["E_full_specstyle"]["automation_coverage"] == 1.0


def test_formal_comparisons_are_paired_reproducible_and_exclude_a() -> None:
    documents = _documents()

    first = json.loads(finalize_evaluation(*documents))
    second = json.loads(finalize_evaluation(*documents))

    assert first == second
    assert [item["comparator"] for item in first["comparisons"]] == [
        "B_random_retry",
        "C_verifier_best_of_k",
        "D_directed_no_guardrail",
    ]
    assert first["comparisons"][0]["primary_effect"] == 0.0
    assert first["comparisons"][2]["primary_effect"] == -0.5
    assert first["repair_lift_conclusion"] == "NOT_DEMONSTRATED"


def test_missing_human_label_returns_incomplete_without_statistics() -> None:
    report = json.loads(finalize_evaluation(*_documents(omit_last=True)))

    assert report["status"] == "INCOMPLETE"
    assert report["arms"] == []
    assert report["comparisons"] == []
    assert report["missing_blind_artifact_ids"] == ["blind-1-4"]
    assert report["repair_lift_conclusion"] == "NOT_EVALUATED"


def test_synthetic_labels_can_never_produce_formal_conclusion() -> None:
    report = json.loads(finalize_evaluation(*_documents(evidence_class="TEST_ONLY")))

    assert report["status"] == "TEST_ONLY"
    assert report["evidence_class"] == "TEST_ONLY"
    assert report["repair_lift_conclusion"] == "NOT_FORMAL"
