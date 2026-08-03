"""Canonical formal-evaluation documents for unit tests."""

from __future__ import annotations

import hashlib

from specstyle.calibration.evidence_io import canonical_json, evidence_sha256
from specstyle.evaluation.protocol import FORMAL_ARMS


def sha(character: str) -> str:
    return character * 64


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def protocol_document(*, evidence_class: str = "FORMAL") -> bytes:
    return canonical_json(
        {
            "schema_version": "specstyle.evaluation.five_arm_protocol.v1",
            "study_id": "repair-lift-heldout-v1",
            "evidence_class": evidence_class,
            "dataset_manifest_sha256": sha("1"),
            "input_ids": ["input-001", "input-002"],
            "initial_request_sha256s": {
                "input-001": sha("f"),
                "input-002": sha("f"),
            },
            "bindings": {
                "compiler_sha256": sha("2"),
                "final_qa_contract_sha256": sha("3"),
                "model_supply_sha256": sha("4"),
                "preprocessor_sha256": sha("5"),
                "runtime_sha256": sha("6"),
            },
            "arms": list(FORMAL_ARMS),
            "budget": {
                "a_generations": 1,
                "max_generations_b_to_e": 3,
            },
            "seed_schedules": {
                "input-001": [101, 102, 103],
                "input-002": [201, 202, 203],
            },
            "strategies": {
                "b_early_stop_rule_sha256": sha("7"),
                "c_tie_break": "lowest_seed_index",
                "c_utility_sha256": sha("8"),
                "d_early_stop_rule_sha256": sha("9"),
                "e_early_stop_rule_sha256": sha("a"),
            },
            "blind": {
                "adjudication": "majority_boolean",
                "minimum_raters_per_artifact": 1,
                "protocol_sha256": sha("b"),
            },
            "statistics": {
                "bootstrap_resamples": 1000,
                "bootstrap_seed": 44,
                "confidence_level": 0.95,
                "method": "paired_percentile_bootstrap",
                "minimum_effect": 0.05,
                "multiple_comparison": "holm_bonferroni",
                "noninferiority_margin": 0.02,
                "sample_size": 2,
            },
            "missingness_rule": "all_inputs_denominator_labels_required_for_artifacts",
        }
    )


def sealed_protocol(protocol: bytes) -> bytes:
    return canonical_json(
        {
            "schema_version": "specstyle.evaluation.sealed_protocol.v1",
            "production_approval_sha256": sha("d"),
            "protocol_sha256": evidence_sha256(protocol).value,
            "repo_sha": sha("c"),
            "sealed_at": "2026-08-03T10:00:00Z",
        }
    )


def machine_ledger(protocol: bytes, sealed: bytes) -> bytes:
    records = []
    seeds = {"input-001": [101, 102, 103], "input-002": [201, 202, 203]}
    terminals = {
        ("input-001", "A_single_pass"): "REJECTED",
        ("input-002", "A_single_pass"): "FAILED",
        ("input-002", "B_random_retry"): "MANUAL_REVIEW",
        ("input-002", "C_verifier_best_of_k"): "REJECTED",
    }
    for input_index, input_id in enumerate(("input-001", "input-002")):
        for arm_index, arm in enumerate(FORMAL_ARMS):
            count = (
                1
                if arm == "A_single_pass"
                else 3
                if arm == "C_verifier_best_of_k"
                else 2
            )
            terminal = terminals.get((input_id, arm), "APPROVED")
            attempts = []
            for generation_index in range(count):
                if generation_index == 0:
                    seed = seeds[input_id][0]
                    seed_reason = "INITIAL"
                elif arm == "B_random_retry":
                    seed = seeds[input_id][generation_index]
                    seed_reason = "RANDOM_RETRY"
                elif arm == "C_verifier_best_of_k":
                    seed = seeds[input_id][generation_index]
                    seed_reason = "VERIFIER_CANDIDATE"
                else:
                    seed = seeds[input_id][0]
                    seed_reason = "FROZEN_REPAIR"
                artifact = digest(
                    f"artifact:{input_index}:{arm_index}:{generation_index}"
                )
                attempts.append(
                    {
                        "artifact_sha256": artifact,
                        "generation_index": generation_index,
                        "gpu_seconds": 10.0 + generation_index,
                        "model_supply_sha256": sha("4"),
                        "qa_pass": terminal == "APPROVED"
                        and generation_index == count - 1,
                        "qa_result_sha256": digest(
                            f"qa:{input_index}:{arm_index}:{generation_index}"
                        ),
                        "request_sha256": sha("f"),
                        "runtime_sha256": sha("6"),
                        "seed": seed,
                        "seed_reason": seed_reason,
                    }
                )
            has_artifact = terminal != "FAILED"
            records.append(
                {
                    "arm": arm,
                    "attempts": attempts,
                    "blind_artifact_id": (
                        f"blind-{input_index}-{arm_index}" if has_artifact else None
                    ),
                    "candidate_sha256": attempts[-1]["artifact_sha256"]
                    if has_artifact
                    else None,
                    "failure_reasons": []
                    if terminal == "APPROVED"
                    else ["QA_NOT_PASSED"],
                    "final_machine_pass": terminal == "APPROVED",
                    "final_qa_contract_sha256": sha("3"),
                    "generations_used": count,
                    "gpu_seconds": sum(item["gpu_seconds"] for item in attempts),
                    "initial_machine_pass": terminal == "APPROVED" and count == 1,
                    "input_id": input_id,
                    "machine_terminal": terminal,
                    "observed_at": "2026-08-03T10:01:00Z",
                    "strategy_trace_sha256": sha("0"),
                }
            )
    return canonical_json(
        {
            "schema_version": "specstyle.evaluation.machine_ledger.v1",
            "sealed_protocol_sha256": evidence_sha256(sealed).value,
            "profile": "production",
            "records": records,
        }
    )


def blind_labels(
    protocol: bytes,
    sealed: bytes,
    ledger: bytes,
    *,
    label_source: str = "EXTERNAL_HUMAN",
    omit_last: bool = False,
) -> bytes:
    import json

    records = json.loads(ledger)["records"]
    labels = []
    for index, record in enumerate(records):
        blind_id = record["blind_artifact_id"]
        if blind_id is None:
            continue
        labels.append(
            {
                "blind_artifact_id": blind_id,
                "blind_protocol_sha256": sha("b"),
                "clarity_acceptable": True,
                "high_priority_degraded": index == 8,
                "overall_usable": index not in {0, 4},
                "rater_pseudonym": f"rater-{index}",
                "rework_minutes": 0.0 if index not in {0, 4} else 5.0,
                "style_faithful": index != 0,
                "subject_preserved": index != 4,
            }
        )
    if omit_last:
        labels.pop()
    return canonical_json(
        {
            "schema_version": "specstyle.evaluation.blind_labels.v1",
            "blind_protocol_sha256": sha("b"),
            "label_source": label_source,
            "labels": labels,
            "machine_ledger_sha256": evidence_sha256(ledger).value,
            "sealed_protocol_sha256": evidence_sha256(sealed).value,
        }
    )


def label_approval_receipt(
    protocol: bytes,
    sealed: bytes,
    ledger: bytes,
    labels: bytes,
    *,
    label_source: str = "EXTERNAL_HUMAN",
) -> bytes:
    import json

    study_id = json.loads(protocol)["study_id"]
    return canonical_json(
        {
            "schema_version": "specstyle.evaluation.label_approval_receipt.v1",
            "approval_kind": "BLIND_HUMAN_LABELS",
            "approved": True,
            "approver_id": "independent-annotation-lead",
            "blind_labels_sha256": evidence_sha256(labels).value,
            "blind_protocol_sha256": sha("b"),
            "issued_at": "2026-08-03T11:00:00Z",
            "label_source": label_source,
            "machine_ledger_sha256": evidence_sha256(ledger).value,
            "receipt_id": "blind-labels-approved-v1",
            "sealed_protocol_sha256": evidence_sha256(sealed).value,
            "study_id": study_id,
        }
    )
