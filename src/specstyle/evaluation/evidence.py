"""Machine and blinded-human evidence contracts for formal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any

from specstyle.calibration.evidence_io import (
    _exact,
    _float,
    _load_canonical,
    _sha,
    _text,
    canonical_json,
    evidence_sha256,
)
from specstyle.errors import DomainError
from specstyle.evaluation.protocol import (
    FORMAL_ARMS,
    load_protocol,
    load_sealed_protocol,
)

_ATTEMPT_KEYS = {
    "artifact_sha256",
    "generation_index",
    "gpu_seconds",
    "model_supply_sha256",
    "qa_pass",
    "qa_result_sha256",
    "request_sha256",
    "runtime_sha256",
    "seed",
    "seed_reason",
}
_RECORD_KEYS = {
    "arm",
    "attempts",
    "blind_artifact_id",
    "candidate_sha256",
    "failure_reasons",
    "final_machine_pass",
    "final_qa_contract_sha256",
    "generations_used",
    "gpu_seconds",
    "initial_machine_pass",
    "input_id",
    "machine_terminal",
    "observed_at",
    "strategy_trace_sha256",
}
_LABEL_KEYS = {
    "blind_artifact_id",
    "blind_protocol_sha256",
    "clarity_acceptable",
    "high_priority_degraded",
    "overall_usable",
    "rater_pseudonym",
    "rework_minutes",
    "style_faithful",
    "subject_preserved",
}
_TERMINALS = {"APPROVED", "REJECTED", "MANUAL_REVIEW", "FAILED"}


@dataclass(frozen=True, slots=True)
class MachineArmRecord:
    input_id: str
    arm: str
    terminal: str
    candidate_sha256: str | None
    blind_artifact_id: str | None
    generations_used: int
    gpu_seconds: float
    initial_machine_pass: bool
    final_machine_pass: bool


@dataclass(frozen=True, slots=True)
class BlindHumanLabel:
    blind_artifact_id: str
    rater_pseudonym: str
    style_faithful: bool
    subject_preserved: bool
    clarity_acceptable: bool
    overall_usable: bool
    high_priority_degraded: bool
    rework_minutes: float


@dataclass(frozen=True, slots=True)
class EvaluationEvidence:
    protocol: dict[str, Any]
    sealed: dict[str, Any]
    records: tuple[MachineArmRecord, ...]
    labels: tuple[BlindHumanLabel, ...] = ()
    label_source: str | None = None
    missing_blind_artifact_ids: tuple[str, ...] = ()
    label_approval_receipt_sha256: str | None = None


def _timestamp(value: object, name: str) -> datetime:
    text = _text(value, name)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise DomainError(f"invalid {name}") from None
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        raise DomainError(f"invalid {name}")
    return parsed


def _nonnegative_float(value: object, name: str) -> float:
    result = _float(value, name)
    if result < 0.0:
        raise DomainError(f"invalid {name}")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or isinstance(value, bool) or value < minimum:
        raise DomainError(f"invalid {name}")
    return value


def _optional_sha(value: object, name: str) -> str | None:
    return None if value is None else _sha(value, name)


def _failure_reasons(value: object) -> tuple[str, ...]:
    if type(value) is not list:
        raise DomainError("invalid machine evaluation ledger")
    reasons = tuple(_text(item, "machine failure reason") for item in value)
    if len(set(reasons)) != len(reasons):
        raise DomainError("invalid machine evaluation ledger")
    return reasons


def _attempts(
    values: object,
    *,
    arm: str,
    seeds: list[int],
    initial_request_sha256: str,
    model_sha256: str,
    runtime_sha256: str,
) -> tuple[dict[str, Any], ...]:
    if type(values) is not list or not values or len(values) > len(seeds):
        raise DomainError("invalid machine evaluation ledger")
    parsed: list[dict[str, Any]] = []
    current_seed = seeds[0]
    next_seed = 1
    for index, value in enumerate(values):
        attempt = _exact(value, _ATTEMPT_KEYS, "machine generation attempt")
        gpu_seconds = _nonnegative_float(attempt["gpu_seconds"], "attempt GPU seconds")
        expected_seed, next_seed = _expected_seed(
            arm, index, attempt["seed_reason"], seeds, current_seed, next_seed
        )
        current_seed = expected_seed
        if (
            _integer(attempt["generation_index"], "generation index") != index
            or _integer(attempt["seed"], "generation seed") != expected_seed
            or _sha(attempt["model_supply_sha256"], "model supply sha256")
            != model_sha256
            or _sha(attempt["runtime_sha256"], "runtime sha256") != runtime_sha256
            or type(attempt["qa_pass"]) is not bool
        ):
            raise DomainError("invalid machine evaluation ledger")
        _sha(attempt["request_sha256"], "generation request sha256")
        if index == 0 and attempt["request_sha256"] != initial_request_sha256:
            raise DomainError("invalid machine evaluation ledger")
        _sha(attempt["qa_result_sha256"], "QA result sha256")
        _optional_sha(attempt["artifact_sha256"], "attempt artifact sha256")
        parsed.append({**attempt, "gpu_seconds": gpu_seconds})
    return tuple(parsed)


def _expected_seed(
    arm: str,
    index: int,
    reason: object,
    seeds: list[int],
    current_seed: int,
    next_seed: int,
) -> tuple[int, int]:
    if index == 0:
        if reason != "INITIAL":
            raise DomainError("invalid machine evaluation ledger")
        return seeds[0], next_seed
    if arm == "B_random_retry":
        if reason != "RANDOM_RETRY":
            raise DomainError("invalid machine evaluation ledger")
        return seeds[index], next_seed
    if arm == "C_verifier_best_of_k":
        if reason != "VERIFIER_CANDIDATE":
            raise DomainError("invalid machine evaluation ledger")
        return seeds[index], next_seed
    if arm in {"D_directed_no_guardrail", "E_full_specstyle"}:
        if reason == "FROZEN_REPAIR":
            return current_seed, next_seed
        if reason == "SAMPLING_DEFECT_RETRY" and next_seed < len(seeds):
            return seeds[next_seed], next_seed + 1
    raise DomainError("invalid machine evaluation ledger")


def _validate_terminal(
    raw: dict[str, Any],
    attempts: tuple[dict[str, Any], ...],
    reasons: tuple[str, ...],
) -> tuple[str | None, str | None]:
    terminal = raw["machine_terminal"]
    candidate = _optional_sha(raw["candidate_sha256"], "candidate sha256")
    blind_id = (
        None
        if raw["blind_artifact_id"] is None
        else _text(raw["blind_artifact_id"], "blind artifact id")
    )
    final_pass = raw["final_machine_pass"]
    if type(final_pass) is not bool or type(raw["initial_machine_pass"]) is not bool:
        raise DomainError("invalid machine evaluation ledger")
    artifact_hashes = {item["artifact_sha256"] for item in attempts}
    approved = terminal == "APPROVED"
    selected = next(
        (item for item in attempts if item["artifact_sha256"] == candidate), None
    )
    if (
        terminal not in _TERMINALS
        or final_pass is not approved
        or (candidate is None) != (blind_id is None)
        or (candidate is not None and candidate not in artifact_hashes)
        or (terminal == "FAILED" and candidate is not None)
        or (terminal != "FAILED" and candidate is None)
        or (candidate is not None and selected is None)
        or (selected is not None and selected["qa_pass"] is not approved)
        or (approved and reasons)
        or (not approved and not reasons)
        or raw["initial_machine_pass"] is not attempts[0]["qa_pass"]
    ):
        raise DomainError("invalid machine evaluation ledger")
    return candidate, blind_id


def _validate_stopping(
    arm: str,
    attempts: tuple[dict[str, Any], ...],
    candidate: str | None,
    maximum: int,
) -> None:
    if arm == FORMAL_ARMS[0] and len(attempts) != 1:
        raise DomainError("invalid machine evaluation ledger")
    if arm == FORMAL_ARMS[2] and len(attempts) != maximum:
        raise DomainError("invalid machine evaluation ledger")
    if arm != FORMAL_ARMS[2]:
        passing = [index for index, item in enumerate(attempts) if item["qa_pass"]]
        if passing and passing[0] != len(attempts) - 1:
            raise DomainError("invalid machine evaluation ledger")
        if candidate is not None and candidate != attempts[-1]["artifact_sha256"]:
            raise DomainError("invalid machine evaluation ledger")


def _machine_record(
    value: object,
    *,
    input_id: str,
    arm: str,
    protocol: dict[str, Any],
    sealed_at: datetime,
) -> MachineArmRecord:
    raw = _exact(value, _RECORD_KEYS, "machine arm record")
    bindings = protocol["bindings"]
    attempts = _attempts(
        raw["attempts"],
        arm=arm,
        seeds=protocol["seed_schedules"][input_id],
        initial_request_sha256=protocol["initial_request_sha256s"][input_id],
        model_sha256=bindings["model_supply_sha256"],
        runtime_sha256=bindings["runtime_sha256"],
    )
    reasons = _failure_reasons(raw["failure_reasons"])
    candidate, blind_id = _validate_terminal(raw, attempts, reasons)
    generations = _integer(raw["generations_used"], "generations used", minimum=1)
    gpu_seconds = _nonnegative_float(raw["gpu_seconds"], "record GPU seconds")
    if (
        raw["input_id"] != input_id
        or raw["arm"] != arm
        or generations != len(attempts)
        or generations > protocol["budget"]["max_generations_b_to_e"]
        or not math.isclose(
            gpu_seconds,
            sum(item["gpu_seconds"] for item in attempts),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or raw["final_qa_contract_sha256"] != bindings["final_qa_contract_sha256"]
        or _timestamp(raw["observed_at"], "machine observation time") < sealed_at
    ):
        raise DomainError("invalid machine evaluation ledger")
    _validate_stopping(
        arm,
        attempts,
        candidate,
        protocol["budget"]["max_generations_b_to_e"],
    )
    _sha(raw["strategy_trace_sha256"], "strategy trace sha256")
    return MachineArmRecord(
        input_id,
        arm,
        raw["machine_terminal"],
        candidate,
        blind_id,
        generations,
        gpu_seconds,
        raw["initial_machine_pass"],
        raw["final_machine_pass"],
    )


def load_machine_evidence(
    protocol_data: bytes, sealed_data: bytes, ledger_data: bytes, /
) -> EvaluationEvidence:
    """Parse a complete Production-only input-by-arm machine matrix."""
    try:
        protocol = load_protocol(protocol_data)
        sealed = load_sealed_protocol(sealed_data)
        if sealed["protocol_sha256"] != evidence_sha256(protocol_data).value:
            raise DomainError("invalid machine evaluation ledger")
        ledger = _exact(
            _load_canonical(ledger_data),
            {"schema_version", "sealed_protocol_sha256", "profile", "records"},
            "machine evaluation ledger",
        )
        expected = [
            (item, arm) for item in protocol["input_ids"] for arm in FORMAL_ARMS
        ]
        if (
            ledger["schema_version"] != "specstyle.evaluation.machine_ledger.v1"
            or ledger["sealed_protocol_sha256"] != evidence_sha256(sealed_data).value
            or ledger["profile"] != "production"
            or type(ledger["records"]) is not list
            or len(ledger["records"]) != len(expected)
        ):
            raise DomainError("invalid machine evaluation ledger")
        sealed_at = _timestamp(sealed["sealed_at"], "protocol seal time")
        records = tuple(
            _machine_record(
                value,
                input_id=input_id,
                arm=arm,
                protocol=protocol,
                sealed_at=sealed_at,
            )
            for value, (input_id, arm) in zip(ledger["records"], expected, strict=True)
        )
        blind_ids = [
            item.blind_artifact_id for item in records if item.blind_artifact_id
        ]
        if len(set(blind_ids)) != len(blind_ids):
            raise DomainError("invalid machine evaluation ledger")
    except (DomainError, KeyError, TypeError):
        raise DomainError("invalid machine evaluation ledger") from None
    return EvaluationEvidence(protocol, sealed, records)


def validate_machine_ledger(
    protocol_data: bytes, sealed_data: bytes, ledger_data: bytes, /
) -> bytes:
    """Return a canonical validation receipt without importing human labels."""
    evidence = load_machine_evidence(protocol_data, sealed_data, ledger_data)
    return canonical_json(
        {
            "schema_version": "specstyle.evaluation.machine_validation.v1",
            "status": "AWAITING_BLIND_LABELS",
            "evidence_class": evidence.protocol["evidence_class"],
            "input_count": len(evidence.protocol["input_ids"]),
            "record_count": len(evidence.records),
            "missing_artifact_count": sum(
                item.candidate_sha256 is None for item in evidence.records
            ),
            "machine_ledger_sha256": evidence_sha256(ledger_data).value,
            "sealed_protocol_sha256": evidence_sha256(sealed_data).value,
        }
    )


def _blind_label(value: object) -> BlindHumanLabel:
    raw = _exact(value, _LABEL_KEYS, "blind evaluation label")
    boolean_keys = _LABEL_KEYS - {
        "blind_artifact_id",
        "blind_protocol_sha256",
        "rater_pseudonym",
        "rework_minutes",
    }
    if any(type(raw[name]) is not bool for name in boolean_keys):
        raise DomainError("invalid blind evaluation labels")
    return BlindHumanLabel(
        _text(raw["blind_artifact_id"], "blind artifact id"),
        _text(raw["rater_pseudonym"], "rater pseudonym"),
        raw["style_faithful"],
        raw["subject_preserved"],
        raw["clarity_acceptable"],
        raw["overall_usable"],
        raw["high_priority_degraded"],
        _nonnegative_float(raw["rework_minutes"], "human rework minutes"),
    )


def _approval_receipt(
    data: bytes,
    *,
    machine: EvaluationEvidence,
    sealed_data: bytes,
    ledger_data: bytes,
    label_data: bytes,
    raters: set[str],
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "approval_kind",
        "approved",
        "approver_id",
        "blind_labels_sha256",
        "blind_protocol_sha256",
        "issued_at",
        "label_source",
        "machine_ledger_sha256",
        "receipt_id",
        "sealed_protocol_sha256",
        "study_id",
    }
    try:
        raw = _exact(_load_canonical(data), keys, "label approval receipt")
        approver = _text(raw["approver_id"], "label approver id")
        if (
            raw["schema_version"] != "specstyle.evaluation.label_approval_receipt.v1"
            or raw["approval_kind"] != "BLIND_HUMAN_LABELS"
            or raw["approved"] is not True
            or raw["label_source"] not in {"EXTERNAL_HUMAN", "SYNTHETIC_TEST"}
            or raw["study_id"] != machine.protocol["study_id"]
            or raw["blind_protocol_sha256"]
            != machine.protocol["blind"]["protocol_sha256"]
            or raw["blind_labels_sha256"] != evidence_sha256(label_data).value
            or raw["machine_ledger_sha256"] != evidence_sha256(ledger_data).value
            or raw["sealed_protocol_sha256"] != evidence_sha256(sealed_data).value
            or _timestamp(raw["issued_at"], "label approval time")
            < _timestamp(machine.sealed["sealed_at"], "protocol seal time")
            or approver in raters
        ):
            raise DomainError("invalid label approval receipt")
        _text(raw["receipt_id"], "label approval receipt id")
    except (DomainError, KeyError, TypeError):
        raise DomainError("invalid label approval receipt") from None
    return raw


def load_blind_evidence(
    protocol_data: bytes,
    sealed_data: bytes,
    ledger_data: bytes,
    label_data: bytes,
    approval_data: bytes,
    /,
) -> EvaluationEvidence:
    """Validate external blind labels without exposing arm or input fields."""
    machine = load_machine_evidence(protocol_data, sealed_data, ledger_data)
    try:
        raw = _exact(
            _load_canonical(label_data),
            {
                "schema_version",
                "blind_protocol_sha256",
                "label_source",
                "labels",
                "machine_ledger_sha256",
                "sealed_protocol_sha256",
            },
            "blind evaluation labels",
        )
        if (
            raw["schema_version"] != "specstyle.evaluation.blind_labels.v1"
            or raw["label_source"] not in {"EXTERNAL_HUMAN", "SYNTHETIC_TEST"}
            or raw["blind_protocol_sha256"]
            != machine.protocol["blind"]["protocol_sha256"]
            or raw["machine_ledger_sha256"] != evidence_sha256(ledger_data).value
            or raw["sealed_protocol_sha256"] != evidence_sha256(sealed_data).value
            or type(raw["labels"]) is not list
        ):
            raise DomainError("invalid blind evaluation labels")
        labels = tuple(_blind_label(item) for item in raw["labels"])
        if any(
            item["blind_protocol_sha256"]
            != machine.protocol["blind"]["protocol_sha256"]
            for item in raw["labels"]
        ):
            raise DomainError("invalid blind evaluation labels")
        pairs = [(item.blind_artifact_id, item.rater_pseudonym) for item in labels]
        if pairs != sorted(pairs) or len(set(pairs)) != len(pairs):
            raise DomainError("invalid blind evaluation labels")
        expected = {
            item.blind_artifact_id
            for item in machine.records
            if item.blind_artifact_id is not None
        }
        observed = {item.blind_artifact_id for item in labels}
        if not observed <= expected:
            raise DomainError("invalid blind evaluation labels")
        minimum = machine.protocol["blind"]["minimum_raters_per_artifact"]
        counts = {
            item: sum(label.blind_artifact_id == item for label in labels)
            for item in observed
        }
        if any(count < minimum or count % 2 == 0 for count in counts.values()):
            raise DomainError("invalid blind evaluation labels")
    except (DomainError, KeyError, TypeError):
        raise DomainError("invalid blind evaluation labels") from None
    approval = _approval_receipt(
        approval_data,
        machine=machine,
        sealed_data=sealed_data,
        ledger_data=ledger_data,
        label_data=label_data,
        raters={item.rater_pseudonym for item in labels},
    )
    if approval["label_source"] != raw["label_source"]:
        raise DomainError("invalid label approval receipt")
    missing = tuple(sorted(expected - observed))
    return EvaluationEvidence(
        machine.protocol,
        machine.sealed,
        machine.records,
        labels,
        raw["label_source"],
        missing,
        evidence_sha256(approval_data).value,
    )


def import_blind_labels(
    protocol_data: bytes,
    sealed_data: bytes,
    ledger_data: bytes,
    label_data: bytes,
    approval_data: bytes,
    /,
) -> bytes:
    """Return a receipt that never promotes synthetic labels to formal evidence."""
    evidence = load_blind_evidence(
        protocol_data,
        sealed_data,
        ledger_data,
        label_data,
        approval_data,
    )
    complete = not evidence.missing_blind_artifact_ids
    formal = (
        complete
        and evidence.protocol["evidence_class"] == "FORMAL"
        and evidence.label_source == "EXTERNAL_HUMAN"
    )
    evidence_class = "FORMAL" if formal else "TEST_ONLY"
    return canonical_json(
        {
            "schema_version": "specstyle.evaluation.label_import.v1",
            "status": (
                "BLIND_LABELS_COMPLETE" if complete else "BLIND_LABELS_INCOMPLETE"
            ),
            "evidence_class": evidence_class,
            "formal_eligible": formal,
            "labeled_artifact_count": len(
                {item.blind_artifact_id for item in evidence.labels}
            ),
            "missing_blind_artifact_ids": list(evidence.missing_blind_artifact_ids),
            "blind_labels_sha256": evidence_sha256(label_data).value,
            "label_approval_receipt_sha256": evidence_sha256(approval_data).value,
            "machine_ledger_sha256": evidence_sha256(ledger_data).value,
            "sealed_protocol_sha256": evidence_sha256(sealed_data).value,
        }
    )
