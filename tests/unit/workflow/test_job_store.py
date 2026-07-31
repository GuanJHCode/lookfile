"""WF-001 JobStore + 状态机 + 恢复契约测试（§5, INV-W1..W15）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from specstyle.domain.enums import (
    ArtifactStatus,
    DecisionReason,
    RepairStopReason,
)
from specstyle.domain.identifiers import (
    ArtifactId,
    AttemptId,
    JobId,
    Sha256,
)
from specstyle.errors import DomainError, InfrastructureError
from specstyle.workflow.job_models import (
    AttemptFinishedPayload,
    AttemptStartedPayload,
    CancelRequestedPayload,
    Event,
    EventType,
    ExportPublishedPayload,
    FatalPayload,
    Job,
    JobBudget,
    JobSnapshot,
    JobStartedPayload,
    VerifierFinishedPayload,
)
from specstyle.workflow.job_store import JobStore
from specstyle.workflow.state_machine import (
    TRANSITIONS,
    validate_transition,
)

_TS = "2026-07-31T10:20:30.123Z"
_TS2 = "2026-07-31T10:20:31.123Z"


def _job(status="CREATED") -> Job:
    from specstyle.workflow.job_models import JobStatus

    return Job(
        JobId("job1"),
        Sha256("a" * 64),
        ("xhs_grid",),
        JobBudget(2),
        JobStatus(status),
        _TS,
        _TS,
    )


def _event(
    seq: int,
    event_type: EventType,
    from_state: str,
    to_state: str,
    payload: object,
    ts: str = _TS2,
) -> Event:
    from specstyle.workflow.job_models import JobStatus

    return Event(
        seq,
        JobId("job1"),
        event_type,
        JobStatus(from_state),
        JobStatus(to_state),
        ts,
        payload,
    )


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path)


def _start_payload() -> JobStartedPayload:
    return JobStartedPayload(Sha256("a" * 64), ("xhs_grid",), JobBudget(2))


def _attempt_payload(attempt: str = "att1") -> AttemptStartedPayload:
    return AttemptStartedPayload(0, 0, AttemptId(attempt), None)


def _attempt_finished_payload(attempt: str = "att1") -> AttemptFinishedPayload:
    return AttemptFinishedPayload(
        0, 0, AttemptId(attempt), ArtifactId("art1"), Sha256("b" * 64)
    )


def _verifier_payload(status="APPROVED") -> VerifierFinishedPayload:
    return VerifierFinishedPayload(
        0,
        0,
        ArtifactId("art1"),
        ArtifactStatus(status),
        DecisionReason.ALL_REQUIRED_PASS,
        RepairStopReason.PASS_ALL_REQUIRED,
    )


def _export_payload(name="bundle1") -> ExportPublishedPayload:
    return ExportPublishedPayload(
        name, Sha256("c" * 64), Sha256("d" * 64), Sha256("e" * 64)
    )


def _seed_started(store: JobStore, tmp_path: Path) -> None:
    store.save_snapshot(
        JobId("job1"), JobSnapshot("specstyle.workflow.snapshot.v1", _job(), 0, (), ())
    )
    store.append_event(
        JobId("job1"),
        _event(1, EventType.JOB_STARTED, "CREATED", "SPEC_VALIDATED", _start_payload()),
    )


def test_save_and_load_initial_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job(), 0, (), ()),
    )
    state = store.load(JobId("job1"))
    assert state.job.status.value == "CREATED"  # type: ignore[attr-defined]
    assert state.last_sequence == 0  # type: ignore[attr-defined]


def test_append_event_and_replay(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_started(store, tmp_path)
    state = store.load(JobId("job1"))
    assert state.job.status.value == "SPEC_VALIDATED"  # type: ignore[attr-defined]
    assert state.last_sequence == 1  # type: ignore[attr-defined]


def test_rejects_invalid_transition(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job(), 0, (), ()),
    )
    with pytest.raises(DomainError, match="invalid job transition"):
        store.append_event(
            JobId("job1"),
            _event(1, EventType.JOB_STARTED, "CREATED", "VERIFYING", _start_payload()),
        )


def test_rejects_duplicate_attempt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job("SPEC_COMPILED"), 0, (), ()),
    )
    store.append_event(
        JobId("job1"),
        _event(
            1,
            EventType.ATTEMPT_STARTED,
            "SPEC_COMPILED",
            "GENERATING",
            _attempt_payload("att1"),
        ),
    )
    with pytest.raises(DomainError, match="duplicate job attempt"):
        store.append_event(
            JobId("job1"),
            _event(
                2,
                EventType.ATTEMPT_STARTED,
                "VERIFYING",
                "GENERATING",
                _attempt_payload("att1"),
            ),
        )


def test_rejects_duplicate_export(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot(
            "specstyle.workflow.snapshot.v1",
            _job("EXPORTING"),
            0,
            (AttemptId("att1"),),
            ("bundle1",),
        ),
    )
    with pytest.raises(DomainError, match="duplicate job export"):
        store.append_event(
            JobId("job1"),
            _event(
                1,
                EventType.EXPORT_PUBLISHED,
                "EXPORTING",
                "COMPLETED",
                _export_payload("bundle1"),
            ),
        )


def test_rejects_terminal_progress(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job("COMPLETED"), 0, (), ()),
    )
    with pytest.raises(DomainError, match="job is terminal"):
        store.append_event(
            JobId("job1"),
            _event(
                1,
                EventType.EXPORT_PUBLISHED,
                "EXPORTING",
                "COMPLETED",
                _export_payload(),
            ),
        )


def test_partial_json_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    directory = tmp_path / "jobs" / "job1"
    directory.mkdir(parents=True)
    (directory / "snapshot.json").write_bytes(
        b'{"schema_version":"specstyle.workflow.snapsho'
    )
    with pytest.raises(InfrastructureError, match="job store corrupted"):
        store.load(JobId("job1"))


def test_corrupted_snapshot_unknown_key_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    directory = tmp_path / "jobs" / "job1"
    directory.mkdir(parents=True)
    (directory / "snapshot.json").write_bytes(
        b'{"schema_version":"specstyle.workflow.snapshot.v1","job":{},"last_sequence":0,"attempt_ids":[],"bundle_names":[],"extra":1}'
    )
    with pytest.raises(InfrastructureError, match="job store corrupted"):
        store.load(JobId("job1"))


def test_cancel_race_blocks_subsequent_events(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job("GENERATING"), 0, (), ()),
    )
    store.append_event(
        JobId("job1"),
        _event(
            1,
            EventType.CANCEL_REQUESTED,
            "GENERATING",
            "CANCELLED",
            CancelRequestedPayload("user abort"),
        ),
    )
    with pytest.raises(DomainError, match="job is terminal"):
        store.append_event(
            JobId("job1"),
            _event(
                2,
                EventType.ATTEMPT_FINISHED,
                "GENERATING",
                "VERIFYING",
                _attempt_finished_payload(),
            ),
        )


def test_fatal_blocks_export(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job("GENERATING"), 0, (), ()),
    )
    store.append_event(
        JobId("job1"),
        _event(
            1,
            EventType.FATAL,
            "GENERATING",
            "JOB_FAILED",
            FatalPayload("GENERATION_OOM", "oom"),
        ),
    )
    with pytest.raises(DomainError, match="job is terminal"):
        store.append_event(
            JobId("job1"),
            _event(
                2,
                EventType.EXPORT_PUBLISHED,
                "EXPORTING",
                "COMPLETED",
                _export_payload(),
            ),
        )


def test_recovery_replays_events_after_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job("SPEC_COMPILED"), 0, (), ()),
    )
    store.append_event(
        JobId("job1"),
        _event(
            1,
            EventType.ATTEMPT_STARTED,
            "SPEC_COMPILED",
            "GENERATING",
            _attempt_payload("att1"),
        ),
    )
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot(
            "specstyle.workflow.snapshot.v1",
            _job("GENERATING"),
            1,
            (AttemptId("att1"),),
            (),
        ),
    )
    store.append_event(
        JobId("job1"),
        _event(
            2,
            EventType.ATTEMPT_FINISHED,
            "GENERATING",
            "VERIFYING",
            _attempt_finished_payload("att1"),
        ),
    )
    state = store.load(JobId("job1"))
    assert state.job.status.value == "VERIFYING"  # type: ignore[attr-defined]
    assert state.last_sequence == 2  # type: ignore[attr-defined]
    assert [a.value for a in state.attempt_ids] == ["att1"]  # type: ignore[attr-defined]


def test_rejects_out_of_order_sequence(tmp_path: Path) -> None:
    from specstyle.workflow.job_store import _canonical_json, _event_to_primitive

    store = _store(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job("SPEC_COMPILED"), 0, (), ()),
    )
    directory = tmp_path / "jobs" / "job1"
    directory.mkdir(parents=True, exist_ok=True)
    e1 = _event(
        1,
        EventType.ATTEMPT_STARTED,
        "SPEC_COMPILED",
        "GENERATING",
        _attempt_payload("att1"),
    )
    e3 = _event(
        3,
        EventType.ATTEMPT_FINISHED,
        "GENERATING",
        "VERIFYING",
        _attempt_finished_payload("att1"),
    )
    with open(directory / "events.ndjson", "wb") as handle:
        handle.write(_canonical_json(_event_to_primitive(e1)) + b"\n")
        handle.write(_canonical_json(_event_to_primitive(e3)) + b"\n")
    with pytest.raises(DomainError, match="invalid job event"):
        store.load(JobId("job1"))


def test_load_unknown_job_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(DomainError, match="job not found"):
        store.load(JobId("missing"))


def test_validate_transition_rejects_unknown_mapping(tmp_path: Path) -> None:
    from specstyle.workflow.job_models import JobStatus

    with pytest.raises(DomainError, match="invalid job transition"):
        validate_transition(
            JobStatus.GENERATING, JobStatus.COMPLETED, EventType.ATTEMPT_STARTED
        )


def test_terminal_states_have_no_exits() -> None:
    from specstyle.workflow.job_models import JobStatus

    for status in (JobStatus.COMPLETED, JobStatus.JOB_FAILED, JobStatus.CANCELLED):
        assert TRANSITIONS[status] == frozenset()
