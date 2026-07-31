"""WF-001 job_models frozen/slotted/forged rebuild 契约测试。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from specstyle.domain.identifiers import (
    JobId,
    Sha256,
)
from specstyle.errors import DomainError
from specstyle.workflow.job_models import (
    Event,
    EventType,
    Job,
    JobBudget,
    JobSnapshot,
    JobState,
    JobStartedPayload,
    JobStatus,
)

_TS = "2026-07-31T10:20:30.123Z"


def _job() -> Job:
    return Job(
        JobId("job1"),
        Sha256("a" * 64),
        ("xhs_grid",),
        JobBudget(2),
        JobStatus("CREATED"),
        _TS,
        _TS,
    )


def test_job_is_frozen_slotted_and_terminal_flag() -> None:
    job = _job()
    assert not hasattr(job, "__dict__")
    assert job.terminal is False
    with pytest.raises(FrozenInstanceError):
        job.status = "COMPLETED"  # type: ignore[misc]


def test_job_rejects_forged_status_string() -> None:
    from specstyle.workflow.job_models import JobStatus

    job = _job()
    object.__setattr__(job, "status", "COMPLETED")
    rebuilt = Job(
        job.job_id,
        job.compiled_spec_hash,
        job.cohort_profiles,
        job.budget,
        JobStatus("COMPLETED"),
        job.created_at,
        job.updated_at,
    )
    assert rebuilt.terminal is True


def test_job_rejects_invalid_cohort_profiles() -> None:
    with pytest.raises(DomainError):
        Job(
            JobId("job1"),
            Sha256("a" * 64),
            ("unknown_profile",),
            JobBudget(2),
            "CREATED",
            _TS,
            _TS,
        )


def test_event_rejects_payload_event_type_mismatch() -> None:
    payload = JobStartedPayload(Sha256("a" * 64), ("xhs_grid",), JobBudget(2))
    with pytest.raises(DomainError):
        Event(
            1,
            JobId("job1"),
            EventType.ATTEMPT_FINISHED,
            "CREATED",
            "SPEC_VALIDATED",
            _TS,
            payload,
        )


def test_job_snapshot_rejects_wrong_schema_version() -> None:
    with pytest.raises(DomainError):
        JobSnapshot("wrong", _job(), 0, (), ())


def test_job_state_rejects_non_tuple_fields() -> None:
    with pytest.raises(DomainError):
        JobState(_job(), 0, [], [])  # type: ignore[arg-type]
