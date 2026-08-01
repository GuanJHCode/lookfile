"""WF-001 状态机转换表与校验契约测试。"""

from __future__ import annotations

import pytest

from specstyle.errors import DomainError
from specstyle.domain.identifiers import JobId, Sha256
from specstyle.workflow.job_models import (
    Event,
    EventType,
    Job,
    JobBudget,
    JobSnapshot,
    JobStartedPayload,
    JobStatus,
)
from specstyle.workflow.state_machine import (
    TRANSITIONS,
    _EVENT_TO_STATE,
    replay_events,
    validate_transition,
)

_TS = "2026-07-31T10:20:30.123Z"


def _snapshot() -> JobSnapshot:
    return JobSnapshot(
        "specstyle.workflow.snapshot.v1",
        Job(
            JobId("job1"),
            Sha256("a" * 64),
            ("xhs_grid",),
            JobBudget(2),
            JobStatus.CREATED,
            _TS,
            _TS,
        ),
        0,
        (),
        (),
    )


def test_terminal_states_have_no_outgoing_transitions() -> None:
    for status in (JobStatus.COMPLETED, JobStatus.JOB_FAILED, JobStatus.CANCELLED):
        assert TRANSITIONS[status] == frozenset()


def test_every_non_terminal_state_reaches_a_terminal() -> None:
    reachable: set[JobStatus] = set()
    frontier = {JobStatus.COMPLETED, JobStatus.JOB_FAILED, JobStatus.CANCELLED}
    while frontier:
        nxt = frontier.pop()
        for src, targets in TRANSITIONS.items():
            if nxt in targets and src not in reachable:
                reachable.add(src)
                frontier.add(src)
    non_terminal = {s for s in TRANSITIONS if TRANSITIONS[s]}
    assert non_terminal <= reachable


def test_event_to_state_mapping_covers_all_event_types() -> None:
    assert set(_EVENT_TO_STATE) == set(EventType)


@pytest.mark.parametrize(
    ("frm", "to", "event_type", "ok"),
    [
        ("CREATED", "SPEC_VALIDATED", "JOB_STARTED", True),
        ("CREATED", "VERIFYING", "ATTEMPT_STARTED", False),
        ("GENERATING", "APPROVED", "VERIFIER_FINISHED", False),
        ("EXPORTING", "COMPLETED", "EXPORT_PUBLISHED", True),
        ("COMPLETED", "JOB_FAILED", "FATAL", False),
    ],
)
def test_validate_transition(frm: str, to: str, event_type: str, ok: bool) -> None:
    if ok:
        validate_transition(JobStatus(frm), JobStatus(to), EventType(event_type))
    else:
        with pytest.raises(DomainError, match="invalid job transition"):
            validate_transition(JobStatus(frm), JobStatus(to), EventType(event_type))


def test_replay_rejects_event_for_another_job() -> None:
    event = Event(
        1,
        JobId("other"),
        EventType.JOB_STARTED,
        JobStatus.CREATED,
        JobStatus.SPEC_VALIDATED,
        _TS,
        JobStartedPayload(Sha256("a" * 64), ("xhs_grid",), JobBudget(2)),
    )
    with pytest.raises(DomainError, match="invalid job event"):
        replay_events(_snapshot(), (event,))


@pytest.mark.parametrize(
    ("sequence", "from_state", "timestamp"),
    [
        (2, JobStatus.CREATED, _TS),
        (1, JobStatus.SPEC_VALIDATED, _TS),
        (1, JobStatus.CREATED, "2026-07-31T10:20:29.123Z"),
    ],
)
def test_replay_rejects_sequence_from_state_and_timestamp_regression(
    sequence: int, from_state: JobStatus, timestamp: str
) -> None:
    event = Event(
        sequence,
        JobId("job1"),
        EventType.JOB_STARTED,
        JobStatus.CREATED,
        JobStatus.SPEC_VALIDATED,
        timestamp,
        JobStartedPayload(Sha256("a" * 64), ("xhs_grid",), JobBudget(2)),
    )
    object.__setattr__(event, "from_state", from_state)
    with pytest.raises(DomainError, match="invalid job event"):
        replay_events(_snapshot(), (event,))


def test_replay_rejects_duck_subclass_and_unbound_genesis_payload() -> None:
    class EventSubclass(Event):
        pass

    valid_args = (
        1,
        JobId("job1"),
        EventType.JOB_STARTED,
        JobStatus.CREATED,
        JobStatus.SPEC_VALIDATED,
        _TS,
        JobStartedPayload(Sha256("a" * 64), ("xhs_grid",), JobBudget(2)),
    )
    with pytest.raises(DomainError, match="invalid job event"):
        replay_events(_snapshot(), [Event(*valid_args)])  # type: ignore[arg-type]
    with pytest.raises(DomainError, match="invalid job event"):
        replay_events(_snapshot(), (EventSubclass(*valid_args),))
    with pytest.raises(DomainError, match="invalid job event"):
        replay_events(
            _snapshot(),
            (
                Event(
                    *valid_args[:-1],
                    JobStartedPayload(Sha256("b" * 64), ("xhs_grid",), JobBudget(2)),
                ),
            ),
        )


def test_replay_normalizes_forged_duplicate_attempt_ledger() -> None:
    from specstyle.domain.identifiers import AttemptId

    snapshot = _snapshot()
    object.__setattr__(snapshot, "attempt_ids", (AttemptId("att1"), AttemptId("att1")))
    with pytest.raises(DomainError, match="invalid job event"):
        replay_events(snapshot, ())
