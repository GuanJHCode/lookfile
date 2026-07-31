"""WF-001 状态机转换表与校验契约测试。"""

from __future__ import annotations

import pytest

from specstyle.errors import DomainError
from specstyle.workflow.job_models import EventType, JobStatus
from specstyle.workflow.state_machine import (
    TRANSITIONS,
    _EVENT_TO_STATE,
    validate_transition,
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
