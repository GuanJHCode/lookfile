"""WF-001 Job 状态机：合法转换表、转换校验与事件重放（contracts §5）。

derived from §5。状态命名以 §5 为准（CREATED/SPEC_VALIDATED/.../COMPLETED/
JOB_FAILED/CANCELLED/RECOVERABLE_ERROR）。APPROVED/MANUAL_REVIEW/REJECTED 是
cohort/item 决策中间态（待 EXPORTING），Job 正常终态为 COMPLETED。
"""

from __future__ import annotations

from specstyle.errors import DomainError

from specstyle.workflow.job_models import (
    EventType,
    Job,
    JobSnapshot,
    JobState,
    JobStatus,
)

TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.CREATED: frozenset(
        {JobStatus.SPEC_VALIDATED, JobStatus.CANCELLED, JobStatus.JOB_FAILED}
    ),
    JobStatus.SPEC_VALIDATED: frozenset(
        {JobStatus.SPEC_COMPILED, JobStatus.CANCELLED, JobStatus.JOB_FAILED}
    ),
    JobStatus.SPEC_COMPILED: frozenset(
        {JobStatus.GENERATING, JobStatus.CANCELLED, JobStatus.JOB_FAILED}
    ),
    JobStatus.GENERATING: frozenset(
        {
            JobStatus.VERIFYING,
            JobStatus.RECOVERABLE_ERROR,
            JobStatus.CANCELLED,
            JobStatus.JOB_FAILED,
        }
    ),
    JobStatus.VERIFYING: frozenset(
        {
            JobStatus.APPROVED,
            JobStatus.MANUAL_REVIEW,
            JobStatus.REJECTED,
            JobStatus.REPAIR_SELECTING,
            JobStatus.RECOVERABLE_ERROR,
            JobStatus.CANCELLED,
            JobStatus.JOB_FAILED,
        }
    ),
    JobStatus.APPROVED: frozenset(
        {JobStatus.EXPORTING, JobStatus.CANCELLED, JobStatus.JOB_FAILED}
    ),
    JobStatus.MANUAL_REVIEW: frozenset(
        {JobStatus.EXPORTING, JobStatus.CANCELLED, JobStatus.JOB_FAILED}
    ),
    JobStatus.REJECTED: frozenset(
        {JobStatus.EXPORTING, JobStatus.CANCELLED, JobStatus.JOB_FAILED}
    ),
    JobStatus.REPAIR_SELECTING: frozenset(
        {
            JobStatus.REPAIRING,
            JobStatus.REJECTED,
            JobStatus.RECOVERABLE_ERROR,
            JobStatus.CANCELLED,
            JobStatus.JOB_FAILED,
        }
    ),
    JobStatus.REPAIRING: frozenset(
        {
            JobStatus.VERIFYING,
            JobStatus.RECOVERABLE_ERROR,
            JobStatus.CANCELLED,
            JobStatus.JOB_FAILED,
        }
    ),
    JobStatus.EXPORTING: frozenset(
        {
            JobStatus.COMPLETED,
            JobStatus.RECOVERABLE_ERROR,
            JobStatus.CANCELLED,
            JobStatus.JOB_FAILED,
        }
    ),
    JobStatus.RECOVERABLE_ERROR: frozenset(
        {
            JobStatus.GENERATING,
            JobStatus.VERIFYING,
            JobStatus.REPAIR_SELECTING,
            JobStatus.REPAIRING,
            JobStatus.EXPORTING,
            JobStatus.JOB_FAILED,
        }
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.JOB_FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}

_EVENT_TO_STATE: dict[EventType, frozenset[JobStatus]] = {
    EventType.JOB_STARTED: frozenset({JobStatus.SPEC_VALIDATED}),
    EventType.ATTEMPT_STARTED: frozenset({JobStatus.GENERATING}),
    EventType.ATTEMPT_FINISHED: frozenset({JobStatus.VERIFYING}),
    EventType.VERIFIER_FINISHED: frozenset(
        {
            JobStatus.APPROVED,
            JobStatus.MANUAL_REVIEW,
            JobStatus.REJECTED,
            JobStatus.REPAIR_SELECTING,
        }
    ),
    EventType.REPAIR_STEP: frozenset({JobStatus.REPAIRING, JobStatus.VERIFYING}),
    EventType.EXPORT_PUBLISHED: frozenset({JobStatus.COMPLETED}),
    EventType.CANCEL_REQUESTED: frozenset({JobStatus.CANCELLED}),
    EventType.FATAL: frozenset({JobStatus.JOB_FAILED}),
}


def validate_transition(
    from_state: JobStatus,
    to_state: JobStatus,
    event_type: EventType,
    /,
) -> None:
    """校验 (from→to)∈TRANSITIONS ∧ event_type↔to_state 匹配。"""
    if to_state not in TRANSITIONS.get(from_state, frozenset()):
        raise DomainError("invalid job transition") from None
    if to_state not in _EVENT_TO_STATE.get(event_type, frozenset()):
        raise DomainError("invalid job transition") from None


def replay_events(snapshot: JobSnapshot, events: tuple[object, ...], /) -> JobState:
    """从 snapshot 之后重放事件，逐事件校验转换+幂等+推进。

    重复 sequence/attempt_id/bundle_name、乱序（非连续递增）→
    ``DomainError("invalid job event") from None``。
    """
    status = snapshot.job.status
    last = snapshot.last_sequence
    attempts: list[str] = [a.value for a in snapshot.attempt_ids]
    bundles: list[str] = list(snapshot.bundle_names)
    updated_at = snapshot.job.updated_at
    for event in events:
        if event.sequence != last + 1:  # type: ignore[attr-defined]
            raise DomainError("invalid job event") from None
        validate_transition(status, event.to_state, event.event_type)  # type: ignore[attr-defined]
        _check_idempotent(event, attempts, bundles)  # type: ignore[arg-type]
        status = event.to_state  # type: ignore[attr-defined]
        last = event.sequence  # type: ignore[attr-defined]
        updated_at = event.timestamp  # type: ignore[attr-defined]
    new_job = Job(
        snapshot.job.job_id,
        snapshot.job.compiled_spec_hash,
        snapshot.job.cohort_profiles,
        snapshot.job.budget,
        status,
        snapshot.job.created_at,
        updated_at,
    )
    from specstyle.domain.identifiers import AttemptId

    return JobState(
        new_job,
        last,
        tuple(AttemptId(a) for a in attempts),
        tuple(bundles),
    )


def _check_idempotent(event: object, attempts: list[str], bundles: list[str]) -> None:
    if event.event_type is EventType.ATTEMPT_STARTED:  # type: ignore[attr-defined]
        aid = event.payload.attempt_id.value  # type: ignore[attr-defined]
        if aid in attempts:
            raise DomainError("invalid job event") from None
        attempts.append(aid)
    elif event.event_type is EventType.EXPORT_PUBLISHED:  # type: ignore[attr-defined]
        bn = event.payload.bundle_name  # type: ignore[attr-defined]
        if bn in bundles:
            raise DomainError("invalid job event") from None
        bundles.append(bn)
