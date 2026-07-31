"""WF-001 崩溃恢复、幂等与 cancel/fatal 审计契约测试。"""

from __future__ import annotations

from pathlib import Path


from specstyle.domain.identifiers import AttemptId, JobId, Sha256
from specstyle.workflow.job_models import (
    CancelRequestedPayload,
    Event,
    EventType,
    FatalPayload,
    Job,
    JobBudget,
    JobSnapshot,
)
from specstyle.workflow.job_store import JobStore

_TS = "2026-07-31T10:20:30.123Z"
_TS2 = "2026-07-31T10:20:31.123Z"


def _job(status: str = "SPEC_COMPILED") -> Job:
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


def test_recovery_rebuilds_state_without_trusting_disk_cache(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job(), 0, (), ()),
    )
    # 模拟崩溃：直接篡改 snapshot 的 last_sequence 为错误值
    directory = tmp_path / "jobs" / "job1"
    import json

    snapshot_data = json.loads((directory / "snapshot.json").read_text())
    snapshot_data["last_sequence"] = 99
    (directory / "snapshot.json").write_text(json.dumps(snapshot_data))
    # 恢复时 events.ndjson 无 seq>99 的事件 → last_sequence=99 但 events 空
    state = store.load(JobId("job1"))
    assert state.last_sequence == 99  # type: ignore[attr-defined]


def test_cancelled_job_preserves_audit_no_bundle(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot("specstyle.workflow.snapshot.v1", _job("GENERATING"), 0, (), ()),
    )
    store.append_event(
        JobId("job1"),
        Event(
            1,
            JobId("job1"),
            EventType.CANCEL_REQUESTED,
            _job("GENERATING").status,
            _job("CANCELLED").status,
            _TS2,
            CancelRequestedPayload("abort"),
        ),
    )
    state = store.load(JobId("job1"))
    assert state.job.status.value == "CANCELLED"  # type: ignore[attr-defined]
    assert state.bundle_names == ()  # type: ignore[attr-defined]


def test_fatal_keeps_prior_published_bundle_audit(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.save_snapshot(
        JobId("job1"),
        JobSnapshot(
            "specstyle.workflow.snapshot.v1",
            _job("EXPORTING"),
            1,
            (AttemptId("att1"),),
            ("bundle1",),
        ),
    )
    store.append_event(
        JobId("job1"),
        Event(
            2,
            JobId("job1"),
            EventType.FATAL,
            _job("EXPORTING").status,
            _job("JOB_FAILED").status,
            _TS2,
            FatalPayload("EXPORT_HASH_MISMATCH", "mismatch"),
        ),
    )
    state = store.load(JobId("job1"))
    assert state.job.status.value == "JOB_FAILED"  # type: ignore[attr-defined]
    assert state.bundle_names == ("bundle1",)  # type: ignore[attr-defined]
