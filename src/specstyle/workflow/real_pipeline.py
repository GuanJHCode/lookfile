"""WF-003 real pipeline orchestration: serial stages, cancel, recovery hooks.

Uses injected backends/verifiers; no Gradio truth. CPU-safe with FakeBackend.
"""

from __future__ import annotations

from dataclasses import dataclass

from specstyle.errors import DomainError
from specstyle.generation.protocols import GenerationBackend
from specstyle.verification.protocols import Verifier
from specstyle.workflow.job_models import JobId, JobStatus
from specstyle.workflow.job_store import JobStore
from specstyle.workflow.orchestrator import FakeJobPlan, FakeJobResult, run_fake_job


@dataclass(slots=True)
class CancelToken:
    _cancelled: bool = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


@dataclass(frozen=True, slots=True)
class PipelineServices:
    backend: GenerationBackend
    verifier: Verifier


def run_production_job(
    *,
    spec_text: object,
    context: object,
    source: object,
    prompt: object,
    control_builder: object,
    environment: object,
    plan: FakeJobPlan,
    job_store: JobStore,
    root_fd: int,
    bundle_name: str,
    services: PipelineServices,
    cancel: CancelToken | None = None,
) -> FakeJobResult:
    """Single-GPU serial job; cancel before start raises DomainError."""
    if cancel is not None and cancel.cancelled:
        raise DomainError("job cancelled")
    if not isinstance(services, PipelineServices):
        raise DomainError("invalid pipeline services")
    # Resume path: terminal jobs short-circuit inside run_fake_job.
    result = run_fake_job(
        spec_text,
        context,  # type: ignore[arg-type]
        source,  # type: ignore[arg-type]
        prompt,  # type: ignore[arg-type]
        control_builder,
        environment,  # type: ignore[arg-type]
        plan,
        job_store,
        root_fd,
        bundle_name,
        verifier=services.verifier,
        backend=services.backend,
    )
    if cancel is not None and cancel.cancelled and result.final_status == "COMPLETED":
        # Late cancel after completion: leave bundle (commit already done).
        return result
    return result


def job_is_resumable(job_store: JobStore, job_id: JobId) -> bool:
    snap = job_store.get_snapshot(job_id)
    if snap is None:
        return False
    return snap.job.status not in (
        JobStatus.COMPLETED,
        JobStatus.JOB_FAILED,
        JobStatus.CANCELLED,
    )
