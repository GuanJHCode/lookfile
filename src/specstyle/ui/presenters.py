"""UI presenters: map domain objects to view models; redact errors."""

from __future__ import annotations

from specstyle.errors import DomainError, SpecStyleError
from specstyle.exporting.bundle import ExportBundle
from specstyle.spec.compiler import compile_style_spec
from specstyle.spec.compiled_models import CompilerContext
from specstyle.spec.loader import load_style_spec_text
from specstyle.ui.view_models import (
    ExportView,
    JobStatusView,
    QaRuleView,
    SpecEditorView,
)
from specstyle.verification.rule_models import RuleResult, VerificationReport
from specstyle.workflow.job_models import JobStatus
from specstyle.workflow.orchestrator import FakeJobResult


def _redact(exc: BaseException) -> str:
    if isinstance(exc, SpecStyleError):
        return str(exc)
    return "internal error"


def present_spec_compile(spec_text: str, context: CompilerContext, /) -> SpecEditorView:
    try:
        raw = load_style_spec_text(spec_text)
        compiled = compile_style_spec(raw, context)
        return SpecEditorView(
            raw.schema_version,
            raw.metadata.spec_id,
            True,
            (),
            compiled.compiled_spec_hash.value,
        )
    except Exception as exc:
        return SpecEditorView("?", "?", False, (_redact(exc),), None)


def present_job_status(
    job_id: str, status: JobStatus | str, message: str = ""
) -> JobStatusView:
    value = status.value if isinstance(status, JobStatus) else str(status)
    return JobStatusView(job_id, value, message)


def present_qa_report(report: VerificationReport) -> tuple[QaRuleView, ...]:
    if type(report) is not VerificationReport:
        raise DomainError("invalid report")
    views: list[QaRuleView] = []
    for result in report.results:
        if type(result) is not RuleResult:
            continue
        views.append(
            QaRuleView(result.rule_id.value, result.status.value, result.score)
        )
    return tuple(views)


def present_export(result: FakeJobResult) -> ExportView:
    if type(result) is not FakeJobResult:
        raise DomainError("invalid job result")
    approved = rejected = review = 0
    for cohort in result.cohorts:
        for item in cohort.items:
            st = item.terminal.artifact_decision.artifact_status.value
            if st == "APPROVED":
                approved += 1
            elif st == "REJECTED":
                rejected += 1
            else:
                review += 1
    bundle_hash = None
    name = ""
    if isinstance(result.bundle, ExportBundle):
        bundle_hash = result.bundle.bundle_sha256.value
        name = result.bundle.bundle_name
    return ExportView(name, approved, rejected, review, bundle_hash)


def present_repair_timeline(result: FakeJobResult) -> tuple[str, ...]:
    lines: list[str] = []
    for history in result.histories:
        lines.append(
            f"artifact={history.current_target_artifact_id.value} rounds={history.rounds}"
        )
    return tuple(lines)
