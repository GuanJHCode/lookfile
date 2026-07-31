"""Gradio application shell — lazy import; services injected.

UI-001: Spec edit/compile feedback
UI-002: job status, preview/production labels, QA table (UNVERIFIABLE ≠ PASS)
UI-003: repair timeline, export summary, replay assessment
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from specstyle.errors import DomainError
from specstyle.ui.presenters import (
    format_qa_table,
    present_export,
    present_job_status,
    present_qa_report,
    present_repair_timeline,
    present_replay,
    present_spec_compile,
)
from specstyle.ui.view_models import (
    ExportView,
    JobStatusView,
    QaRuleView,
    RepairStepView,
    ReplayView,
    SpecEditorView,
)
from specstyle.workflow.job_models import JobStatus
from specstyle.workflow.orchestrator import FakeJobResult


@dataclass(frozen=True, slots=True)
class UiServices:
    """Injected application services — UI never constructs domain backends."""

    compile_spec: Callable[[str], SpecEditorView]
    get_job_status: Callable[[], JobStatusView] | None = None
    cancel_job: Callable[[], str] | None = None
    get_qa_table: Callable[[], str] | None = None
    get_repair_timeline: Callable[[], str] | None = None
    get_export_summary: Callable[[], str] | None = None
    run_replay: Callable[[], str] | None = None


def build_default_services(context: object) -> UiServices:
    from specstyle.spec.compiled_models import CompilerContext

    if type(context) is not CompilerContext:
        raise DomainError("invalid compiler context")

    def _compile(text: str) -> SpecEditorView:
        return present_spec_compile(text, context)

    return UiServices(_compile)


def bind_job_result_services(
    base: UiServices,
    *,
    job_id: str,
    result: FakeJobResult | None,
    profile: str,
    on_cancel: Callable[[], None] | None,
    qa_report: object | None = None,
    replay_assessment: object | None = None,
) -> UiServices:
    """Attach UI-002/003 handlers from a finished or in-flight job snapshot."""

    def status() -> JobStatusView:
        if result is None:
            return present_job_status(
                job_id, JobStatus.GENERATING, "running", profile=profile
            )
        return present_job_status(job_id, result.final_status, "", profile=profile)

    def cancel() -> str:
        if on_cancel is None:
            return "cancel unavailable"
        on_cancel()
        return "cancel requested"

    def qa() -> str:
        if qa_report is None:
            return "no qa"
        views = present_qa_report(qa_report)  # type: ignore[arg-type]
        return format_qa_table(views)

    def repair() -> str:
        if result is None:
            return "no repair"
        steps = present_repair_timeline(result)
        return "\n".join(
            f"{s.artifact_id}\trounds={s.rounds}\tstop={s.stop_reason}" for s in steps
        )

    def export() -> str:
        if result is None:
            return "no export"
        view = present_export(result)
        return (
            f"bundle={view.bundle_name} approved={view.approved_count} "
            f"rejected={view.rejected_count} review={view.manual_review_count} "
            f"sha={view.bundle_sha256}"
        )

    def replay() -> str:
        if replay_assessment is None:
            return "no replay"
        view = present_replay(replay_assessment)  # type: ignore[arg-type]
        return f"{view.status}\t{view.mode}\t{','.join(view.reasons)}"

    return UiServices(
        base.compile_spec,
        get_job_status=status,
        cancel_job=cancel,
        get_qa_table=qa,
        get_repair_timeline=repair,
        get_export_summary=export,
        run_replay=replay,
    )


def create_app(services: UiServices) -> Any:
    """Create Gradio Blocks app. Requires `gradio` installed."""
    if type(services) is not UiServices:
        raise DomainError("invalid ui services")
    try:
        import gradio as gr
    except ImportError as exc:
        raise DomainError("gradio not installed") from exc

    def on_compile(spec_text: str) -> str:
        view = services.compile_spec(spec_text or "")
        if view.compile_ok:
            return f"OK {view.spec_id} hash={view.compiled_hash}"
        return "ERR " + "; ".join(view.errors)

    def on_status() -> str:
        if services.get_job_status is None:
            return "no job"
        v = services.get_job_status()
        return (
            f"{v.job_id}\t{v.status}\tprofile={v.profile_label}\t"
            f"cancelable={v.can_cancel}\t{v.message}"
        )

    def on_cancel() -> str:
        if services.cancel_job is None:
            return "cancel unavailable"
        return services.cancel_job()

    def on_qa() -> str:
        if services.get_qa_table is None:
            return "no qa"
        return services.get_qa_table()

    def on_repair() -> str:
        if services.get_repair_timeline is None:
            return "no repair"
        return services.get_repair_timeline()

    def on_export() -> str:
        if services.get_export_summary is None:
            return "no export"
        return services.get_export_summary()

    def on_replay() -> str:
        if services.run_replay is None:
            return "no replay"
        return services.run_replay()

    with gr.Blocks(title="SpecStyle") as demo:
        gr.Markdown("# SpecStyle StyleOps")
        with gr.Tab("Spec (UI-001)"):
            spec_in = gr.Textbox(label="Style Spec YAML/JSON", lines=12)
            compile_out = gr.Textbox(label="Compile result")
            gr.Button("Compile").click(
                on_compile, inputs=[spec_in], outputs=[compile_out]
            )
        with gr.Tab("Job / QA (UI-002)"):
            status_out = gr.Textbox(label="Job status (preview|production labeled)")
            gr.Button("Refresh status").click(on_status, outputs=[status_out])
            gr.Button("Cancel job").click(on_cancel, outputs=[status_out])
            qa_out = gr.Textbox(label="QA table (UNVERIFIABLE not green PASS)")
            gr.Button("Refresh QA").click(on_qa, outputs=[qa_out])
        with gr.Tab("Repair / Export / Replay (UI-003)"):
            repair_out = gr.Textbox(label="Repair timeline")
            gr.Button("Repair timeline").click(on_repair, outputs=[repair_out])
            export_out = gr.Textbox(label="Export summary")
            gr.Button("Export summary").click(on_export, outputs=[export_out])
            replay_out = gr.Textbox(label="Replay assessment")
            gr.Button("Replay").click(on_replay, outputs=[replay_out])
    return demo


def launch_app(services: UiServices, **kwargs: object) -> Any:
    app = create_app(services)
    return app.launch(**kwargs)


# Re-export presenters helpers for tests without Gradio
__all__ = [
    "UiServices",
    "bind_job_result_services",
    "build_default_services",
    "create_app",
    "launch_app",
    "ExportView",
    "JobStatusView",
    "QaRuleView",
    "RepairStepView",
    "ReplayView",
    "SpecEditorView",
    "format_qa_table",
    "present_export",
    "present_job_status",
    "present_qa_report",
    "present_repair_timeline",
    "present_replay",
    "present_spec_compile",
]
