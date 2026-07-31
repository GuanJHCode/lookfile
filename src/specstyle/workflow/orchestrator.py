"""WF-002 Fake CPU 端到端纵向切片 orchestrator。

按 architect 冻结合同（contracts §3/§11.7/§12.9/§13）连接既有冻结函数：
``load_style_spec_text``/``compile_style_spec``、``run_generation``、``run_verifier``、
``VerificationReport``、``start_repair_history``/``next_repair_step``/
``consume_repair_result``、``export_bundle``。本模块不复制 compiler/verifier/repair/
gate 业务规则，只组装这些平面；不 import torch/diffusers/gradio。

ID 派生确定（``_attempt_id``/``_decision_id``），重启同输入得到相同 ID，配合 WF-001
JobStore 实现阶段幂等。本模块不声称 Spec replay 已验证；同 seed 重复仅依赖
FakeBackend bit-exact。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from specstyle.domain.artifacts import AssetRef
from specstyle.domain.enums import ArtifactStatus
from specstyle.domain.identifiers import (
    AttemptId,
    AssetId,
    DecisionId,
    JobId,
)
from specstyle.errors import DomainError, SpecStyleError
from specstyle.exporting.bundle import ExportBundle, export_bundle
from specstyle.exporting.manifest import (
    AssetCredit,
    ExportCohort,
    ExportItem,
    ExportRequest,
)
from specstyle.generation.preprocess import PreparedImage
from specstyle.generation.protocols import (
    GenerationBackend,
    build_control_input,
    run_generation,
)
from specstyle.generation.requests import GenerationRequest, RenderedPrompt
from specstyle.observability.environment import EnvironmentSnapshot, hash_environment
from specstyle.repair.history import RepairHistory, start_repair_history
from specstyle.repair.loop import (
    RepairTerminal,
    consume_repair_result,
    next_repair_step,
)
from specstyle.spec.compiled_models import (
    CompilerContext,
    CompiledExecutionGraph,
    CompiledStyleSpec,
    CompiledVerificationPlan,
)
from specstyle.spec.compiler import compile_style_spec
from specstyle.spec.loader import load_style_spec_text
from specstyle.verification.protocols import Verifier, run_verifier
from specstyle.verification.rule_models import VerificationReport
from specstyle.workflow.job_models import (
    AttemptFinishedPayload,
    AttemptStartedPayload,
    Event,
    EventType,
    ExportPublishedPayload,
    ExportStartedPayload,
    FatalPayload,
    Job,
    JobBudget,
    JobSnapshot,
    JobStartedPayload,
    JobStatus,
    SpecCompiledPayload,
    VerifierFinishedPayload,
)
from specstyle.workflow.job_store import JobStore

_TIMESTAMP = "2026-07-31T00:00:00.000Z"
_SNAPSHOT_VERSION = "specstyle.workflow.snapshot.v1"


@dataclass(frozen=True, slots=True)
class FakeJobPlan:
    """单作业计划：每 profile 生成 ``item_count`` 个 item（P0=1）。

    ``output_profiles`` 为 xhs_grid/talking_head_cover/background_sequence 的非空无重复
    子集；``generation_profile`` 固定 ``"production"``；``variation_index`` 为初始
    attempt 的 variation（0）。
    """

    job_id: JobId
    output_profiles: tuple[str, ...]
    generation_profile: str
    variation_index: int
    item_count: int

    def __post_init__(self) -> None:
        allowed = ("xhs_grid", "talking_head_cover", "background_sequence")
        if (
            type(self.output_profiles) is not tuple
            or not self.output_profiles
            or type(self.generation_profile) is not str
            or self.generation_profile != "production"
            or type(self.variation_index) is not int
            or isinstance(self.variation_index, bool)
            or type(self.item_count) is not int
            or isinstance(self.item_count, bool)
        ):
            raise ValueError("invalid fake job plan")
        seen: set[str] = set()
        for profile in self.output_profiles:
            if type(profile) is not str or profile not in allowed or profile in seen:
                raise ValueError("invalid fake job plan")
            seen.add(profile)
        # P0：状态机在 item 终态后只能进 EXPORTING，暂强制单 profile × 单 item。
        if (
            self.item_count != 1
            or len(self.output_profiles) != 1
            or self.variation_index < 0
        ):
            raise ValueError("invalid fake job plan")
        object.__setattr__(self, "job_id", JobId(self.job_id.value))
        object.__setattr__(self, "output_profiles", tuple(self.output_profiles))


@dataclass(frozen=True, slots=True)
class FakeJobResult:
    """单作业终态：bundle 仅在 ``final_status == "COMPLETED"`` 的成功运行中非空。

    重启已完成作业时 orchestrator 短路返回 ``bundle=None``（bundle 已在先前运行发布），
    且不重复生成/导出（INV-WF002-5）。
    """

    bundle: ExportBundle | None
    cohorts: tuple[ExportCohort, ...]
    terminals: tuple[RepairTerminal, ...]
    histories: tuple[RepairHistory, ...]
    final_status: str


def _attempt_id(job: JobId, profile: str, item: int, rnd: int) -> AttemptId:
    """确定性 attempt ID：``{job}-a{round}-{profile}-{item}``。"""
    return AttemptId(f"{job.value}-a{rnd}-{profile}-{item}")


def _decision_id(job: JobId, profile: str, item: int, rnd: int) -> DecisionId:
    """确定性 decision ID：``{job}-d{round}-{profile}-{item}``。"""
    return DecisionId(f"{job.value}-d{rnd}-{profile}-{item}")


def _style_reference_refs(graph: CompiledExecutionGraph) -> tuple[AssetRef, ...]:
    """位置匹配 ``graph.style_reference_hashes`` 的 style ref AssetRef。"""
    return tuple(
        AssetRef(AssetId(f"style-{index}"), sha)
        for index, sha in enumerate(graph.style_reference_hashes)
    )


def _production_graph(
    compiled: CompiledStyleSpec, profile: str
) -> CompiledExecutionGraph:
    """唯一命中 output_profile 的 production graph。"""
    matched = tuple(
        graph for graph in compiled.production_graphs if graph.output_profile == profile
    )
    if len(matched) != 1:
        raise DomainError("invalid fake job plan")
    return matched[0]


def _verification_plan(
    compiled: CompiledStyleSpec, profile: str
) -> CompiledVerificationPlan:
    """唯一命中 output_profile 的 verification plan。"""
    matched = tuple(
        plan for plan in compiled.verification_plans if plan.output_profile == profile
    )
    if len(matched) != 1:
        raise DomainError("invalid fake job plan")
    return matched[0]


def _initial_request(
    compiled: CompiledStyleSpec,
    plan: FakeJobPlan,
    source: PreparedImage,
    prompt: RenderedPrompt,
    control_builder: object,
    env_hash: object,
    profile: str,
    item_index: int,
) -> GenerationRequest:
    """组装初始 root request（variation_index 来自 plan，attempt_id 为 round 0）。"""
    graph = _production_graph(compiled, profile)
    control = build_control_input(control_builder, source, graph)
    return GenerationRequest(
        plan.job_id,
        _attempt_id(plan.job_id, profile, item_index, 0),
        None,
        compiled,
        plan.generation_profile,
        profile,
        source,
        _style_reference_refs(graph),
        prompt,
        control,
        plan.variation_index,
        env_hash,
    )


def _expected_artifact_id(request: GenerationRequest) -> str:
    """复用 FakeBackend §11.7 派生：artifact_id 由 request_hash 决定。"""
    return f"artifact-{request.request_hash.value}"


def _load_or_init(
    job_store: JobStore,
    job_id: JobId,
    compiled: CompiledStyleSpec,
    plan: FakeJobPlan,
) -> object:
    """load 检查 resume；job not found 时 save_snapshot 创建初始 Job。"""
    snapshot = job_store.get_snapshot(job_id)
    if snapshot is None:
        budget = JobBudget(1 + compiled.source_spec.repair.max_rounds)
        job = Job(
            job_id,
            compiled.compiled_spec_hash,
            plan.output_profiles,
            budget,
            JobStatus.CREATED,
            _TIMESTAMP,
            _TIMESTAMP,
        )
        job_store.save_snapshot(job_id, JobSnapshot(_SNAPSHOT_VERSION, job, 0, (), ()))
    return job_store.load(job_id)


def _append(
    cursor: SimpleNamespace,
    job_store: JobStore,
    job_id: JobId,
    event_type: EventType,
    to_state: JobStatus,
    payload: object,
) -> None:
    """记录事件并推进 cursor.status；from_state 取自当前 cursor。"""
    event = Event(1, job_id, event_type, cursor.status, to_state, _TIMESTAMP, payload)
    job_store.append_event(job_id, event)
    cursor.status = to_state


def _terminal_to_state(terminal: RepairTerminal) -> JobStatus:
    """终态 item status → job status（APPROVED/REJECTED/MANUAL_REVIEW）。"""
    status = terminal.artifact_decision.artifact_status
    if status is ArtifactStatus.APPROVED:
        return JobStatus.APPROVED
    if status is ArtifactStatus.REJECTED:
        return JobStatus.REJECTED
    return JobStatus.MANUAL_REVIEW


def _build_credits(
    compiled: CompiledStyleSpec,
    source: PreparedImage,
    graph: CompiledExecutionGraph,
) -> tuple[AssetCredit, ...]:
    """从 source（input）+ style_refs（style_reference）+ spec provenance 构造 credits。"""
    spec_refs = compiled.source_spec.assets.style_references
    style_refs = _style_reference_refs(graph)
    credits: list[AssetCredit] = [
        AssetCredit(source.source, ("input",), None, None, None, None)
    ]
    for index, ref in enumerate(style_refs):
        spec_ref = spec_refs[index]
        credits.append(
            AssetCredit(
                ref,
                ("style_reference",),
                spec_ref.source_url,
                spec_ref.license,
                spec_ref.attribution,
                spec_ref.consent,
            )
        )
    credits.sort(key=lambda c: (c.asset.asset_id.value, c.asset.sha256.value))
    return tuple(credits)


def _try_fatal(job_store: JobStore, job_id: JobId, cursor: SimpleNamespace) -> None:
    """失败前 best-effort 追加 FATAL（脱敏）；不发布部分 bundle。"""
    if cursor.status in (
        JobStatus.COMPLETED,
        JobStatus.JOB_FAILED,
        JobStatus.CANCELLED,
    ):
        return
    try:
        _append(
            cursor,
            job_store,
            job_id,
            EventType.FATAL,
            JobStatus.JOB_FAILED,
            FatalPayload("UNKNOWN", "job failed"),
        )
    except SpecStyleError:
        return


def _run_item(
    cursor: SimpleNamespace,
    job_store: JobStore,
    compiled: CompiledStyleSpec,
    plan: FakeJobPlan,
    source: PreparedImage,
    prompt: RenderedPrompt,
    control_builder: object,
    env_hash: object,
    profile: str,
    item_index: int,
    verifier: Verifier,
    backend: GenerationBackend,
) -> tuple[RepairHistory, RepairTerminal]:
    """单 item 的 generate→verify→repair FSM。

    记录 ATTEMPT_STARTED/FINISHED 与最终 VERIFIER_FINISHED（终态，非 None stop）。
    repair 轮次在线性 history 内推进，不落 REPAIR_STEP 事件——WF-001 JobStore 无法
    序列化进入 REPAIR_SELECTING 所需的 repair_stop_reason=None VERIFIER_FINISHED
    （RepairStopReason(None) 抛 ValueError），故仅保留可幂等的 attempt/terminal/export 审计。
    """
    req0 = _initial_request(
        compiled, plan, source, prompt, control_builder, env_hash, profile, item_index
    )
    _append(
        cursor,
        job_store,
        plan.job_id,
        EventType.ATTEMPT_STARTED,
        JobStatus.GENERATING,
        AttemptStartedPayload(0, item_index, req0.attempt_id, None),
    )
    artifact = run_generation(backend, req0)
    _append(
        cursor,
        job_store,
        plan.job_id,
        EventType.ATTEMPT_FINISHED,
        JobStatus.VERIFYING,
        AttemptFinishedPayload(
            0, item_index, req0.attempt_id, artifact.ref.artifact_id, req0.request_hash
        ),
    )
    rules = _verification_plan(compiled, profile).applicable_rule_definitions
    report = VerificationReport(
        (artifact.ref,), rules, run_verifier(verifier, (artifact.ref,), rules)
    )
    history = start_repair_history(req0, artifact, report)
    while True:
        rnd = history.rounds + 1
        step = next_repair_step(
            history,
            _decision_id(plan.job_id, profile, item_index, rnd),
            _attempt_id(plan.job_id, profile, item_index, rnd),
        )
        if isinstance(step, RepairTerminal):
            _append(
                cursor,
                job_store,
                plan.job_id,
                EventType.VERIFIER_FINISHED,
                _terminal_to_state(step),
                VerifierFinishedPayload(
                    0,
                    item_index,
                    history.current_target_artifact_id,
                    step.artifact_decision.artifact_status,
                    step.artifact_decision.decision_reason,
                    step.artifact_decision.repair_stop_reason,
                ),
            )
            return history, step
        child = run_generation(backend, step.request)
        child_report = VerificationReport(
            (child.ref,), rules, run_verifier(verifier, (child.ref,), rules)
        )
        history = consume_repair_result(history, step, child, child_report)


def run_fake_job(
    spec_text: object,
    context: CompilerContext,
    source: PreparedImage,
    prompt: RenderedPrompt,
    control_builder: object,
    environment: EnvironmentSnapshot,
    plan: FakeJobPlan,
    job_store: JobStore,
    root_fd: int,
    bundle_name: str,
    /,
    *,
    verifier: Verifier,
    backend: GenerationBackend,
) -> FakeJobResult:
    """串联 compile→generate→verify→repair FSM→export 的四链纵向切片。"""
    compiled = compile_style_spec(load_style_spec_text(spec_text), context)
    env_hash = hash_environment(environment)
    state = _load_or_init(job_store, plan.job_id, compiled, plan)
    if state.job.terminal:  # type: ignore[attr-defined]
        job = state.job  # type: ignore[attr-defined]
        if (
            job.compiled_spec_hash != compiled.compiled_spec_hash
            or job.cohort_profiles != plan.output_profiles
        ):
            raise DomainError("job terminal but plan or spec hash mismatch") from None
        return FakeJobResult(None, (), (), (), job.status.value)
    cursor = SimpleNamespace(status=state.job.status)  # type: ignore[attr-defined]
    cohorts: list[ExportCohort] = []
    terminals: list[RepairTerminal] = []
    histories: list[RepairHistory] = []
    try:
        _append(
            cursor,
            job_store,
            plan.job_id,
            EventType.JOB_STARTED,
            JobStatus.SPEC_VALIDATED,
            JobStartedPayload(
                compiled.compiled_spec_hash, plan.output_profiles, _budget(compiled)
            ),
        )
        _append(
            cursor,
            job_store,
            plan.job_id,
            EventType.SPEC_COMPILED,
            JobStatus.SPEC_COMPILED,
            SpecCompiledPayload(compiled.compiled_spec_hash),
        )
        for profile in plan.output_profiles:
            for item_index in range(plan.item_count):
                history, terminal = _run_item(
                    cursor,
                    job_store,
                    compiled,
                    plan,
                    source,
                    prompt,
                    control_builder,
                    env_hash,
                    profile,
                    item_index,
                    verifier,
                    backend,
                )
                histories.append(history)
                terminals.append(terminal)
                seq = item_index if profile == "background_sequence" else None
                cohorts.append(
                    ExportCohort(
                        profile,
                        history.current_report,
                        (ExportItem(history, terminal, seq),),
                    )
                )
        export_request = ExportRequest(
            tuple(cohorts),
            environment,
            _build_credits(
                compiled, source, _production_graph(compiled, plan.output_profiles[0])
            ),
        )
        _append(
            cursor,
            job_store,
            plan.job_id,
            EventType.EXPORT_STARTED,
            JobStatus.EXPORTING,
            ExportStartedPayload(bundle_name),
        )
        bundle = export_bundle(export_request, root_fd, bundle_name)
        _append(
            cursor,
            job_store,
            plan.job_id,
            EventType.EXPORT_PUBLISHED,
            JobStatus.COMPLETED,
            ExportPublishedPayload(
                bundle.bundle_name,
                bundle.manifest_sha256,
                bundle.payload_sha256,
                bundle.bundle_sha256,
            ),
        )
        return FakeJobResult(
            bundle, tuple(cohorts), tuple(terminals), tuple(histories), "COMPLETED"
        )
    except SpecStyleError:
        _try_fatal(job_store, plan.job_id, cursor)
        return FakeJobResult(None, (), (), (), "JOB_FAILED")


def _budget(compiled: CompiledStyleSpec) -> JobBudget:
    return JobBudget(1 + compiled.source_spec.repair.max_rounds)
