"""Private production composition for the initial generation attempt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from specstyle.domain.artifacts import AssetRef
from specstyle.domain.enums import ArtifactStatus
from specstyle.domain.identifiers import AttemptId, JobId
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.diffusers_backend import DiffusersBackend, StyleAssetResolver
from specstyle.generation.diffusers_loader import (
    LoadedPipeline,
    load_production_pipeline,
)
from specstyle.generation.model_approval import VerifiedPipelineSupply
from specstyle.generation.pipeline_factory import PipelineGraph
from specstyle.generation.preprocess import PreparedImage
from specstyle.generation.protocols import (
    ControlInputBuilder,
    GeneratedArtifact,
    build_control_input,
    run_generation,
)
from specstyle.generation.requests import GenerationRequest, RenderedPrompt
from specstyle.observability.environment import (
    EnvironmentSnapshot,
    hash_environment,
)
from specstyle.repair.history import RepairHistory
from specstyle.repair.loop import NextGeneration, RepairTerminal
from specstyle.spec.compiled_models import (
    CompiledExecutionGraph,
    CompiledStyleSpec,
    CompiledVerificationPlan,
    CompilerContext,
    OutputProfile,
)
from specstyle.spec.compiler import compile_style_spec
from specstyle.spec.loader import load_style_spec_text
from specstyle.verification.production import (
    _ProductionVerificationAllowlist,
    _create_production_verifier_factory,
)
from specstyle.verification.protocols import Verifier, run_verifier
from specstyle.verification.rule_models import VerificationReport
from specstyle.workflow.job_models import (
    AttemptFinishedPayload,
    AttemptStartedPayload,
    Event,
    EventType,
    FatalPayload,
    Job,
    JobBudget,
    JobSnapshot,
    JobStartedPayload,
    JobState,
    JobStatus,
    RepairStepPayload,
    SpecCompiledPayload,
    VerifierFinishedPayload,
    _bundle_name,
    _timestamp,
)
from specstyle.workflow.job_store import JobStore
from specstyle.workflow.production_artifacts import _open_production_artifact_store
from specstyle.workflow.production_repair import (
    _compose_initial_repair,
    _compose_repair_result,
    _repair_ids,
    _validate_repair_contract,
)
from specstyle.workflow.production_reports import _open_production_report_store

__all__ = ("ProductionJobRequest", "ProductionJobResult")

_OUTPUT_PROFILES = {"xhs_grid", "talking_head_cover", "background_sequence"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _NondecreasingAuditClock:
    __slots__ = ("_source", "_last")

    def __init__(self, source: Callable[[], str], /) -> None:
        self._source = source
        self._last: str | None = None

    def __call__(self) -> str:
        current = self._source()
        try:
            current = _timestamp(current)
        except DomainError:
            return current
        if self._last is None or current >= self._last:
            self._last = current
        return self._last


@dataclass(frozen=True, slots=True)
class ProductionJobRequest:
    """Validated input for one production generation attempt."""

    job_id: JobId
    spec_text: str
    source: PreparedImage
    style_references: tuple[AssetRef, ...]
    prompt: RenderedPrompt
    output_profile: OutputProfile
    variation_index: int
    bundle_name: str

    def __post_init__(self) -> None:
        if (
            type(self.job_id) is not JobId
            or type(self.spec_text) is not str
            or type(self.source) is not PreparedImage
            or type(self.style_references) is not tuple
            or not self.style_references
            or any(
                type(reference) is not AssetRef for reference in self.style_references
            )
            or type(self.prompt) is not RenderedPrompt
            or type(self.output_profile) is not str
            or self.output_profile not in _OUTPUT_PROFILES
            or type(self.variation_index) is not int
            or self.variation_index < 0
            or type(self.bundle_name) is not str
        ):
            raise DomainError("invalid production job request")
        _bundle_name(self.bundle_name)
        AttemptId(f"{self.job_id.value}-a0-{self.output_profile}-0")
        _repair_ids(self.job_id, self.output_profile)


@dataclass(frozen=True, slots=True)
class ProductionJobResult:
    compiled: CompiledStyleSpec
    graph: CompiledExecutionGraph
    verification_plan: CompiledVerificationPlan
    request: GenerationRequest
    artifact: GeneratedArtifact
    report: VerificationReport
    history: RepairHistory
    terminal: RepairTerminal
    job_state: JobState


@dataclass(frozen=True, slots=True)
class _PreparedInitialAttempt:
    compiled: CompiledStyleSpec
    graph: CompiledExecutionGraph
    verification_plan: CompiledVerificationPlan
    request: GenerationRequest
    repository: Any
    report_repository: Any
    verifier: Verifier


def _select_initial_contract(
    request: ProductionJobRequest, compiler_context: CompilerContext
) -> tuple[CompiledStyleSpec, CompiledExecutionGraph, CompiledVerificationPlan]:
    compiled = compile_style_spec(
        load_style_spec_text(request.spec_text), compiler_context
    )
    graphs = tuple(
        graph
        for graph in compiled.production_graphs
        if graph.output_profile == request.output_profile
    )
    plans = tuple(
        plan
        for plan in compiled.verification_plans
        if plan.output_profile == request.output_profile
    )
    if len(graphs) != 1 or len(plans) != 1:
        raise DomainError("production selectors must resolve exactly once")
    _validate_repair_contract(compiled, request.output_profile)
    return compiled, graphs[0], plans[0]


def _matches_descriptor(resolved: object, descriptor: object) -> bool:
    pin = getattr(resolved, "pin", None)
    return (
        getattr(pin, "id", None) == getattr(descriptor, "model_id", None)
        and getattr(pin, "revision", None) == getattr(descriptor, "revision", None)
        and getattr(pin, "sha256", None) == getattr(descriptor, "expected_sha256", None)
    )


def _preflight_bindings(
    request: ProductionJobRequest,
    graph: CompiledExecutionGraph,
    loaded: LoadedPipeline,
) -> None:
    loaded_graph = loaded._graph
    runtime = graph.runtime
    if (
        graph.generation_profile != "production"
        or graph.pipeline != "sdxl_base"
        or graph.scheduler != "euler"
        or graph.controlnet.controlnet_type != "canny"
        or runtime.backend != "rocm"
        or runtime.dtype != "float16"
        or (
            runtime.rocm_version,
            runtime.torch_version,
            runtime.diffusers_version,
            runtime.dtype,
        )
        != loaded._runtime
        or not _matches_descriptor(graph.base_model, loaded_graph.base)
        or not _matches_descriptor(graph.ip_adapter, loaded_graph.ip_adapter)
        or not _matches_descriptor(graph.controlnet, loaded_graph.controlnet)
        or tuple(reference.sha256 for reference in request.style_references)
        != graph.style_reference_hashes
    ):
        raise DomainError("production preflight binding mismatch")


def _initial_generation_request(
    request: ProductionJobRequest,
    compiled: CompiledStyleSpec,
    graph: CompiledExecutionGraph,
    control_builder: ControlInputBuilder,
    environment: EnvironmentSnapshot,
) -> GenerationRequest:
    control = build_control_input(control_builder, request.source, graph)
    return GenerationRequest(
        request.job_id,
        AttemptId(f"{request.job_id.value}-a0-{request.output_profile}-0"),
        None,
        compiled,
        "production",
        request.output_profile,
        request.source,
        request.style_references,
        request.prompt,
        control,
        request.variation_index,
        hash_environment(environment),
    )


def _append_event(
    store: JobStore,
    job_id: JobId,
    event_type: EventType,
    from_state: JobStatus,
    to_state: JobStatus,
    timestamp: str,
    payload: object,
) -> None:
    store.append_event(
        job_id,
        Event(1, job_id, event_type, from_state, to_state, timestamp, payload),
    )


def _create_initial_job(
    store: JobStore,
    request: ProductionJobRequest,
    compiled: CompiledStyleSpec,
    clock: Callable[[], str],
) -> JobBudget:
    timestamp = clock()
    budget = JobBudget(1 + compiled.source_spec.repair.max_rounds)
    job = Job(
        request.job_id,
        compiled.compiled_spec_hash,
        (request.output_profile,),
        budget,
        JobStatus.CREATED,
        timestamp,
        timestamp,
    )
    store.save_snapshot(
        request.job_id,
        JobSnapshot("specstyle.workflow.snapshot.v1", job, 0, (), ()),
    )
    return budget


def _reject_duplicate_job(store: JobStore, job_id: JobId) -> None:
    if store.get_snapshot(job_id) is not None:
        raise DomainError("duplicate production job id")


def _record_attempt_start(
    store: JobStore,
    request: ProductionJobRequest,
    compiled: CompiledStyleSpec,
    generation_request: GenerationRequest,
    budget: JobBudget,
    clock: Callable[[], str],
) -> None:
    transitions = (
        (
            EventType.JOB_STARTED,
            JobStatus.CREATED,
            JobStatus.SPEC_VALIDATED,
            JobStartedPayload(
                compiled.compiled_spec_hash, (request.output_profile,), budget
            ),
        ),
        (
            EventType.SPEC_COMPILED,
            JobStatus.SPEC_VALIDATED,
            JobStatus.SPEC_COMPILED,
            SpecCompiledPayload(compiled.compiled_spec_hash),
        ),
        (
            EventType.ATTEMPT_STARTED,
            JobStatus.SPEC_COMPILED,
            JobStatus.GENERATING,
            AttemptStartedPayload(0, 0, generation_request.attempt_id, None),
        ),
    )
    for event_type, from_state, to_state, payload in transitions:
        _append_event(
            store, request.job_id, event_type, from_state, to_state, clock(), payload
        )


def _run_initial_generation(
    loaded: LoadedPipeline,
    style_assets: StyleAssetResolver,
    store: JobStore,
    request: GenerationRequest,
    clock: Callable[[], str],
    from_state: JobStatus = JobStatus.GENERATING,
) -> GeneratedArtifact:
    try:
        return run_generation(DiffusersBackend(loaded, style_assets), request)
    except (DomainError, InfrastructureError) as error:
        oom = type(error) is InfrastructureError and error.args == ("generation OOM",)
        try:
            _append_event(
                store,
                request.job_id,
                EventType.FATAL,
                from_state,
                JobStatus.JOB_FAILED,
                clock(),
                FatalPayload(
                    "GENERATION_OOM" if oom else "GENERATION_FAILED",
                    "generation OOM" if oom else "generation failed",
                ),
            )
        except Exception:
            pass
        raise


def _record_attempt_finish(
    store: JobStore,
    request: GenerationRequest,
    artifact: GeneratedArtifact,
    timestamp: str,
    from_state: JobStatus = JobStatus.GENERATING,
) -> None:
    _append_event(
        store,
        request.job_id,
        EventType.ATTEMPT_FINISHED,
        from_state,
        JobStatus.VERIFYING,
        timestamp,
        AttemptFinishedPayload(
            0,
            0,
            request.attempt_id,
            artifact.ref.artifact_id,
            request.request_hash,
        ),
    )


def _record_verifier_fatal(
    store: JobStore, job_id: JobId, clock: Callable[[], str]
) -> None:
    try:
        _append_event(
            store,
            job_id,
            EventType.FATAL,
            JobStatus.VERIFYING,
            JobStatus.JOB_FAILED,
            clock(),
            FatalPayload("VERIFIER_UNAVAILABLE", "verifier unavailable"),
        )
    except Exception:
        pass


def _run_initial_verification(
    verifier: Verifier,
    store: JobStore,
    request: GenerationRequest,
    artifact: GeneratedArtifact,
    plan: CompiledVerificationPlan,
    clock: Callable[[], str],
) -> VerificationReport:
    artifacts = (artifact.ref,)
    rules = plan.applicable_rule_definitions
    failed = False
    try:
        results = run_verifier(verifier, artifacts, rules)
        report = VerificationReport(artifacts, rules, results)
    except Exception:
        failed = True
    if failed:
        _record_verifier_fatal(store, request.job_id, clock)
        raise InfrastructureError("verifier unavailable") from None
    return report


def _record_unknown_fatal(
    store: JobStore,
    job_id: JobId,
    from_state: JobStatus,
    clock: Callable[[], str],
    reason: str,
) -> None:
    try:
        _append_event(
            store,
            job_id,
            EventType.FATAL,
            from_state,
            JobStatus.JOB_FAILED,
            clock(),
            FatalPayload("UNKNOWN", reason),
        )
    except Exception:
        pass


def _persist_artifact(
    repository: Any,
    artifact: GeneratedArtifact,
    store: JobStore,
    request: GenerationRequest,
    state: JobStatus,
    clock: Callable[[], str],
) -> None:
    failed = False
    try:
        repository.put(artifact)
        if repository(artifact.ref) != artifact:
            raise InfrastructureError("artifact readback mismatch")
    except Exception:
        failed = True
    if failed:
        _record_unknown_fatal(
            store, request.job_id, state, clock, "artifact persistence failed"
        )
        raise InfrastructureError("artifact persistence failed") from None


def _persist_report(
    repository: Any,
    request: GenerationRequest,
    report: VerificationReport,
    store: JobStore,
    clock: Callable[[], str],
) -> None:
    failed = False
    try:
        repository.put(request, report)
        if repository() != report:
            raise InfrastructureError("report readback mismatch")
    except Exception:
        failed = True
    if failed:
        reason = "verification report persistence failed"
        _record_unknown_fatal(store, request.job_id, JobStatus.VERIFYING, clock, reason)
        raise InfrastructureError(reason) from None


def _decision_state(status: ArtifactStatus) -> JobStatus:
    return JobStatus(status.value)


def _record_verifier_decision(
    store: JobStore,
    job_id: JobId,
    artifact: GeneratedArtifact,
    decision: Any,
    from_state: JobStatus,
    to_state: JobStatus,
    clock: Callable[[], str],
) -> None:
    _append_event(
        store,
        job_id,
        EventType.VERIFIER_FINISHED,
        from_state,
        to_state,
        clock(),
        VerifierFinishedPayload(
            0,
            0,
            artifact.ref.artifact_id,
            decision.artifact_status,
            decision.decision_reason,
            decision.repair_stop_reason,
        ),
    )


def _repair_call(
    operation: Callable[[], Any],
    store: JobStore,
    job_id: JobId,
    state: JobStatus,
    clock: Callable[[], str],
) -> Any:
    failed = False
    result: Any = None
    try:
        result = operation()
    except Exception:
        failed = True
    if failed:
        reason = "repair composition failed"
        _record_unknown_fatal(store, job_id, state, clock, reason)
        raise InfrastructureError(reason) from None
    return result


def _close_repositories(repositories: tuple[Any, ...], *, quiet: bool) -> None:
    first: Exception | None = None
    for repository in repositories:
        try:
            repository.close()
        except Exception as error:
            if first is None:
                first = error
    if first is not None and not quiet:
        raise first


class _ProductionGenerationRuntime:
    __slots__ = (
        "_loaded",
        "_verifier_factory",
        "_report_store",
        "_artifact_store",
        "_environment",
        "_compiler_context",
        "_style_assets",
        "_control_builder",
        "_job_store",
        "_clock",
        "_closed",
    )

    def __init__(
        self,
        loaded: LoadedPipeline,
        verifier_factory: Any,
        report_store: Any,
        artifact_store: Any,
        environment: EnvironmentSnapshot,
        compiler_context: CompilerContext,
        style_assets: StyleAssetResolver,
        control_builder: ControlInputBuilder,
        job_store: JobStore,
        clock: Callable[[], str],
        /,
    ) -> None:
        self._loaded = loaded
        self._verifier_factory = verifier_factory
        self._report_store = report_store
        self._artifact_store = artifact_store
        self._environment = environment
        self._compiler_context = compiler_context
        self._style_assets = style_assets
        self._control_builder = control_builder
        self._job_store = job_store
        self._clock = _NondecreasingAuditClock(clock)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        factory, self._verifier_factory = self._verifier_factory, None
        cleanup = (
            getattr(factory, "close", None),
            self._loaded.close,
            self._report_store.close,
            self._artifact_store.close,
        )
        first_error: Exception | None = None
        for close in cleanup:
            if not callable(close):
                continue
            try:
                close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _open_attempt(
        self, request: GenerationRequest, plan: CompiledVerificationPlan
    ) -> tuple[Any, Any, Verifier]:
        artifact_repository = self._artifact_store.for_job(request.job_id)
        try:
            report_repository = self._report_store.for_attempt(
                request.job_id, request.attempt_id
            )
        except Exception:
            _close_repositories((artifact_repository,), quiet=True)
            raise
        try:
            verifier = self._verifier_factory.create(
                request, plan, artifact_repository, self._style_assets
            )
        except Exception:
            _close_repositories((report_repository, artifact_repository), quiet=True)
            raise
        return artifact_repository, report_repository, verifier

    def _prepare_initial_attempt(
        self, request: ProductionJobRequest
    ) -> _PreparedInitialAttempt:
        compiled, graph, verification_plan = _select_initial_contract(
            request, self._compiler_context
        )
        _preflight_bindings(request, graph, self._loaded)
        _reject_duplicate_job(self._job_store, request.job_id)
        generation_request = _initial_generation_request(
            request,
            compiled,
            graph,
            self._control_builder,
            self._environment,
        )
        repository, report_repository, verifier = self._open_attempt(
            generation_request, verification_plan
        )
        return _PreparedInitialAttempt(
            compiled,
            graph,
            verification_plan,
            generation_request,
            repository,
            report_repository,
            verifier,
        )

    def _run_prepared_attempt(
        self, request: ProductionJobRequest, prepared: _PreparedInitialAttempt
    ) -> ProductionJobResult:
        budget = _create_initial_job(
            self._job_store, request, prepared.compiled, self._clock
        )
        _record_attempt_start(
            self._job_store,
            request,
            prepared.compiled,
            prepared.request,
            budget,
            self._clock,
        )
        artifact, report = self._run_attempt(
            prepared.request,
            prepared.verification_plan,
            prepared.repository,
            prepared.report_repository,
            prepared.verifier,
            JobStatus.GENERATING,
        )
        composed = _repair_call(
            lambda: _compose_initial_repair(prepared.request, artifact, report),
            self._job_store,
            request.job_id,
            JobStatus.VERIFYING,
            self._clock,
        )
        history, step = composed.history, composed.step
        if composed.selecting_decision is not None:
            _record_verifier_decision(
                self._job_store,
                request.job_id,
                artifact,
                composed.selecting_decision,
                JobStatus.VERIFYING,
                JobStatus.REPAIR_SELECTING,
                self._clock,
            )
            if type(step) is RepairTerminal:
                self._record_terminal(
                    request.job_id, artifact, step, JobStatus.REPAIR_SELECTING
                )
                return self._result(prepared, history, step)
            return self._run_repair(prepared, history, step)
        self._record_terminal(request.job_id, artifact, step, JobStatus.VERIFYING)
        return self._result(prepared, history, step)

    def _run_attempt(
        self, request, plan, artifact_repository, report_repository, verifier, state
    ) -> tuple[GeneratedArtifact, VerificationReport]:
        artifact = _run_initial_generation(
            self._loaded,
            self._style_assets,
            self._job_store,
            request,
            self._clock,
            state,
        )
        _persist_artifact(
            artifact_repository, artifact, self._job_store, request, state, self._clock
        )
        _record_attempt_finish(self._job_store, request, artifact, self._clock(), state)
        report = _run_initial_verification(
            verifier, self._job_store, request, artifact, plan, self._clock
        )
        _persist_report(
            report_repository, request, report, self._job_store, self._clock
        )
        return artifact, report

    def _record_terminal(self, job_id, artifact, terminal, from_state) -> None:
        if type(terminal) is not RepairTerminal:
            _repair_call(
                lambda: (_ for _ in ()).throw(DomainError("invalid terminal")),
                self._job_store,
                job_id,
                from_state,
                self._clock,
            )
        decision = terminal.artifact_decision
        _record_verifier_decision(
            self._job_store,
            job_id,
            artifact,
            decision,
            from_state,
            _decision_state(decision.artifact_status),
            self._clock,
        )

    def _run_repair(self, prepared, history, command):
        resources = _repair_call(
            lambda: self._open_attempt(command.request, prepared.verification_plan),
            self._job_store,
            command.request.job_id,
            JobStatus.REPAIR_SELECTING,
            self._clock,
        )
        repository, report_repository, verifier = resources
        try:
            self._record_repair_step(command)
            artifact, report = self._run_attempt(
                command.request,
                prepared.verification_plan,
                repository,
                report_repository,
                verifier,
                JobStatus.REPAIRING,
            )
        except Exception:
            _close_repositories((report_repository, repository), quiet=True)
            raise
        _close_repositories((report_repository, repository), quiet=False)
        composed = _repair_call(
            lambda: _compose_repair_result(history, command, artifact, report),
            self._job_store,
            command.request.job_id,
            JobStatus.VERIFYING,
            self._clock,
        )
        history, terminal = composed.history, composed.terminal
        self._record_terminal(
            command.request.job_id, artifact, terminal, JobStatus.VERIFYING
        )
        return self._result(prepared, history, terminal)

    def _record_repair_step(self, command: NextGeneration) -> None:
        decision, request = command.decision, command.request
        _append_event(
            self._job_store,
            request.job_id,
            EventType.REPAIR_STEP,
            JobStatus.REPAIR_SELECTING,
            JobStatus.REPAIRING,
            self._clock(),
            RepairStepPayload(
                0,
                0,
                decision.decision_id,
                decision.action_id,
                decision.trigger_rule_id,
                request.parent_attempt_id,
                request.attempt_id,
            ),
        )

    def _result(self, prepared, history, terminal) -> ProductionJobResult:
        return ProductionJobResult(
            prepared.compiled,
            prepared.graph,
            prepared.verification_plan,
            history.current_request,
            history.current_artifact,
            history.current_report,
            history,
            terminal,
            self._job_store.load(history.current_request.job_id),
        )

    def _execute_initial_attempt(
        self, request: ProductionJobRequest, /
    ) -> ProductionJobResult:
        if self._closed:
            raise InfrastructureError("production runtime closed")
        if type(request) is not ProductionJobRequest:
            raise DomainError("invalid production job request")
        prepared = self._prepare_initial_attempt(request)
        try:
            result = self._run_prepared_attempt(request, prepared)
        except Exception:
            _close_repositories(
                (prepared.report_repository, prepared.repository), quiet=True
            )
            raise
        _close_repositories(
            (prepared.report_repository, prepared.repository), quiet=False
        )
        return result


def _cleanup_failed_open(
    loaded: LoadedPipeline,
    factory: Any | None,
    report_store: Any | None,
    artifact_store: Any | None,
) -> None:
    for resource in (factory, loaded, report_store, artifact_store):
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            pass


def _validate_runtime_dependencies(
    compiler_context: object,
    style_assets: object,
    control_builder: object,
    allowlist: object,
    job_store: object,
    clock: object,
) -> None:
    if (
        type(compiler_context) is not CompilerContext
        or not callable(style_assets)
        or not callable(getattr(control_builder, "build", None))
        or type(allowlist) is not _ProductionVerificationAllowlist
        or type(job_store) is not JobStore
        or not callable(clock)
    ):
        raise DomainError("invalid production runtime dependency")


def _open_production_generation_runtime(
    supply: VerifiedPipelineSupply,
    pipeline_graph: PipelineGraph,
    environment: EnvironmentSnapshot,
    compiler_context: CompilerContext,
    style_assets: StyleAssetResolver,
    control_builder: ControlInputBuilder,
    allowlist: _ProductionVerificationAllowlist,
    job_store: JobStore,
    artifact_root_fd: int,
    /,
    *,
    torch_module: Any | None = None,
    diffusers_module: Any | None = None,
    clock: Callable[[], str] = _utc_now,
) -> _ProductionGenerationRuntime:
    _validate_runtime_dependencies(
        compiler_context, style_assets, control_builder, allowlist, job_store, clock
    )
    loaded = load_production_pipeline(
        supply,
        pipeline_graph,
        environment,
        torch_module=torch_module,
        diffusers_module=diffusers_module,
    )
    artifact_store = None
    report_store = None
    verifier_factory = None
    try:
        verifier_factory = _create_production_verifier_factory(loaded, allowlist)
        artifact_store = _open_production_artifact_store(artifact_root_fd)
        report_store = _open_production_report_store(artifact_root_fd)
        return _ProductionGenerationRuntime(
            loaded,
            verifier_factory,
            report_store,
            artifact_store,
            environment,
            compiler_context,
            style_assets,
            control_builder,
            job_store,
            clock,
        )
    except Exception:
        _cleanup_failed_open(loaded, verifier_factory, report_store, artifact_store)
        raise
