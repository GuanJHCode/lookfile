"""APP-COMPOSE-001E production export command contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
import importlib

import pytest

from specstyle.domain.artifacts import ArtifactRef, AssetRef
from specstyle.domain.enums import (
    ArtifactStatus,
    RepairStopReason,
    RuleScope,
    RuleStatus,
)
from specstyle.domain.identifiers import (
    ArtifactId,
    AssetId,
    AttemptId,
    JobId,
    Sha256,
)
from specstyle.errors import DomainError
from specstyle.exporting.bundle import ExportBundle, ExportedFile
from specstyle.exporting.manifest import (
    AssetCredit,
    ExportCohort,
    ExportItem,
    ExportRequest,
    _prepare_export,
)
from specstyle.generation.protocols import GeneratedArtifact
from specstyle.generation.requests import (
    GenerationRequest,
    PreparedControlInput,
    RenderedPrompt,
)
from specstyle.observability.environment import TextObservation, hash_environment
from specstyle.observability.hashing import hash_bytes
from specstyle.reliability.fixtures import (
    sample_approved_export_request,
    sample_compiler_context,
    sample_environment,
    sample_source,
    sample_style_spec,
)
from specstyle.repair.history import start_repair_history
from specstyle.repair.loop import RepairTerminal
from specstyle.spec.compiled_models import ResourcePin
from specstyle.spec.compiler import compile_style_spec
from specstyle.spec.models import StyleSpecV1
from specstyle.verification.routing import decide_artifact
from specstyle.verification.rule_models import RuleResult, VerificationReport
from specstyle.workflow.job_models import Job, JobBudget, JobState, JobStatus
from specstyle.workflow.production_service import (
    ProductionJobRequest,
    ProductionJobResult,
    _ProductionGenerationRuntime,
)


_PROFILES = ("xhs_grid", "talking_head_cover", "background_sequence")
_DECISIONS = (
    (ArtifactStatus.APPROVED, RuleStatus.PASS, RepairStopReason.PASS_ALL_REQUIRED),
    (ArtifactStatus.MANUAL_REVIEW, RuleStatus.WARNING, RepairStopReason.MANUAL_REQUEST),
    (ArtifactStatus.REJECTED, RuleStatus.FAIL, RepairStopReason.NO_IMPROVEMENT),
)


class _HostileStr(str):
    pass


class _HostileInt(int):
    pass


class _HostileJobId(JobId):
    pass


def _export_module():
    return importlib.import_module("specstyle.workflow.production_export")


@dataclass(frozen=True)
class _Case:
    request: ProductionJobRequest
    result: ProductionJobResult
    environment: object
    compiler_context: object
    credits: tuple[AssetCredit, ...]


def _compiled(profile: str):
    raw = sample_style_spec().model_dump(mode="python")
    raw["repair"]["policy_version"] = "1.0"
    raw["outputs"]["profiles"] = (profile,)
    context = sample_compiler_context()
    output = replace(context.output_profile_capabilities[0], profile=profile)
    alternate = next(item for item in _PROFILES if item != profile)
    rules = tuple(
        replace(
            rule,
            supported_output_profiles=(
                (alternate,) if rule.scope is RuleScope.BATCH else (profile,)
            ),
        )
        for rule in context.rule_catalogs[0].rules
    )
    context = replace(
        context,
        output_profile_capabilities=(output,),
        rule_catalogs=(replace(context.rule_catalogs[0], rules=rules),),
        threshold_profiles=(
            replace(
                context.threshold_profiles[0],
                metrics=tuple(
                    metric
                    for metric in context.threshold_profiles[0].metrics
                    if metric.metric_id.value != "batch-metric"
                ),
            ),
        ),
    )
    spec = StyleSpecV1.model_validate(raw)
    return compile_style_spec(spec, context), spec.model_dump_json(), context


def _case(
    profile: str = "xhs_grid",
    status: ArtifactStatus = ArtifactStatus.APPROVED,
    *,
    job_id: str = "job",
) -> _Case:
    compiled, spec_text, compiler_context = _compiled(profile)
    graph = compiled.production_graphs[0]
    plan = compiled.verification_plans[0]
    environment = sample_environment()
    source = sample_source()
    prompt = RenderedPrompt(
        ResourcePin("template", "r1", Sha256("e" * 64)),
        graph.preset_id,
        "a prompt",
        "",
    )
    generation_request = GenerationRequest(
        JobId(job_id),
        AttemptId(f"{job_id}-a0-{profile}-0"),
        None,
        compiled,
        "production",
        profile,
        source,
        (AssetRef(AssetId("style"), graph.style_reference_hashes[0]),),
        prompt,
        PreparedControlInput("canny", source),
        0,
        hash_environment(environment),
    )
    artifact = GeneratedArtifact(
        ArtifactRef(ArtifactId(f"artifact-{profile}"), hash_bytes(source.content)),
        source.content,
        generation_request.request_hash,
        generation_request.generation_fingerprint,
    )
    rule_status, stop = next(
        (rule_status, stop)
        for artifact_status, rule_status, stop in _DECISIONS
        if artifact_status is status
    )
    report = VerificationReport(
        (artifact.ref,),
        plan.applicable_rule_definitions,
        tuple(
            RuleResult(
                rule.rule_id,
                rule_status,
                (artifact.ref.artifact_id,),
                None,
            )
            for rule in plan.applicable_rule_definitions
        ),
    )
    history = start_repair_history(generation_request, artifact, report)
    terminal = RepairTerminal(
        decide_artifact(
            report,
            artifact.ref.artifact_id,
            repair_stop_reason=stop,
        ),
        None,
    )
    job = Job(
        generation_request.job_id,
        compiled.compiled_spec_hash,
        (profile,),
        JobBudget(2),
        JobStatus(status.value),
        "2026-08-02T00:00:00.000Z",
        "2026-08-02T00:00:01.000Z",
    )
    result = ProductionJobResult(
        compiled,
        graph,
        plan,
        generation_request,
        artifact,
        report,
        history,
        terminal,
        JobState(job, 5, (generation_request.attempt_id,), ()),
    )
    style = compiled.source_spec.assets.style_references[0]
    credits = tuple(
        sorted(
            (
                AssetCredit(source.source, ("input",), None, None, None, None),
                AssetCredit(
                    generation_request.style_references[0],
                    ("style_reference",),
                    style.source_url,
                    style.license,
                    style.attribution,
                    style.consent,
                ),
            ),
            key=lambda credit: credit.identity,
        )
    )
    request = ProductionJobRequest(
        generation_request.job_id,
        spec_text,
        source,
        generation_request.style_references,
        prompt,
        profile,
        0,
        "bundle",
    )
    return _Case(request, result, environment, compiler_context, credits)


def _runtime(case: _Case) -> _ProductionGenerationRuntime:
    runtime = object.__new__(_ProductionGenerationRuntime)
    runtime._environment = case.environment
    runtime._compiler_context = case.compiler_context
    return runtime


def _completed_state(bundle_name: str = "bundle") -> JobState:
    export_request = sample_approved_export_request()
    item = export_request.cohorts[0].items[0]
    request = item.history.current_request
    job = Job(
        request.job_id,
        request.compiled_spec.compiled_spec_hash,
        (request.output_profile,),
        JobBudget(2),
        JobStatus.COMPLETED,
        "2026-08-02T00:00:00.000Z",
        "2026-08-02T00:00:01.000Z",
    )
    return JobState(job, 6, (request.attempt_id,), (bundle_name,))


def _bundle(bundle_name: str = "bundle") -> ExportBundle:
    return ExportBundle(
        bundle_name,
        1,
        2,
        Sha256("1" * 64),
        Sha256("2" * 64),
        Sha256("3" * 64),
        (ExportedFile("manifest.json", Sha256("1" * 64), 1),),
    )


def test_public_export_contracts_are_frozen_slotted_and_exactly_exported() -> None:
    module = _export_module()
    export_request = sample_approved_export_request()
    command = module.ProductionExportCommand(JobId("job"), "bundle", export_request)
    result = module.ProductionExportResult(_bundle(), _completed_state())
    entry = module.ProductionRecoveryEntry(
        JobId("job"), module.ProductionRecoveryDisposition.RECOVERED, result
    )

    assert module.__all__ == (
        "ProductionRecoveryDisposition",
        "ProductionExportCommand",
        "ProductionExportResult",
        "ProductionRecoveryEntry",
    )
    assert tuple(module.ProductionRecoveryDisposition) == tuple(
        module.ProductionRecoveryDisposition(value)
        for value in (
            "RECOVERED",
            "ALREADY_COMPLETED",
            "SKIPPED_MISSING_COMMAND",
            "SKIPPED_NOT_EXPORTABLE",
            "SKIPPED_TERMINAL",
        )
    )
    assert tuple(field.name for field in fields(command)) == (
        "job_id",
        "bundle_name",
        "export_request",
    )
    assert tuple(field.name for field in fields(result)) == ("bundle", "job_state")
    assert tuple(field.name for field in fields(entry)) == (
        "job_id",
        "disposition",
        "result",
    )
    assert not hasattr(command, "__dict__")
    with pytest.raises(FrozenInstanceError):
        command.bundle_name = "other"
    with pytest.raises(ValueError):
        module.ProductionRecoveryDisposition("recovered")


@pytest.mark.parametrize("disposition_name", ("RECOVERED", "ALREADY_COMPLETED"))
def test_recovery_success_entries_require_a_job_bound_result(
    disposition_name: str,
) -> None:
    module = _export_module()
    result = module.ProductionExportResult(_bundle(), _completed_state())
    disposition = module.ProductionRecoveryDisposition[disposition_name]

    assert (
        module.ProductionRecoveryEntry(JobId("job"), disposition, result).result
        is not result
    )
    with pytest.raises(DomainError):
        module.ProductionRecoveryEntry(JobId("other"), disposition, result)
    with pytest.raises(DomainError):
        module.ProductionRecoveryEntry(JobId("job"), disposition, None)


@pytest.mark.parametrize(
    "disposition_name",
    (
        "SKIPPED_MISSING_COMMAND",
        "SKIPPED_NOT_EXPORTABLE",
        "SKIPPED_TERMINAL",
    ),
)
def test_recovery_skips_forbid_results(disposition_name: str) -> None:
    module = _export_module()
    disposition = module.ProductionRecoveryDisposition[disposition_name]

    assert (
        module.ProductionRecoveryEntry(JobId("job"), disposition, None).result is None
    )
    with pytest.raises(DomainError):
        module.ProductionRecoveryEntry(
            JobId("job"),
            disposition,
            module.ProductionExportResult(_bundle(), _completed_state()),
        )


def test_export_command_and_result_rebuild_values_and_reject_bad_bindings() -> None:
    module = _export_module()
    case = _case()
    export_request = ExportRequest(
        (
            ExportCohort(
                "xhs_grid",
                case.result.report,
                (ExportItem(case.result.history, case.result.terminal, None),),
            ),
        ),
        case.environment,
        case.credits,
    )

    command = module.ProductionExportCommand(JobId("job"), "bundle", export_request)
    bundle = _bundle()
    completed = module.ProductionExportResult(bundle, _completed_state())

    assert command.export_request is not export_request
    assert completed.bundle is not bundle
    with pytest.raises(DomainError):
        module.ProductionExportCommand(JobId("other"), "bundle", export_request)
    with pytest.raises(DomainError):
        module.ProductionExportCommand(JobId("job"), "../bundle", export_request)
    with pytest.raises(DomainError):
        module.ProductionExportResult(_bundle("other"), _completed_state("bundle"))
    with pytest.raises(DomainError):
        module.ProductionExportResult(_bundle(), _case().result.job_state)


@pytest.mark.parametrize("profile", _PROFILES)
@pytest.mark.parametrize("status", tuple(case[0] for case in _DECISIONS))
def test_prepare_export_supports_every_profile_and_terminal_route(
    profile: str, status: ArtifactStatus
) -> None:
    case = _case(profile, status)

    command = _runtime(case).prepare_export(case.request, case.result, case.credits)
    prepared = _prepare_export(command.export_request)

    assert type(command) is _export_module().ProductionExportCommand
    assert command.job_id == case.request.job_id
    assert command.bundle_name == case.request.bundle_name
    assert len(command.export_request.cohorts) == 1
    cohort = command.export_request.cohorts[0]
    assert cohort.output_profile == profile
    assert len(cohort.items) == 1
    assert cohort.items[0].sequence_index == (
        0 if profile == "background_sequence" else None
    )
    routed = tuple(
        item.relative_path
        for item in prepared.payload_files
        if item.relative_path.endswith(".png")
    )
    assert len(routed) == 1
    assert routed[0].startswith(
        {
            ArtifactStatus.APPROVED: f"approved/{profile}/",
            ArtifactStatus.MANUAL_REVIEW: "manual_review/",
            ArtifactStatus.REJECTED: "rejected/",
        }[status]
    )


def test_prepare_export_rejects_non_exact_arguments_and_credit_contracts() -> None:
    case = _case()
    runtime = _runtime(case)

    with pytest.raises(DomainError):
        runtime.prepare_export(object(), case.result, case.credits)
    with pytest.raises(DomainError):
        runtime.prepare_export(case.request, object(), case.credits)
    with pytest.raises(DomainError):
        runtime.prepare_export(case.request, case.result, list(case.credits))
    with pytest.raises(DomainError):
        runtime.prepare_export(case.request, case.result, tuple(reversed(case.credits)))
    with pytest.raises(DomainError):
        runtime.prepare_export(case.request, case.result, case.credits[:-1])
    style = case.credits[-1]
    incomplete = AssetCredit(
        style.asset,
        style.roles,
        style.source_url,
        None,
        style.attribution,
        style.consent,
    )
    with pytest.raises(DomainError):
        runtime.prepare_export(case.request, case.result, (case.credits[0], incomplete))


def test_prepare_export_rejects_cross_job_profile_and_environment() -> None:
    case = _case()
    other_job = _case(job_id="other")
    other_profile = _case("talking_head_cover")

    with pytest.raises(DomainError):
        _runtime(case).prepare_export(other_job.request, case.result, case.credits)
    with pytest.raises(DomainError):
        _runtime(case).prepare_export(other_profile.request, case.result, case.credits)
    runtime = _runtime(case)
    runtime._environment = replace(
        case.environment,
        os_name=TextObservation("AVAILABLE", "SpecOS", None),
    )
    with pytest.raises(DomainError):
        runtime.prepare_export(case.request, case.result, case.credits)


def test_prepare_export_recompiles_and_rejects_forged_outer_spec_text() -> None:
    case = _case()
    forged = replace(
        case.request, spec_text=_case("talking_head_cover").request.spec_text
    )

    with pytest.raises(DomainError):
        _runtime(case).prepare_export(forged, case.result, case.credits)


@pytest.mark.parametrize(
    ("field", "mutate"),
    (
        ("job_id", lambda case: _HostileJobId(case.request.job_id.value)),
        ("spec_text", lambda case: _HostileStr(case.request.spec_text)),
        ("source", lambda _case: object()),
        ("style_references", lambda case: list(case.request.style_references)),
        ("prompt", lambda _case: object()),
        ("output_profile", lambda case: _HostileStr(case.request.output_profile)),
        ("variation_index", lambda _case: False),
        ("variation_index", lambda _case: True),
        ("variation_index", lambda _case: _HostileInt(0)),
        ("bundle_name", lambda case: _HostileStr(case.request.bundle_name)),
    ),
)
def test_prepare_export_rejects_hostile_outer_request_field_mutations(
    field: str, mutate: object
) -> None:
    case = _case()
    object.__setattr__(case.request, field, mutate(case))  # type: ignore[operator]

    with pytest.raises(DomainError):
        _runtime(case).prepare_export(case.request, case.result, case.credits)


def test_prepare_export_rejects_every_forged_result_binding() -> None:
    case = _case()
    profile = _case("talking_head_cover")
    rejected = _case(status=ArtifactStatus.REJECTED)
    other_job = _case(job_id="other")
    forged = (
        replace(case.result, compiled=profile.result.compiled),
        replace(case.result, graph=profile.result.graph),
        replace(case.result, verification_plan=profile.result.verification_plan),
        replace(case.result, request=other_job.result.request),
        replace(case.result, artifact=other_job.result.artifact),
        replace(case.result, report=rejected.result.report),
        replace(case.result, history=other_job.result.history),
        replace(case.result, terminal=rejected.result.terminal),
        replace(case.result, job_state=other_job.result.job_state),
    )

    for result in forged:
        with pytest.raises(DomainError):
            _runtime(case).prepare_export(case.request, result, case.credits)


def test_prepare_export_does_not_touch_store_gpu_or_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from specstyle.workflow import production_service

    case = _case()
    runtime = _runtime(case)

    class Trap:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("side effect dependency was touched")

        def __enter__(self) -> None:
            raise AssertionError("GPU lease was touched")

        def __exit__(self, *_args: object) -> None:
            raise AssertionError("GPU lease was touched")

    trap = Trap()
    runtime._job_store = trap
    monkeypatch.setattr(production_service, "_GPU_LEASE", trap)

    command = runtime.prepare_export(case.request, case.result, case.credits)

    assert command.export_request.environment == case.environment
    assert runtime._job_store is trap
