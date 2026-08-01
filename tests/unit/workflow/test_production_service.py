"""APP-COMPOSE-001A production generation composition contracts."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from dataclasses import fields
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from specstyle.domain.artifacts import ArtifactRef, AssetRef
from specstyle.domain.enums import (
    RuleLevel,
    RuleScope,
    RuleStatus,
    StaticApplicability,
)
from specstyle.domain.identifiers import (
    ArtifactId,
    AssetId,
    AttemptId,
    Identifier,
    JobId,
    RuleId,
    Sha256,
)
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.preprocess import (
    PreprocessPlan,
    PreparedImage,
    preprocess_image,
)
from specstyle.generation.protocols import GeneratedArtifact
from specstyle.generation.requests import RenderedPrompt
from specstyle.observability.hashing import hash_bytes
from specstyle.spec.compiled_models import ResourcePin
from specstyle.workflow.job_store import JobStore
from specstyle.verification.rule_models import GatePolicy, RuleDefinition, RuleResult


class _HostileStr(str):
    def __eq__(self, _other: object) -> bool:
        return True

    __hash__ = str.__hash__


class _HostileInt(int):
    pass


class _HostileTuple(tuple):
    pass


class _HostileJobId(JobId):
    pass


class _HostileAssetRef(AssetRef):
    pass


class _HostilePreparedImage(PreparedImage):
    pass


class _HostilePrompt(RenderedPrompt):
    pass


def _png(size: tuple[int, int] = (64, 64), color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, "PNG")
    return output.getvalue()


def _source() -> PreparedImage:
    content = _png()
    return preprocess_image(
        content,
        AssetRef(AssetId("source"), hash_bytes(content)),
        PreprocessPlan(
            (64, 64),
            "contain_pad",
            (0, 0, 0),
            ResourcePin("processor", "r1", hash_bytes(b"processor")),
        ),
    )


def _prompt() -> RenderedPrompt:
    return RenderedPrompt(
        ResourcePin("template", "r1", hash_bytes(b"template")),
        Identifier("preset"),
        "positive",
        "",
    )


def _style_references() -> tuple[AssetRef, ...]:
    return (
        AssetRef(AssetId("style-0"), hash_bytes(b"style-0")),
        AssetRef(AssetId("style-1"), hash_bytes(b"style-1")),
    )


def _request_type():
    try:
        module = importlib.import_module("specstyle.workflow.production_service")
    except ModuleNotFoundError:
        pytest.fail("production service module is missing")
    return module, module.ProductionJobRequest


def _request_kwargs() -> dict[str, object]:
    return {
        "job_id": JobId("job-1"),
        "spec_text": "spec",
        "source": _source(),
        "style_references": _style_references(),
        "prompt": _prompt(),
        "output_profile": "xhs_grid",
        "variation_index": 0,
        "bundle_name": "bundle-1",
    }


def test_request_is_the_only_public_frozen_slotted_surface() -> None:
    module, request_type = _request_type()

    request = request_type(**_request_kwargs())

    assert module.__all__ == ("ProductionJobRequest",)
    assert not hasattr(request, "__dict__")
    assert request.style_references == _style_references()
    with pytest.raises(FrozenInstanceError):
        request.bundle_name = "other"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", object()),
        ("job_id", _HostileJobId("job-1")),
        ("spec_text", object()),
        ("spec_text", _HostileStr("spec")),
        ("source", object()),
        (
            "source",
            lambda: _HostilePreparedImage(
                _source().source, _source().content, _source().snapshot
            ),
        ),
        ("style_references", []),
        ("style_references", ()),
        ("style_references", _HostileTuple(_style_references())),
        (
            "style_references",
            lambda: (
                _HostileAssetRef(
                    _style_references()[0].asset_id,
                    _style_references()[0].sha256,
                ),
            ),
        ),
        ("prompt", object()),
        (
            "prompt",
            lambda: _HostilePrompt(
                _prompt().template_pin,
                _prompt().preset_id,
                _prompt().positive,
                _prompt().negative,
            ),
        ),
        ("output_profile", "unknown"),
        ("output_profile", _HostileStr("xhs_grid")),
        ("variation_index", True),
        ("variation_index", -1),
        ("variation_index", _HostileInt(0)),
        ("bundle_name", "bad/name"),
        ("bundle_name", _HostileStr("bundle-1")),
    ],
)
def test_request_rejects_noncanonical_fields(field: str, value: object) -> None:
    _module, request_type = _request_type()
    kwargs = _request_kwargs()
    kwargs[field] = value() if callable(value) else value

    with pytest.raises(DomainError):
        request_type(**kwargs)


@pytest.mark.parametrize(
    ("output_profile", "maximum_job_id_length"),
    [
        ("xhs_grid", 114),
        ("talking_head_cover", 104),
        ("background_sequence", 103),
    ],
)
def test_request_accepts_only_job_ids_that_fit_the_exact_attempt_id(
    output_profile: str, maximum_job_id_length: int
) -> None:
    _module, request_type = _request_type()
    kwargs = _request_kwargs()
    kwargs["output_profile"] = output_profile
    kwargs["job_id"] = JobId("j" * maximum_job_id_length)

    request = request_type(**kwargs)
    expected_attempt_id = f"{'j' * maximum_job_id_length}-a0-{output_profile}-0"

    assert request.job_id.value == "j" * maximum_job_id_length
    assert len(expected_attempt_id) == 128
    assert AttemptId(expected_attempt_id).value == expected_attempt_id

    kwargs["job_id"] = JobId("j" * (maximum_job_id_length + 1))
    with pytest.raises(DomainError):
        request_type(**kwargs)


def test_module_has_no_forbidden_production_dependencies() -> None:
    module = _request_type()[0]
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    forbidden = (
        "specstyle.generation.fake_backend",
        "specstyle.workflow.real_pipeline",
        "specstyle.workflow.orchestrator",
        "specstyle.workflow.batch_runner",
        "specstyle.repair",
        "specstyle.exporting",
        "gradio",
        "tests",
    )
    assert not any(
        imported == prefix or imported.startswith(f"{prefix}.")
        for imported in imports
        for prefix in forbidden
    )
    assert {
        "specstyle.workflow.production_artifacts",
        "specstyle.verification.production",
        "specstyle.verification.protocols",
        "specstyle.verification.rule_models",
    }.issubset(imports)


class _UnusedControlBuilder:
    def build(self, _source: object, _graph: object) -> object:
        raise AssertionError("control input must not be built while opening")


def test_private_factory_loads_and_owns_only_the_loaded_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _request_type()[0]
    factory = getattr(module, "_open_production_generation_runtime", None)
    assert callable(factory)

    from tests.unit.generation.test_diffusers_loader import (
        _CLIPImageProcessor,
        _Diffusers,
        _Torch,
        _Transformers,
        _environment,
        _supply,
    )
    from tests.unit.spec.test_compiler import context
    from specstyle.generation.image_evidence import _build_processor_provenance
    from specstyle.verification.production import (
        _L1RuleMapping,
        _ProductionVerificationAllowlist,
    )

    supply, graph = _supply(tmp_path / "weights")
    store_root = tmp_path / "store"
    store_root.mkdir()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifact_root_fd = os.open(artifact_root, os.O_RDONLY | os.O_DIRECTORY)
    diffusers = _Diffusers()
    allowlist = _ProductionVerificationAllowlist(
        "specstyle.production_verifier.v1",
        context(),
        _build_processor_provenance(
            _Transformers, _CLIPImageProcessor(), _Transformers.__version__
        ),
        (
            _L1RuleMapping(RuleId("l1_bundle"), "technical_rgb_png_bundle_v1"),
            _L1RuleMapping(RuleId("l1_decode"), "decode_png_rgb_no_metadata_v1"),
            _L1RuleMapping(RuleId("l1_dimensions"), "dimensions_exact_v1"),
            _L1RuleMapping(RuleId("l1_pixels"), "pixels_nonblank_v1"),
        ),
    )
    issued_factories: list[object] = []
    create_factory = module._create_production_verifier_factory

    def tracked_create_factory(*args: object) -> object:
        issued = create_factory(*args)
        issued_factories.append(issued)
        return issued

    monkeypatch.setattr(
        module, "_create_production_verifier_factory", tracked_create_factory
    )
    runtime = factory(
        supply,
        graph,
        _environment(),
        context(),
        lambda _reference: b"",
        _UnusedControlBuilder(),
        allowlist,
        JobStore(store_root),
        artifact_root_fd,
        torch_module=_Torch(),
        diffusers_module=diffusers,
    )
    os.close(artifact_root_fd)
    pipeline = runtime._loaded.borrow_pipeline()
    owned_factory = runtime._verifier_factory

    runtime.close()
    runtime.close()

    assert issued_factories == [owned_factory]
    assert runtime._verifier_factory is None
    assert pipeline.hooks == 1
    assert supply.borrow_component("base").model_id == "base"
    supply.close()


def test_private_factory_and_runtime_shapes_are_frozen() -> None:
    module = _request_type()[0]
    signature = inspect.signature(module._open_production_generation_runtime)
    parameters = tuple(signature.parameters.values())

    assert tuple(parameter.name for parameter in parameters) == (
        "supply",
        "pipeline_graph",
        "environment",
        "compiler_context",
        "style_assets",
        "control_builder",
        "allowlist",
        "job_store",
        "artifact_root_fd",
        "torch_module",
        "diffusers_module",
        "clock",
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in parameters[:9]
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[9:]
    )
    assert parameters[-1].default is module._utc_now
    assert tuple(field.name for field in fields(module._InitialAttemptResult)) == (
        "compiled",
        "graph",
        "verification_plan",
        "request",
        "artifact",
        "report",
        "job_state",
    )
    assert module._ProductionGenerationRuntime.__slots__ == (
        "_loaded",
        "_verifier_factory",
        "_artifact_store",
        "_environment",
        "_compiler_context",
        "_style_assets",
        "_control_builder",
        "_job_store",
        "_clock",
        "_closed",
    )


def _runtime_with_close_probes(module, events: list[str], first_error: Exception):
    runtime = object.__new__(module._ProductionGenerationRuntime)

    class Loaded:
        def close(self) -> None:
            assert runtime._verifier_factory is None
            events.append("loaded")
            raise first_error

    class ArtifactStore:
        def close(self) -> None:
            events.append("artifact")
            raise InfrastructureError("later artifact close failure")

    values = {
        "_loaded": Loaded(),
        "_verifier_factory": object(),
        "_artifact_store": ArtifactStore(),
        "_environment": object(),
        "_compiler_context": object(),
        "_style_assets": object(),
        "_control_builder": object(),
        "_job_store": object(),
        "_clock": lambda: "2026-08-01T00:00:00.000Z",
        "_closed": False,
    }
    for name, value in values.items():
        setattr(runtime, name, value)
    return runtime


def test_close_detaches_factory_attempts_all_cleanup_and_preserves_first_error() -> (
    None
):
    module = _request_type()[0]
    events: list[str] = []
    first_error = InfrastructureError("first loaded close failure")
    runtime = _runtime_with_close_probes(module, events, first_error)

    with pytest.raises(InfrastructureError) as raised:
        runtime.close()
    runtime.close()

    assert raised.value is first_error
    assert events == ["loaded", "artifact"]
    assert runtime._closed is True


def test_execute_after_close_fails_before_validating_the_request() -> None:
    module = _request_type()[0]
    runtime = object.__new__(module._ProductionGenerationRuntime)
    runtime._closed = True

    with pytest.raises(InfrastructureError, match="^production runtime closed$"):
        runtime._execute_initial_attempt(object())


class _FlowRepository:
    def __init__(self, events: list[str], put_error: Exception | None = None) -> None:
        self.events = events
        self.put_error = put_error

    def put(self, _artifact: GeneratedArtifact, /) -> None:
        self.events.append("put")
        if self.put_error is not None:
            raise self.put_error

    def __call__(self, _reference: ArtifactRef, /) -> GeneratedArtifact | None:
        raise AssertionError("resolver execution is mocked in this flow test")

    def close(self) -> None:
        self.events.append("repository.close")


class _FlowArtifactStore:
    def __init__(self, events: list[str], repository: _FlowRepository) -> None:
        self.events = events
        self.repository = repository

    def for_job(self, _job_id: JobId, /) -> _FlowRepository:
        self.events.append("for_job")
        return self.repository


class _FlowFactory:
    def __init__(
        self, events: list[str], create_error: Exception | None = None
    ) -> None:
        self.events = events
        self.create_error = create_error

    def create(self, *_args: object) -> object:
        self.events.append("verifier.create")
        if self.create_error is not None:
            raise self.create_error
        return object()


class _FlowJobStore:
    def __init__(self, events: list[str], state: object) -> None:
        self.events = events
        self.state = state

    def get_snapshot(self, _job_id: JobId) -> None:
        self.events.append("duplicate.read")
        return None

    def load(self, _job_id: JobId) -> object:
        self.events.append("job.load")
        return self.state


def _flow_artifact() -> GeneratedArtifact:
    content = _png()
    return GeneratedArtifact(
        ArtifactRef(ArtifactId("artifact-1"), hash_bytes(content)),
        content,
        Sha256("1" * 64),
        Sha256("2" * 64),
    )


def _install_flow_mocks(module, monkeypatch, events: list[str]) -> tuple[object, ...]:
    compiled, graph, plan, generation_request = (object() for _ in range(4))
    artifact, report = _flow_artifact(), object()

    def marked(name: str, value: object = None) -> object:
        events.append(name)
        return value

    monkeypatch.setattr(
        module,
        "_select_initial_contract",
        lambda *_args: marked("compile/select", (compiled, graph, plan)),
    )
    monkeypatch.setattr(
        module, "_preflight_bindings", lambda *_args: marked("preflight")
    )
    monkeypatch.setattr(
        module,
        "_initial_generation_request",
        lambda *_args: marked("request.build", generation_request),
    )
    monkeypatch.setattr(
        module, "_create_initial_job", lambda *_args: marked("genesis", object())
    )
    monkeypatch.setattr(
        module, "_record_attempt_start", lambda *_args: marked("events.1-3")
    )
    monkeypatch.setattr(
        module,
        "_run_initial_generation",
        lambda *_args: marked("generation", artifact),
    )
    monkeypatch.setattr(
        module, "_record_attempt_finish", lambda *_args: marked("event.4")
    )
    monkeypatch.setattr(
        module,
        "_run_initial_verification",
        lambda *_args: marked("verification", report),
    )
    return compiled, graph, plan, generation_request, artifact, report


def _flow_runtime(
    module, events: list[str], repository: _FlowRepository, factory: _FlowFactory
):
    runtime = object.__new__(module._ProductionGenerationRuntime)
    state = object()
    values = {
        "_loaded": object(),
        "_verifier_factory": factory,
        "_artifact_store": _FlowArtifactStore(events, repository),
        "_environment": object(),
        "_compiler_context": object(),
        "_style_assets": object(),
        "_control_builder": object(),
        "_job_store": _FlowJobStore(events, state),
        "_clock": lambda: "2026-08-01T00:00:00.000Z",
        "_closed": False,
    }
    for name, value in values.items():
        setattr(runtime, name, value)
    return runtime, state


def test_initial_attempt_order_binds_verifier_before_genesis_and_persists_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, request_type = _request_type()
    events: list[str] = []
    repository = _FlowRepository(events)
    runtime, state = _flow_runtime(module, events, repository, _FlowFactory(events))
    expected = _install_flow_mocks(module, monkeypatch, events)

    result = runtime._execute_initial_attempt(request_type(**_request_kwargs()))

    assert events == [
        "compile/select",
        "preflight",
        "duplicate.read",
        "request.build",
        "for_job",
        "verifier.create",
        "genesis",
        "events.1-3",
        "generation",
        "put",
        "event.4",
        "verification",
        "job.load",
        "repository.close",
    ]
    assert tuple(getattr(result, name) for name in result.__dataclass_fields__) == (
        *expected[:5],
        expected[5],
        state,
    )


def test_verifier_bind_failure_is_pre_genesis_and_closes_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, request_type = _request_type()
    events: list[str] = []
    error = DomainError("invalid production verifier dependency")
    repository = _FlowRepository(events)
    runtime, _state = _flow_runtime(
        module, events, repository, _FlowFactory(events, error)
    )
    _install_flow_mocks(module, monkeypatch, events)

    with pytest.raises(DomainError) as raised:
        runtime._execute_initial_attempt(request_type(**_request_kwargs()))

    assert raised.value is error
    assert events == [
        "compile/select",
        "preflight",
        "duplicate.read",
        "request.build",
        "for_job",
        "verifier.create",
        "repository.close",
    ]


def test_artifact_persistence_failure_never_records_attempt_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, request_type = _request_type()
    events: list[str] = []
    error = InfrastructureError("production artifact store unavailable")
    repository = _FlowRepository(events, error)
    runtime, _state = _flow_runtime(module, events, repository, _FlowFactory(events))
    _install_flow_mocks(module, monkeypatch, events)

    with pytest.raises(InfrastructureError) as raised:
        runtime._execute_initial_attempt(request_type(**_request_kwargs()))

    assert raised.value is error
    assert events[-3:] == ["generation", "put", "repository.close"]
    assert "event.4" not in events
    assert "verification" not in events


def _verification_rule() -> RuleDefinition:
    return RuleDefinition(
        RuleId("l1-test"),
        RuleLevel.L1,
        RuleScope.ITEM,
        True,
        StaticApplicability.APPLICABLE,
        GatePolicy("reject", "reject", "reject"),
    )


class _PassingVerifier:
    def verify(self, artifacts, rules, /):
        return (
            RuleResult(
                rules[0].rule_id,
                RuleStatus.PASS,
                (artifacts[0].artifact_id,),
                1.0,
            ),
        )


class _FailingVerifier:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def verify(self, _artifacts, _rules, /):
        raise self.error


class _VerificationEventStore:
    def __init__(self) -> None:
        self.events: list[object] = []

    def append_event(self, _job_id: JobId, event: object) -> None:
        self.events.append(event)


def test_verification_report_preserves_exact_artifact_rules_and_results() -> None:
    module = _request_type()[0]
    artifact, rule = _flow_artifact(), _verification_rule()
    plan = SimpleNamespace(applicable_rule_definitions=(rule,))
    request = SimpleNamespace(job_id=JobId("job-1"))
    store = _VerificationEventStore()

    report = module._run_initial_verification(
        _PassingVerifier(),
        store,
        request,
        artifact,
        plan,
        lambda: "2026-08-01T00:00:00.000Z",
    )

    assert report.artifacts == (artifact.ref,)
    assert report.rules == (rule,)
    assert report.results[0].status is RuleStatus.PASS
    assert store.events == []


@pytest.mark.parametrize(
    "error",
    (
        DomainError("sensitive domain cause"),
        InfrastructureError("sensitive infrastructure cause"),
        RuntimeError("sensitive unknown cause"),
    ),
)
def test_verifier_failure_records_fatal_then_raises_sanitized_boundary_error(
    error: Exception,
) -> None:
    module = _request_type()[0]
    artifact, rule = _flow_artifact(), _verification_rule()
    plan = SimpleNamespace(applicable_rule_definitions=(rule,))
    request = SimpleNamespace(job_id=JobId("job-1"))
    store = _VerificationEventStore()

    with pytest.raises(InfrastructureError, match="^verifier unavailable$") as raised:
        module._run_initial_verification(
            _FailingVerifier(error),
            store,
            request,
            artifact,
            plan,
            lambda: "2026-08-01T00:00:00.000Z",
        )

    assert raised.value is not error
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert len(store.events) == 1
    event = store.events[0]
    assert (event.event_type.value, event.from_state.value, event.to_state.value) == (
        "FATAL",
        "VERIFYING",
        "JOB_FAILED",
    )
    assert (event.payload.error_family, event.payload.reason) == (
        "VERIFIER_UNAVAILABLE",
        "verifier unavailable",
    )
    assert "sensitive" not in event.payload.reason
