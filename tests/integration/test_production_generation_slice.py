"""APP-COMPOSE-001A real production generation composition slice."""

from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import specstyle.workflow.production_service as production_service
from specstyle.domain.artifacts import AssetRef
from specstyle.domain.enums import RuleScope
from specstyle.domain.identifiers import AssetId, Identifier, JobId
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.preprocess import (
    PreprocessPlan,
    PreparedImage,
    preprocess_image,
)
from specstyle.generation.requests import PreparedControlInput, RenderedPrompt
from specstyle.observability.hashing import hash_bytes
from specstyle.spec.compiled_models import (
    CompilerContext,
    ModelCapability,
    ResourcePin,
    RuntimeCapability,
)
from specstyle.workflow.job_models import EventType, JobStatus
from specstyle.workflow.job_store import JobStore
from specstyle.workflow.production_service import (
    ProductionJobRequest,
    _open_production_generation_runtime,
)
from tests.unit.generation.test_diffusers_loader import (
    _Diffusers,
    _Pipeline,
    _Torch,
    _environment,
    _supply,
)
from tests.unit.spec.test_compiler import context, raw_spec

_TIMESTAMP = "2026-08-01T00:00:00.000Z"


def _png(color: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (1024, 1024), color).save(output, "PNG")
    return output.getvalue()


def _compiler_inputs(
    pipeline_graph: object,
    style_contents: tuple[bytes, ...],
    *,
    applicable_batch: bool = False,
    mismatch: str | None = None,
) -> tuple[str, CompilerContext]:
    raw = raw_spec().model_dump(mode="json")
    raw["runtime"] = {
        "backend": "rocm",
        "rocm_version": "7.2.0" if mismatch == "runtime" else "7.2.1",
        "torch_version": "2.8.0",
        "diffusers_version": "0.39.0",
        "dtype": "float16",
    }
    for key, descriptor in (
        ("base", pipeline_graph.base),
        ("ip_adapter", pipeline_graph.ip_adapter),
        ("controlnet", pipeline_graph.controlnet),
    ):
        raw["models"][key]["id"] = (
            "other-base"
            if mismatch == "model" and key == "base"
            else descriptor.model_id
        )
        raw["models"][key]["revision"] = descriptor.revision
        raw["models"][key]["sha256"] = descriptor.expected_sha256.value
    if mismatch == "control":
        raw["models"]["controlnet"]["type"] = "depth"
    raw["assets"]["style_references"] = [
        {
            "asset_sha256": hash_bytes(content).value,
            "source_url": f"https://example.com/style-{index}",
            "license": "CC0",
            "attribution": "author",
            "consent": "not_applicable",
        }
        for index, content in enumerate(style_contents)
    ]

    base_context = context()
    runtime_pin = ResourcePin(
        "runtime", "r1", base_context.runtime_capabilities[0].pin.sha256
    )
    model_capabilities = tuple(
        ModelCapability(
            descriptor.role,
            ResourcePin(
                "other-base"
                if mismatch == "model" and descriptor.role == "base"
                else descriptor.model_id,
                descriptor.revision,
                descriptor.expected_sha256,
            ),
            ("depth" if mismatch == "control" else "canny")
            if descriptor.role == "controlnet"
            else None,
            ("sdxl_turbo", "sdxl_base"),
            ("float16",),
            (runtime_pin.sha256,),
        )
        for descriptor in (
            pipeline_graph.base,
            pipeline_graph.ip_adapter,
            pipeline_graph.controlnet,
        )
    )
    catalog = base_context.rule_catalogs[0]
    rules = (
        catalog.rules
        if applicable_batch
        else tuple(
            replace(rule, supported_output_profiles=("talking_head_cover",))
            if rule.scope is RuleScope.BATCH
            else rule
            for rule in catalog.rules
        )
    )
    threshold = base_context.threshold_profiles[0]
    if not applicable_batch:
        threshold = replace(
            threshold,
            metrics=tuple(
                metric
                for metric in threshold.metrics
                if metric.metric_id.value == "style-metric"
            ),
        )
    compiler_context = replace(
        base_context,
        runtime_capabilities=(
            RuntimeCapability(
                runtime_pin,
                "rocm",
                "7.2.0" if mismatch == "runtime" else "7.2.1",
                "2.8.0",
                "0.39.0",
                "float16",
            ),
        ),
        model_capabilities=model_capabilities,
        rule_catalogs=(replace(catalog, rules=rules),),
        threshold_profiles=(threshold,),
    )
    return json.dumps(raw), compiler_context


def _source() -> PreparedImage:
    content = _png("white")
    return preprocess_image(
        content,
        AssetRef(AssetId("source"), hash_bytes(content)),
        PreprocessPlan(
            (1024, 1024),
            "contain_pad",
            (0, 0, 0),
            ResourcePin("processor", "r1", hash_bytes(b"processor")),
        ),
    )


class _CannyBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[PreparedImage, object]] = []

    def build(self, source: PreparedImage, graph: object) -> PreparedControlInput:
        self.calls.append((source, graph))
        return PreparedControlInput("canny", source)


class _Generator:
    def __init__(self, device: str) -> None:
        self.device = device

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def _job_request(
    job_id: str,
    spec_text: str,
    style_references: tuple[AssetRef, ...],
) -> ProductionJobRequest:
    return ProductionJobRequest(
        JobId(job_id),
        spec_text,
        _source(),
        style_references,
        RenderedPrompt(
            ResourcePin("template", "r1", hash_bytes(b"template")),
            Identifier("preset"),
            "positive",
            "negative",
        ),
        "xhs_grid",
        0,
        f"bundle-{job_id}",
    )


def test_real_initial_attempt_reaches_verifying_with_exact_audit_history(
    tmp_path: Path, monkeypatch
) -> None:
    style_contents = (_png("red"), _png("blue"))
    supply, pipeline_graph = _supply(tmp_path / "weights")
    spec_text, compiler_context = _compiler_inputs(pipeline_graph, style_contents)
    style_references = tuple(
        AssetRef(AssetId(f"style-{index}"), hash_bytes(content))
        for index, content in enumerate(style_contents)
    )
    request = _job_request("job-success", spec_text, style_references)
    builder = _CannyBuilder()
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = JobStore(store_root)
    torch, diffusers = _Torch(), _Diffusers()
    torch.Generator = _Generator
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        _Pipeline, "set_ip_adapter_scale", lambda self, value: None, raising=False
    )

    def generate(pipeline: _Pipeline, **kwargs: object) -> object:
        observed["style_colors"] = tuple(
            image.getpixel((0, 0)) for image in kwargs["ip_adapter_image"][0]
        )
        observed["control_image"] = kwargs["control_image"]
        return type(
            "Result", (), {"images": [Image.new("RGB", (1024, 1024), "green")]}
        )()

    monkeypatch.setattr(_Pipeline, "__call__", generate, raising=False)
    runtime = _open_production_generation_runtime(
        supply,
        pipeline_graph,
        _environment(),
        compiler_context,
        dict(zip(style_references, style_contents)).__getitem__,
        builder,
        store,
        torch_module=torch,
        diffusers_module=diffusers,
        clock=lambda: _TIMESTAMP,
    )
    try:
        result = runtime._execute_initial_attempt(request)

        assert result.graph.output_profile == "xhs_grid"
        assert result.verification_plan.output_profile == "xhs_grid"
        assert result.request.attempt_id.value == "job-success-a0-xhs_grid-0"
        assert result.artifact.content.startswith(b"\x89PNG")
        assert result.artifact.request_hash == result.request.request_hash
        assert result.job_state.job.status is JobStatus.VERIFYING
        assert result.job_state.last_sequence == 4
        assert len(builder.calls) == 1
        assert observed["style_colors"] == ((255, 0, 0), (0, 0, 255))
        assert observed["control_image"].size == (1024, 1024)
        events = store.list_events(request.job_id)
        assert tuple(event.event_type for event in events) == (
            EventType.JOB_STARTED,
            EventType.SPEC_COMPILED,
            EventType.ATTEMPT_STARTED,
            EventType.ATTEMPT_FINISHED,
        )
        assert tuple((event.from_state, event.to_state) for event in events) == (
            (JobStatus.CREATED, JobStatus.SPEC_VALIDATED),
            (JobStatus.SPEC_VALIDATED, JobStatus.SPEC_COMPILED),
            (JobStatus.SPEC_COMPILED, JobStatus.GENERATING),
            (JobStatus.GENERATING, JobStatus.VERIFYING),
        )
    finally:
        runtime.close()
        supply.close()


@pytest.mark.parametrize("case", ["batch", "model", "runtime", "control", "style"])
def test_preflight_rejects_before_genesis_and_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    style_contents = (_png("red"), _png("blue"))
    supply, pipeline_graph = _supply(tmp_path / "weights")
    spec_text, compiler_context = _compiler_inputs(
        pipeline_graph,
        style_contents,
        applicable_batch=case == "batch",
        mismatch=case if case in {"model", "runtime", "control"} else None,
    )
    style_references = tuple(
        AssetRef(AssetId(f"style-{index}"), hash_bytes(content))
        for index, content in enumerate(style_contents)
    )
    if case == "style":
        style_references = (
            AssetRef(AssetId("style-0"), hash_bytes(b"wrong-style")),
            style_references[1],
        )
    request = _job_request(f"job-{case}", spec_text, style_references)
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = JobStore(store_root)
    builder = _CannyBuilder()
    inference_calls: list[object] = []
    monkeypatch.setattr(
        _Pipeline,
        "__call__",
        lambda self, **kwargs: inference_calls.append(kwargs),
        raising=False,
    )
    runtime = _open_production_generation_runtime(
        supply,
        pipeline_graph,
        _environment(),
        compiler_context,
        dict(zip(style_references, style_contents)).__getitem__,
        builder,
        store,
        torch_module=_Torch(),
        diffusers_module=_Diffusers(),
        clock=lambda: _TIMESTAMP,
    )
    try:
        with pytest.raises(DomainError):
            runtime._execute_initial_attempt(request)
        assert store.get_snapshot(request.job_id) is None
        assert store.list_events(request.job_id) == ()
        assert inference_calls == []
        assert builder.calls == []
    finally:
        runtime.close()
        supply.close()


@pytest.mark.parametrize("failure", ["oom", "style", "output"])
def test_generation_failure_is_fatal_and_reraises_the_same_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    style_contents = (_png("red"),)
    supply, pipeline_graph = _supply(tmp_path / "weights")
    spec_text, compiler_context = _compiler_inputs(pipeline_graph, style_contents)
    style_references = (AssetRef(AssetId("style-0"), hash_bytes(style_contents[0])),)
    request = _job_request(f"job-{failure}", spec_text, style_references)
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = JobStore(store_root)
    torch, diffusers = _Torch(), _Diffusers()
    torch.Generator = _Generator

    class OutOfMemoryError(Exception):
        pass

    torch.cuda.OutOfMemoryError = OutOfMemoryError
    monkeypatch.setattr(
        _Pipeline, "set_ip_adapter_scale", lambda self, value: None, raising=False
    )

    def generate(_pipeline: _Pipeline, **_kwargs: object) -> object:
        if failure == "oom":
            raise OutOfMemoryError("GPU exhausted")
        size = (64, 64) if failure == "output" else (1024, 1024)
        return type("Result", (), {"images": [Image.new("RGB", size)]})()

    monkeypatch.setattr(_Pipeline, "__call__", generate, raising=False)
    resolver = (
        (lambda _reference: b"not-an-image")
        if failure == "style"
        else dict(zip(style_references, style_contents)).__getitem__
    )
    original_run_generation = production_service.run_generation
    observed: dict[str, InfrastructureError] = {}

    def capture_error(backend: object, generation_request: object):
        try:
            return original_run_generation(backend, generation_request)
        except InfrastructureError as error:
            observed["error"] = error
            raise

    monkeypatch.setattr(production_service, "run_generation", capture_error)
    runtime = _open_production_generation_runtime(
        supply,
        pipeline_graph,
        _environment(),
        compiler_context,
        resolver,
        _CannyBuilder(),
        store,
        torch_module=torch,
        diffusers_module=diffusers,
        clock=lambda: _TIMESTAMP,
    )
    try:
        with pytest.raises(InfrastructureError) as raised:
            runtime._execute_initial_attempt(request)

        assert raised.value is observed["error"]
        assert raised.value.args == (
            ("generation OOM",) if failure == "oom" else raised.value.args
        )
        events = store.list_events(request.job_id)
        assert tuple(event.event_type for event in events) == (
            EventType.JOB_STARTED,
            EventType.SPEC_COMPILED,
            EventType.ATTEMPT_STARTED,
            EventType.FATAL,
        )
        assert events[-1].from_state is JobStatus.GENERATING
        assert events[-1].to_state is JobStatus.JOB_FAILED
        assert events[-1].payload.error_family == (
            "GENERATION_OOM" if failure == "oom" else "GENERATION_FAILED"
        )
        assert store.load(request.job_id).job.status is JobStatus.JOB_FAILED
    finally:
        runtime.close()
        supply.close()


def test_duplicate_job_is_rejected_without_events_or_inference_and_backends_are_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    style_contents = (_png("red"),)
    supply, pipeline_graph = _supply(tmp_path / "weights")
    spec_text, compiler_context = _compiler_inputs(pipeline_graph, style_contents)
    style_references = (AssetRef(AssetId("style-0"), hash_bytes(style_contents[0])),)
    first = _job_request("job-first", spec_text, style_references)
    second = _job_request("job-second", spec_text, style_references)
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = JobStore(store_root)
    torch, diffusers = _Torch(), _Diffusers()
    torch.Generator = _Generator
    inference_calls: list[object] = []
    monkeypatch.setattr(
        _Pipeline, "set_ip_adapter_scale", lambda self, value: None, raising=False
    )

    def generate(_pipeline: _Pipeline, **kwargs: object) -> object:
        inference_calls.append(kwargs)
        return type("Result", (), {"images": [Image.new("RGB", (1024, 1024))]})()

    monkeypatch.setattr(_Pipeline, "__call__", generate, raising=False)
    real_backend = production_service.DiffusersBackend
    backend_instances: list[object] = []

    def create_backend(*args: object) -> object:
        backend = real_backend(*args)
        backend_instances.append(backend)
        return backend

    monkeypatch.setattr(production_service, "DiffusersBackend", create_backend)
    runtime = _open_production_generation_runtime(
        supply,
        pipeline_graph,
        _environment(),
        compiler_context,
        dict(zip(style_references, style_contents)).__getitem__,
        _CannyBuilder(),
        store,
        torch_module=torch,
        diffusers_module=diffusers,
        clock=lambda: _TIMESTAMP,
    )
    try:
        runtime._execute_initial_attempt(first)
        original_events = store.list_events(first.job_id)
        with pytest.raises(DomainError, match="duplicate"):
            runtime._execute_initial_attempt(first)
        assert store.list_events(first.job_id) == original_events
        assert len(inference_calls) == 1

        runtime._execute_initial_attempt(second)
        assert len(inference_calls) == 2
        assert len(backend_instances) == 2
        assert backend_instances[0] is not backend_instances[1]
    finally:
        runtime.close()
        supply.close()


@pytest.mark.parametrize("failure_stage", ["finish", "fatal"])
def test_job_store_write_failure_propagates_the_original_infrastructure_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    style_contents = (_png("red"),)
    supply, pipeline_graph = _supply(tmp_path / "weights")
    spec_text, compiler_context = _compiler_inputs(pipeline_graph, style_contents)
    style_references = (AssetRef(AssetId("style-0"), hash_bytes(style_contents[0])),)
    request = _job_request(f"job-store-{failure_stage}", spec_text, style_references)
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = JobStore(store_root)
    torch, diffusers = _Torch(), _Diffusers()
    torch.Generator = _Generator
    monkeypatch.setattr(
        _Pipeline, "set_ip_adapter_scale", lambda self, value: None, raising=False
    )

    def generate(_pipeline: _Pipeline, **_kwargs: object) -> object:
        size = (64, 64) if failure_stage == "fatal" else (1024, 1024)
        return type("Result", (), {"images": [Image.new("RGB", size)]})()

    monkeypatch.setattr(_Pipeline, "__call__", generate, raising=False)
    write_error = InfrastructureError("injected job store write failure")
    real_append = store.append_event

    def append_event(job_id: JobId, event: object) -> object:
        target = (
            EventType.ATTEMPT_FINISHED if failure_stage == "finish" else EventType.FATAL
        )
        if event.event_type is target:
            raise write_error
        return real_append(job_id, event)

    monkeypatch.setattr(store, "append_event", append_event)
    runtime = _open_production_generation_runtime(
        supply,
        pipeline_graph,
        _environment(),
        compiler_context,
        dict(zip(style_references, style_contents)).__getitem__,
        _CannyBuilder(),
        store,
        torch_module=torch,
        diffusers_module=diffusers,
        clock=lambda: _TIMESTAMP,
    )
    try:
        with pytest.raises(InfrastructureError) as raised:
            runtime._execute_initial_attempt(request)
        assert raised.value is write_error
        assert tuple(
            event.event_type for event in store.list_events(request.job_id)
        ) == (
            EventType.JOB_STARTED,
            EventType.SPEC_COMPILED,
            EventType.ATTEMPT_STARTED,
        )
        assert store.load(request.job_id).job.status is JobStatus.GENERATING
    finally:
        runtime.close()
        supply.close()


def _configure_clock_pipeline(
    monkeypatch: pytest.MonkeyPatch, torch: object, failure: str | None
) -> dict[str, InfrastructureError]:
    class OutOfMemoryError(Exception):
        pass

    torch.Generator = _Generator
    torch.cuda.OutOfMemoryError = OutOfMemoryError
    monkeypatch.setattr(
        _Pipeline, "set_ip_adapter_scale", lambda self, value: None, raising=False
    )

    def generate(_pipeline: _Pipeline, **_kwargs: object) -> object:
        if failure == "oom":
            raise OutOfMemoryError("GPU exhausted")
        if failure == "generation":
            raise RuntimeError("inference failed")
        return type("Result", (), {"images": [Image.new("RGB", (1024, 1024))]})()

    monkeypatch.setattr(_Pipeline, "__call__", generate, raising=False)
    original_run_generation = production_service.run_generation
    observed: dict[str, InfrastructureError] = {}

    def capture_error(backend: object, generation_request: object):
        try:
            return original_run_generation(backend, generation_request)
        except InfrastructureError as error:
            observed["error"] = error
            raise

    monkeypatch.setattr(production_service, "run_generation", capture_error)
    return observed


def _open_clock_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timestamps: tuple[str, ...],
    *,
    failure: str | None = None,
) -> tuple[Any, Any, JobStore, ProductionJobRequest, dict[str, InfrastructureError]]:
    style_contents = (_png("red"),)
    supply, pipeline_graph = _supply(tmp_path / "weights")
    spec_text, compiler_context = _compiler_inputs(pipeline_graph, style_contents)
    style_references = (AssetRef(AssetId("style-0"), hash_bytes(style_contents[0])),)
    request = _job_request("job-clock", spec_text, style_references)
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = JobStore(store_root)
    torch, diffusers = _Torch(), _Diffusers()
    observed = _configure_clock_pipeline(monkeypatch, torch, failure)
    clock = iter(timestamps).__next__
    runtime = _open_production_generation_runtime(
        supply,
        pipeline_graph,
        _environment(),
        compiler_context,
        dict(zip(style_references, style_contents)).__getitem__,
        _CannyBuilder(),
        store,
        torch_module=torch,
        diffusers_module=diffusers,
        clock=clock,
    )
    return runtime, supply, store, request, observed


def test_start_event_clock_rollbacks_are_clamped_through_verifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, supply, store, request, _observed = _open_clock_case(
        tmp_path,
        monkeypatch,
        (
            "2026-08-01T00:00:00.400Z",
            "2026-08-01T00:00:00.300Z",
            "2026-08-01T00:00:00.200Z",
            "2026-08-01T00:00:00.100Z",
            "2026-08-01T00:00:00.500Z",
        ),
    )
    try:
        result = runtime._execute_initial_attempt(request)

        assert result.job_state.job.status is JobStatus.VERIFYING
        assert tuple(
            event.timestamp for event in store.list_events(request.job_id)
        ) == (
            "2026-08-01T00:00:00.400Z",
            "2026-08-01T00:00:00.400Z",
            "2026-08-01T00:00:00.400Z",
            "2026-08-01T00:00:00.500Z",
        )
    finally:
        runtime.close()
        supply.close()


def test_finish_clock_rollback_is_clamped_without_changing_success_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, supply, store, request, _observed = _open_clock_case(
        tmp_path,
        monkeypatch,
        (
            "2026-08-01T00:00:00.000Z",
            "2026-08-01T00:00:00.100Z",
            "2026-08-01T00:00:00.200Z",
            "2026-08-01T00:00:00.300Z",
            "2026-08-01T00:00:00.250Z",
        ),
    )
    try:
        result = runtime._execute_initial_attempt(request)

        events = store.list_events(request.job_id)
        assert result.job_state.job.status is JobStatus.VERIFYING
        assert events[-1].event_type is EventType.ATTEMPT_FINISHED
        assert events[-1].timestamp == "2026-08-01T00:00:00.300Z"
    finally:
        runtime.close()
        supply.close()


@pytest.mark.parametrize(
    ("failure", "error_family", "message"),
    [
        ("oom", "GENERATION_OOM", "generation OOM"),
        ("generation", "GENERATION_FAILED", "generation failed"),
    ],
)
def test_fatal_clock_rollback_persists_fatal_and_reraises_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    error_family: str,
    message: str,
) -> None:
    runtime, supply, store, request, observed = _open_clock_case(
        tmp_path,
        monkeypatch,
        (
            "2026-08-01T00:00:00.000Z",
            "2026-08-01T00:00:00.100Z",
            "2026-08-01T00:00:00.200Z",
            "2026-08-01T00:00:00.300Z",
            "2026-08-01T00:00:00.250Z",
        ),
        failure=failure,
    )
    try:
        with pytest.raises(InfrastructureError, match=message) as raised:
            runtime._execute_initial_attempt(request)

        events = store.list_events(request.job_id)
        assert raised.value is observed["error"]
        assert events[-1].event_type is EventType.FATAL
        assert events[-1].timestamp == "2026-08-01T00:00:00.300Z"
        assert events[-1].payload.error_family == error_family
        assert store.load(request.job_id).job.status is JobStatus.JOB_FAILED
    finally:
        runtime.close()
        supply.close()


def test_invalid_clock_value_is_still_rejected_by_job_event_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, supply, store, request, _observed = _open_clock_case(
        tmp_path,
        monkeypatch,
        ("2026-08-01T00:00:00.000Z", "!"),
    )
    try:
        with pytest.raises(DomainError, match="invalid timestamp"):
            runtime._execute_initial_attempt(request)
        assert store.get_snapshot(request.job_id) is not None
        assert store.list_events(request.job_id) == ()
    finally:
        runtime.close()
        supply.close()
