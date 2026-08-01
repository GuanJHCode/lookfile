"""Verified, local-only Diffusers/ROCm production loader."""

from __future__ import annotations

import importlib
import gc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from specstyle.domain.identifiers import Sha256
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.local_weights import ResolvedWeight
from specstyle.generation.model_approval import VerifiedPipelineSupply
from specstyle.generation.pipeline_factory import PipelineGraph
from specstyle.generation.rocm_probe import RocmProbeResult
from specstyle.observability.environment import EnvironmentSnapshot, hash_environment

PipelineBuilder = Callable[[PipelineGraph, tuple[ResolvedWeight, ...]], Any]
_CAPABILITY_SEAL = object()
_ROCM_VERSION = "7.2.1"
_DIFFUSERS_VERSION = "0.39.0"


@dataclass(frozen=True, slots=True, init=False)
class LoadedPipeline:
    """A loader-issued capability; the verified model supply remains caller-owned."""

    _pipeline: Any = field(repr=False, compare=False)
    _graph: PipelineGraph = field(repr=False, compare=False)
    _environment_hash: Sha256 = field(repr=False, compare=False)
    _runtime: tuple[str, str, str, str] = field(repr=False, compare=False)
    _torch: Any = field(repr=False, compare=False)
    _closed: bool = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("loaded pipelines are issued only by the production loader")

    def borrow_pipeline(self, /) -> Any:
        _validate_loaded_pipeline(self, require_open=True)
        return self._pipeline

    def close(self) -> None:
        _validate_loaded_pipeline(self, require_open=False)
        if self._closed:
            return
        pipeline = self._pipeline
        object.__setattr__(self, "_closed", True)
        object.__setattr__(self, "_pipeline", None)
        failure: InfrastructureError | None = None
        try:
            _release_resource(pipeline)
        except Exception:
            failure = InfrastructureError("pipeline release failed")
        pipeline = None
        gc.collect()
        _empty_cache(self._torch)
        if failure is not None:
            raise failure

    def __copy__(self) -> LoadedPipeline:
        raise TypeError("loaded pipelines cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> LoadedPipeline:
        raise TypeError("loaded pipelines cannot be copied")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("loaded pipelines cannot be serialized")


@dataclass(frozen=True, slots=True)
class LoadPlan:
    graph: PipelineGraph
    base: ResolvedWeight
    ip_adapter: ResolvedWeight
    controlnet: ResolvedWeight
    preview: ResolvedWeight | None
    rocm: RocmProbeResult


def plan_local_load(
    graph: PipelineGraph,
    cache_root: Path,
    relpaths: dict[str, str],
    *,
    require_gpu: bool = True,
    torch_module: Any | None = None,
) -> LoadPlan:
    """Fail closed: this pre-manifest entry point is not production-authorized."""
    raise DomainError("legacy production loader disabled; verified supply required")


def load_pipeline(
    plan: LoadPlan,
    *,
    builder: PipelineBuilder | None = None,
    local_files_only: bool = True,
) -> Any:
    """Fail closed before builders or download flags can affect behavior."""
    raise DomainError("legacy production loader disabled; verified supply required")


def _require_text(observation: object, expected: str) -> None:
    if (
        getattr(observation, "status", None) != "AVAILABLE"
        or getattr(observation, "value", None) != expected
        or getattr(observation, "reason", None) is not None
    ):
        raise DomainError("production environment mismatch")


def _validate_environment(environment: object, torch: Any, diffusers: Any) -> Sha256:
    if type(environment) is not EnvironmentSnapshot:
        raise DomainError("invalid production environment")
    _require_text(environment.rocm_version, _ROCM_VERSION)
    _require_text(environment.hip_version, getattr(torch.version, "hip", None))
    _require_text(environment.pytorch_version, getattr(torch, "__version__", None))
    _require_text(environment.diffusers_version, _DIFFUSERS_VERSION)
    if getattr(diffusers, "__version__", None) != _DIFFUSERS_VERSION:
        raise DomainError("production environment mismatch")
    devices = environment.hip_devices
    if getattr(devices, "status", None) != "AVAILABLE" or not devices.devices:
        raise DomainError("production environment mismatch")
    cuda = getattr(torch, "cuda", None)
    if (
        cuda is None
        or cuda.is_available() is not True
        or cuda.device_count() != len(devices.devices)
    ):
        raise DomainError("production device mismatch")
    for index, device in enumerate(devices.devices):
        if device.index != index:
            raise DomainError("production device mismatch")
        properties = cuda.get_device_properties(index)
        _require_text(device.name, cuda.get_device_name(index))
        _require_integer(device.total_memory_bytes, properties.total_memory)
        _require_text(device.gfx_arch, getattr(properties, "gcnArchName", None))
    return hash_environment(environment)


def _require_integer(observation: object, expected: object) -> None:
    if (
        getattr(observation, "status", None) != "AVAILABLE"
        or getattr(observation, "value", None) != expected
        or getattr(observation, "reason", None) is not None
    ):
        raise DomainError("production environment mismatch")


def _validated_components(supply: object, graph: object) -> dict[str, Any]:
    if type(supply) is not VerifiedPipelineSupply or type(graph) is not PipelineGraph:
        raise DomainError("invalid verified production supply")
    if graph.profile != "production" or graph.preview_adapter is not None:
        raise DomainError("invalid production pipeline graph")
    models = supply.models
    descriptors = (graph.base, graph.ip_adapter, graph.controlnet)
    roles = ("base", "ip_adapter", "controlnet")
    if tuple(item.role for item in descriptors) != roles:
        raise DomainError("invalid production pipeline graph")
    by_role = {component.role: component for component in models}
    if tuple(by_role) != roles:
        raise DomainError("invalid verified production supply")
    for descriptor, role in zip(descriptors, roles):
        component = by_role[role]
        if (
            component.model_id != descriptor.model_id
            or component.manifest.model_id != descriptor.model_id
            or component.manifest.revision != descriptor.revision
            or component.manifest.root_sha256 != descriptor.expected_sha256
            or component.approval.revision != descriptor.revision
        ):
            raise DomainError("graph and verified supply mismatch")
    return by_role


def _pretrained_kwargs(entrypoint: Any, torch: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "subfolder": entrypoint.subfolder,
        "local_files_only": True,
        "use_safetensors": True,
        "torch_dtype": torch.float16,
    }
    if entrypoint.variant is not None:
        kwargs["variant"] = entrypoint.variant
    return kwargs


def _ip_adapter_kwargs(entrypoint: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "subfolder": entrypoint.subfolder,
        "weight_name": entrypoint.weight_name,
        "local_files_only": True,
        "use_safetensors": True,
    }
    if entrypoint.image_encoder_folder is not None:
        kwargs["image_encoder_folder"] = entrypoint.image_encoder_folder
    if entrypoint.variant is not None:
        kwargs["variant"] = entrypoint.variant
    return kwargs


def _empty_cache(torch: Any) -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _release_resource(resource: Any) -> None:
    if resource is None:
        return
    failure = False
    if hasattr(resource, "maybe_free_model_hooks"):
        try:
            resource.maybe_free_model_hooks()
        except Exception:
            failure = True
    if hasattr(resource, "to"):
        try:
            resource.to("cpu")
        except Exception:
            failure = True
    if failure:
        raise InfrastructureError("pipeline resource release failed")


def _release_failed_pipeline(control: Any, pipeline: Any, torch: Any) -> None:
    try:
        _release_resource(pipeline)
    except Exception:
        pass
    try:
        _release_resource(control)
    except Exception:
        pass
    control = None
    pipeline = None


def _validate_loaded_pipeline(value: object, *, require_open: bool) -> None:
    if (
        type(value) is not LoadedPipeline
        or getattr(value, "_seal", None) is not _CAPABILITY_SEAL
        or type(getattr(value, "_graph", None)) is not PipelineGraph
        or type(getattr(value, "_environment_hash", None)) is not Sha256
        or type(getattr(value, "_runtime", None)) is not tuple
        or len(value._runtime) != 4
        or any(type(item) is not str for item in value._runtime)
        or value._runtime[3] != "float16"
        or type(getattr(value, "_closed", None)) is not bool
    ):
        raise DomainError("invalid loaded pipeline capability")
    if require_open and (value._closed or value._pipeline is None):
        raise DomainError("loaded pipeline is closed")


def load_production_pipeline(
    supply: VerifiedPipelineSupply,
    graph: PipelineGraph,
    environment: EnvironmentSnapshot,
    /,
    *,
    torch_module: Any | None = None,
    diffusers_module: Any | None = None,
) -> LoadedPipeline:
    """Load the one supported SDXL/Canny production topology from borrowed paths."""
    runtime_failure: InfrastructureError | None = None
    try:
        torch = (
            importlib.import_module("torch") if torch_module is None else torch_module
        )
        diffusers = (
            importlib.import_module("diffusers")
            if diffusers_module is None
            else diffusers_module
        )
    except Exception:
        torch = diffusers = None
        runtime_failure = InfrastructureError("production runtime unavailable")
    if runtime_failure is not None:
        raise runtime_failure
    environment_hash = _validate_environment(environment, torch, diffusers)
    components = _validated_components(supply, graph)
    control = None
    pipeline = None
    failure: InfrastructureError | None = None
    try:
        control = diffusers.ControlNetModel.from_pretrained(
            components["controlnet"].borrow_loader_path(),
            **_pretrained_kwargs(components["controlnet"].manifest.entrypoint, torch),
        )
        pipeline = diffusers.StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            components["base"].borrow_loader_path(),
            controlnet=control,
            **_pretrained_kwargs(components["base"].manifest.entrypoint, torch),
        )
        pipeline.scheduler = diffusers.EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline.to("cuda:0", torch.float16)
        pipeline.load_ip_adapter(
            components["ip_adapter"].borrow_loader_path(),
            **_ip_adapter_kwargs(components["ip_adapter"].manifest.entrypoint),
        )
    except Exception:
        failure = InfrastructureError("pipeline loading failed")

    if failure is not None:
        _release_failed_pipeline(control, pipeline, torch)
        control = pipeline = components = None
        gc.collect()
        _empty_cache(torch)
        raise failure

    instance = None
    try:
        instance = object.__new__(LoadedPipeline)
        object.__setattr__(instance, "_pipeline", pipeline)
        object.__setattr__(instance, "_graph", graph)
        object.__setattr__(instance, "_environment_hash", environment_hash)
        object.__setattr__(
            instance,
            "_runtime",
            (
                environment.rocm_version.value,
                environment.pytorch_version.value,
                environment.diffusers_version.value,
                "float16",
            ),
        )
        object.__setattr__(instance, "_torch", torch)
        object.__setattr__(instance, "_closed", False)
        object.__setattr__(instance, "_seal", _CAPABILITY_SEAL)
        _validate_loaded_pipeline(instance, require_open=True)
    except Exception:
        failure = InfrastructureError("pipeline loading failed")
    if failure is not None:
        _release_failed_pipeline(control, pipeline, torch)
        control = pipeline = instance = None
        gc.collect()
        _empty_cache(torch)
        raise failure
    return instance
