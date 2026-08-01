"""GEN-004R Diffusers loader contract (all dependencies are injected fakes)."""

from __future__ import annotations

import copy
import os
import pickle
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

from specstyle.domain.identifiers import Sha256
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.diffusers_loader import load_production_pipeline
from specstyle.generation.model_approval import verify_pipeline_supply
from specstyle.generation.model_registry import ModelDescriptor, ModelRegistry
from specstyle.generation.pipeline_factory import PipelineFactory
from specstyle.generation.weight_manifest import (
    ModelLoadEntrypoint,
    WeightFile,
    WeightManifest,
)
from specstyle.observability.environment import (
    DeviceInventory,
    DeviceSnapshot,
    EnvironmentSnapshot,
    IntegerObservation,
    TextObservation,
)
from specstyle.observability.hashing import hash_bytes


_REVISION = "a" * 40


class _Cuda:
    def __init__(self, devices: tuple[tuple[str, int, str], ...] | None = None) -> None:
        self.empty_cache_calls = 0
        self.devices = devices or (("AMD test", 16, "gfx1100"),)

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self.devices)

    def get_device_name(self, index: int) -> str:
        return self.devices[index][0]

    def get_device_properties(self, index: int):
        name, memory, arch = self.devices[index]
        return type("Properties", (), {"total_memory": memory, "gcnArchName": arch})()

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _Torch:
    __version__ = "2.8.0"
    float16 = object()
    version = type("Version", (), {"hip": "7.2.1"})()

    def __init__(self) -> None:
        self.cuda = _Cuda()


class _Pipeline:
    def __init__(self) -> None:
        self.scheduler = type("Scheduler", (), {"config": {"x": 1}})()
        self.to_calls: list[tuple[object, ...]] = []
        self.ip_calls: list[tuple[object, dict[str, object]]] = []
        self.hooks = 0

    def to(self, *args: object) -> _Pipeline:
        self.to_calls.append(args)
        return self

    def load_ip_adapter(self, *args: object, **kwargs: object) -> None:
        self.ip_calls.append((args, kwargs))

    def maybe_free_model_hooks(self) -> None:
        self.hooks += 1


class _Diffusers:
    __version__ = "0.39.0"

    def __init__(self) -> None:
        self.control_calls: list[tuple[object, dict[str, object]]] = []
        self.pipeline_calls: list[tuple[object, dict[str, object]]] = []
        self.scheduler_calls: list[object] = []
        outer = self

        class ControlNetModel:
            @classmethod
            def from_pretrained(cls, *args: object, **kwargs: object) -> object:
                outer.control_calls.append((args, kwargs))
                return object()

        class StableDiffusionXLControlNetImg2ImgPipeline:
            @classmethod
            def from_pretrained(cls, *args: object, **kwargs: object) -> _Pipeline:
                outer.pipeline_calls.append((args, kwargs))
                return _Pipeline()

        class EulerDiscreteScheduler:
            @classmethod
            def from_config(cls, config: object) -> object:
                outer.scheduler_calls.append(config)
                return object()

        self.ControlNetModel = ControlNetModel
        self.StableDiffusionXLControlNetImg2ImgPipeline = (
            StableDiffusionXLControlNetImg2ImgPipeline
        )
        self.EulerDiscreteScheduler = EulerDiscreteScheduler


def _available(value: str) -> TextObservation:
    return TextObservation("AVAILABLE", value, None)


def _environment() -> EnvironmentSnapshot:
    device = DeviceSnapshot(
        0,
        _available("AMD test"),
        IntegerObservation("AVAILABLE", 16, None),
        _available("gfx1100"),
    )
    return EnvironmentSnapshot(
        "1.0",
        *(_available("x") for _ in range(6)),
        _available("7.2.1"),
        _available("7.2.1"),
        _available("2.8.0"),
        _available("0.39.0"),
        DeviceInventory("AVAILABLE", None, (device,)),
    )


def _supply(tmp_path: Path):
    descriptors = []
    manifests = []
    approvals = []
    for model_id, role in (
        ("base", "base"),
        ("ip", "ip_adapter"),
        ("cn", "controlnet"),
    ):
        payload = model_id.encode()
        root = tmp_path / model_id
        (root / "pipeline").mkdir(parents=True)
        (root / "pipeline" / "model.safetensors").write_bytes(payload)
        entrypoint = ModelLoadEntrypoint(
            "diffusers_ip_adapter" if role == "ip_adapter" else "diffusers_pretrained",
            "pipeline",
            "model.safetensors" if role == "ip_adapter" else None,
        )
        manifest = WeightManifest(
            model_id,
            role,
            _REVISION,
            model_id,
            entrypoint,
            (
                WeightFile(
                    "pipeline/model.safetensors", len(payload), hash_bytes(payload)
                ),
            ),
            Sha256("0" * 64),
        ).with_computed_root()
        descriptor = ModelDescriptor(
            model_id, role, _REVISION, manifest.root_sha256, "MIT", "APPROVED", "sdxl"
        )
        from specstyle.generation.model_approval import LicenseApproval
        from specstyle.generation.weight_manifest import manifest_sha256

        approvals.append(
            LicenseApproval(
                model_id,
                _REVISION,
                manifest_sha256(manifest),
                "MIT",
                f"https://example.com/{model_id}",
            )
        )
        descriptors.append(descriptor)
        manifests.append(manifest)
    graph = PipelineFactory(
        ModelRegistry(tuple(descriptors)), tmp_path
    ).build_production("base", "ip", "cn")
    root_fd = os.open(tmp_path, os.O_RDONLY)
    return verify_pipeline_supply(
        root_fd, graph, tuple(manifests), tuple(approvals)
    ), graph


def test_loads_local_safetensors_with_torch_dtype_and_sealed_capability(
    tmp_path: Path,
) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()

    loaded = load_production_pipeline(
        supply, graph, _environment(), torch_module=torch, diffusers_module=diffusers
    )

    assert type(loaded).__name__ == "LoadedPipeline"
    assert diffusers.control_calls[0][1]["torch_dtype"] is torch.float16
    assert "dtype" not in diffusers.control_calls[0][1]
    assert diffusers.pipeline_calls[0][1]["torch_dtype"] is torch.float16
    assert diffusers.pipeline_calls[0][1]["local_files_only"] is True
    assert not hasattr(loaded, "pipeline")
    borrowed = loaded.borrow_pipeline()
    assert borrowed.to_calls == [("cuda:0", torch.float16)]
    assert "dtype" not in borrowed.ip_calls[0][1]
    with pytest.raises(TypeError):
        copy.copy(loaded)
    with pytest.raises(TypeError):
        copy.deepcopy(loaded)
    with pytest.raises(TypeError):
        pickle.dumps(loaded)
    loaded.close()
    loaded.close()
    with pytest.raises(DomainError, match="closed"):
        loaded.borrow_pipeline()
    assert torch.cuda.empty_cache_calls >= 1
    supply.close()


@pytest.mark.parametrize("mutate", ["rocm", "torch", "device"])
def test_rejects_unmatched_runtime_before_loading(tmp_path: Path, mutate: str) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()
    environment = _environment()
    if mutate == "rocm":
        environment = EnvironmentSnapshot(
            environment.schema_version,
            environment.os_name,
            environment.os_release,
            environment.kernel_version,
            environment.machine,
            environment.python_implementation,
            environment.python_version,
            _available("7.2.0"),
            environment.hip_version,
            environment.pytorch_version,
            environment.diffusers_version,
            environment.hip_devices,
        )
    elif mutate == "torch":
        torch.__version__ = "other"
    else:
        torch.cuda.device_count = lambda: 2  # type: ignore[method-assign]
    with pytest.raises(DomainError):
        load_production_pipeline(
            supply, graph, environment, torch_module=torch, diffusers_module=diffusers
        )
    assert not diffusers.control_calls
    supply.close()


def test_rejects_supply_graph_mismatch_before_loading(tmp_path: Path) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()
    graph = replace(graph, base=replace(graph.base, model_id="other"))
    with pytest.raises(DomainError, match="mismatch"):
        load_production_pipeline(
            supply,
            graph,
            _environment(),
            torch_module=torch,
            diffusers_module=diffusers,
        )
    assert not diffusers.control_calls
    supply.close()


@pytest.mark.parametrize("field", ["name", "memory", "arch"])
def test_rejects_mismatch_on_second_reported_device(tmp_path: Path, field: str) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()
    torch.cuda = _Cuda((("AMD test", 16, "gfx1100"), ("AMD second", 32, "gfx1200")))
    second = DeviceSnapshot(
        1,
        _available("wrong" if field == "name" else "AMD second"),
        IntegerObservation("AVAILABLE", 33 if field == "memory" else 32, None),
        _available("wrong" if field == "arch" else "gfx1200"),
    )
    environment = _environment()
    environment = EnvironmentSnapshot(
        environment.schema_version,
        environment.os_name,
        environment.os_release,
        environment.kernel_version,
        environment.machine,
        environment.python_implementation,
        environment.python_version,
        environment.rocm_version,
        environment.hip_version,
        environment.pytorch_version,
        environment.diffusers_version,
        DeviceInventory("AVAILABLE", None, (*environment.hip_devices.devices, second)),
    )
    with pytest.raises(DomainError, match="device|environment"):
        load_production_pipeline(
            supply, graph, environment, torch_module=torch, diffusers_module=diffusers
        )
    supply.close()


def test_cleans_gpu_cache_after_pipeline_stage_failure(tmp_path: Path) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()
    diffusers.StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained = classmethod(
        lambda cls, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(Exception, match="pipeline loading failed"):
        load_production_pipeline(
            supply,
            graph,
            _environment(),
            torch_module=torch,
            diffusers_module=diffusers,
        )
    assert torch.cuda.empty_cache_calls >= 1
    assert supply.borrow_component("base").model_id == "base"
    supply.close()


@pytest.mark.parametrize("stage", ["control", "base", "scheduler", "to", "ip"])
def test_loader_stage_failure_has_no_exception_chain_and_keeps_supply_open(
    tmp_path: Path, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()
    if stage == "control":
        diffusers.ControlNetModel.from_pretrained = classmethod(
            lambda cls, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    elif stage == "base":
        diffusers.StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained = (
            classmethod(
                lambda cls, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
            )
        )
    elif stage == "scheduler":
        diffusers.EulerDiscreteScheduler.from_config = classmethod(
            lambda cls, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        )
    elif stage == "to":
        monkeypatch.setattr(
            _Pipeline,
            "to",
            lambda self, *args: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    else:
        monkeypatch.setattr(
            _Pipeline,
            "load_ip_adapter",
            lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    with pytest.raises(InfrastructureError, match="pipeline loading failed") as raised:
        load_production_pipeline(
            supply,
            graph,
            _environment(),
            torch_module=torch,
            diffusers_module=diffusers,
        )
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert supply.borrow_component("controlnet").model_id == "cn"
    assert torch.cuda.empty_cache_calls >= 1
    supply.close()


def test_close_failure_detaches_exception_and_releases_cache(tmp_path: Path) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()
    loaded = load_production_pipeline(
        supply, graph, _environment(), torch_module=torch, diffusers_module=diffusers
    )
    pipeline = loaded.borrow_pipeline()
    pipeline.maybe_free_model_hooks = lambda: (_ for _ in ()).throw(
        RuntimeError("boom")
    )
    with pytest.raises(InfrastructureError, match="pipeline release failed") as raised:
        loaded.close()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert torch.cuda.empty_cache_calls >= 1
    with pytest.raises(DomainError, match="closed"):
        loaded.borrow_pipeline()
    supply.close()


def test_close_failure_collects_pipeline_and_hook_frame_before_empty_cache(
    tmp_path: Path,
) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()
    loaded = load_production_pipeline(
        supply, graph, _environment(), torch_module=torch, diffusers_module=diffusers
    )
    borrowed = loaded.borrow_pipeline()
    pipeline_ref = weakref.ref(borrowed)

    class FrameSentinel:
        pass

    sentinel_refs: list[weakref.ReferenceType[FrameSentinel]] = []
    cache_observations: list[tuple[bool, bool]] = []

    def failing_hook() -> None:
        sentinel = FrameSentinel()
        sentinel_refs.append(weakref.ref(sentinel))
        try:
            raise RuntimeError("boom")
        except Exception as error:
            retained_error = [error]
            assert retained_error[0] is error
            raise

    def empty_cache() -> None:
        cache_observations.append(
            (pipeline_ref() is None, all(ref() is None for ref in sentinel_refs))
        )
        assert cache_observations[-1] == (True, True)
        torch.cuda.empty_cache_calls += 1

    borrowed.maybe_free_model_hooks = failing_hook
    del borrowed
    torch.cuda.empty_cache = empty_cache

    try:
        with pytest.raises(
            InfrastructureError, match="pipeline release failed"
        ) as raised:
            loaded.close()
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert pipeline_ref() is None
        assert sentinel_refs and all(ref() is None for ref in sentinel_refs)
        assert cache_observations == [(True, True)]
        assert torch.cuda.empty_cache_calls == 1
        with pytest.raises(DomainError, match="closed"):
            loaded.borrow_pipeline()
    finally:
        loaded.close()
        supply.close()


def test_failed_load_releases_pipeline_before_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supply, graph = _supply(tmp_path)
    torch, diffusers = _Torch(), _Diffusers()
    released: list[weakref.ReferenceType[_Pipeline]] = []

    def issue_pipeline(cls, *args, **kwargs):
        pipeline = _Pipeline()
        released.append(weakref.ref(pipeline))
        return pipeline

    monkeypatch.setattr(
        diffusers.StableDiffusionXLControlNetImg2ImgPipeline,
        "from_pretrained",
        classmethod(issue_pipeline),
    )
    monkeypatch.setattr(
        _Pipeline,
        "load_ip_adapter",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    torch.cuda.empty_cache = lambda: (
        (_ for _ in ()).throw(AssertionError("pipeline retained"))
        if released[0]() is not None
        else setattr(torch.cuda, "empty_cache_calls", torch.cuda.empty_cache_calls + 1)
    )
    with pytest.raises(InfrastructureError, match="pipeline loading failed"):
        load_production_pipeline(
            supply,
            graph,
            _environment(),
            torch_module=torch,
            diffusers_module=diffusers,
        )
    assert supply.borrow_component("base").model_id == "base"
    supply.close()
