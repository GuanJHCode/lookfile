"""GEN-003..006 + ROCM probe + SEC registry — shipped entry points."""

from __future__ import annotations

from pathlib import Path

import pytest

from specstyle.domain.identifiers import Sha256
from specstyle.errors import DomainError
from specstyle.generation.diffusers_backend import (
    DiffusersBackend,
    MockDiffusersPipeline,
)
from specstyle.generation.model_registry import ModelDescriptor, ModelRegistry
from specstyle.generation.output_profiles import PROFILES, render_output_profile
from specstyle.generation.pipeline_factory import PipelineFactory
from specstyle.generation.preview import StrengthMapping, map_user_strength
from specstyle.generation.rocm_probe import probe_rocm, require_rocm


def _sha(c: str = "a") -> Sha256:
    return Sha256(c * 64)


_REVISION = "a" * 40


def _registry() -> ModelRegistry:
    fam = "sdxl"
    return ModelRegistry(
        (
            ModelDescriptor(
                "base1", "base", _REVISION, _sha("1"), "Apache-2.0", "APPROVED", fam
            ),
            ModelDescriptor(
                "ip1", "ip_adapter", _REVISION, _sha("2"), "Apache-2.0", "APPROVED", fam
            ),
            ModelDescriptor(
                "cn1", "controlnet", _REVISION, _sha("3"), "Apache-2.0", "APPROVED", fam
            ),
            ModelDescriptor(
                "prev1",
                "preview_adapter",
                _REVISION,
                _sha("4"),
                "Apache-2.0",
                "APPROVED",
                fam,
            ),
            ModelDescriptor(
                "bad", "base", _REVISION, _sha("5"), "Unknown", "UNKNOWN", fam
            ),
        )
    )


def test_registry_blocks_unknown_and_floating() -> None:
    reg = _registry()
    with pytest.raises(DomainError, match="UNKNOWN"):
        reg.require_production("bad")
    with pytest.raises(DomainError):
        ModelDescriptor("x", "base", "main", _sha("a"), "MIT", "APPROVED", "sdxl")


def test_pipeline_factory_family_gate(tmp_path: Path) -> None:
    factory = PipelineFactory(_registry(), tmp_path / "cache")
    graph = factory.build_production("base1", "ip1", "cn1")
    assert graph.profile == "production"
    assert graph.cache_root == "cache"
    preview = factory.build_preview("base1", "ip1", "cn1", "prev1")
    assert preview.preview_adapter is not None


def test_rocm_probe_honest_without_gpu() -> None:
    result = probe_rocm(None)
    # On this Mac, likely unavailable — never forge True without hip.
    if not result.available:
        with pytest.raises(DomainError, match="rocm not available"):
            require_rocm(result)
    assert result.snapshot_hash


def test_strength_mapping_monotone() -> None:
    m = StrengthMapping("map-v1", 4, 30, 0.0, 5.0)
    low = map_user_strength(0.2, m, "production")
    high = map_user_strength(0.8, m, "production")
    assert low["ip_adapter_scale"] < high["ip_adapter_scale"]  # type: ignore[operator]
    assert low["steps"] == 30


def test_output_profiles_sizes_and_deterministic() -> None:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buf, format="PNG")
    src = buf.getvalue()
    for name, layout in PROFILES.items():
        kwargs = {}
        if name == "background_sequence":
            kwargs["sequence_index"] = 0
        a = render_output_profile(src, name, text="hello", **kwargs)
        b = render_output_profile(src, name, text="hello", **kwargs)
        assert a == b
        img = Image.open(BytesIO(a))
        assert img.size == layout.size


def test_mock_diffusers_backend_parameter_mapping() -> None:
    from tests.unit.exporting.test_manifest import _production_request

    request = _production_request()
    pipe = MockDiffusersPipeline()
    reg = _registry()
    factory = PipelineFactory(reg, Path("cache"))
    graph = factory.build_production("base1", "ip1", "cn1")
    backend = DiffusersBackend(graph, pipe)
    artifact = backend.generate(request)
    assert artifact.content.startswith(b"\x89PNG")
    assert pipe.calls
    assert pipe.calls[0]["generator_seed"] == request.seed.seed


def test_family_mismatch_blocked() -> None:
    reg = ModelRegistry(
        (
            ModelDescriptor(
                "base1", "base", _REVISION, _sha("1"), "Apache-2.0", "APPROVED", "sdxl"
            ),
            ModelDescriptor(
                "ip1",
                "ip_adapter",
                _REVISION,
                _sha("2"),
                "Apache-2.0",
                "APPROVED",
                "other",
            ),
            ModelDescriptor(
                "cn1",
                "controlnet",
                _REVISION,
                _sha("3"),
                "Apache-2.0",
                "APPROVED",
                "sdxl",
            ),
        )
    )
    factory = PipelineFactory(reg, Path("cache"))
    with pytest.raises(DomainError, match="family"):
        factory.build_production("base1", "ip1", "cn1")


def test_blocked_license_not_production() -> None:
    reg = ModelRegistry(
        (
            ModelDescriptor(
                "base1", "base", _REVISION, _sha("1"), "Proprietary", "BLOCKED", "sdxl"
            ),
        )
    )
    with pytest.raises(DomainError, match="BLOCKED"):
        reg.require_production("base1")


def test_preview_backend_rejects_production_profile() -> None:
    from specstyle.generation.pipeline_factory import PipelineGraph
    from specstyle.generation.preview import PreviewBackend, StrengthMapping
    from tests.unit.exporting.test_manifest import _production_request

    reg = _registry()
    factory = PipelineFactory(reg, Path("cache"))
    g = factory.build_preview("base1", "ip1", "cn1", "prev1")
    pgraph = PipelineGraph(
        "preview", g.base, g.ip_adapter, g.controlnet, g.preview_adapter, g.cache_root
    )

    class Stub:
        def generate(self, request):  # pragma: no cover
            raise AssertionError("should not be called")

    prev = PreviewBackend(pgraph, Stub(), StrengthMapping("map-v1", 4, 30, 0.0, 5.0))
    req = _production_request()
    with pytest.raises(DomainError, match="preview"):
        prev.generate(req)
