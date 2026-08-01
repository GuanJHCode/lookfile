"""APP-COMPOSE-001A production generation composition contracts."""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from specstyle.domain.artifacts import AssetRef
from specstyle.domain.identifiers import AssetId, AttemptId, Identifier, JobId
from specstyle.errors import DomainError
from specstyle.generation.preprocess import (
    PreprocessPlan,
    PreparedImage,
    preprocess_image,
)
from specstyle.generation.requests import RenderedPrompt
from specstyle.observability.hashing import hash_bytes
from specstyle.spec.compiled_models import ResourcePin
from specstyle.workflow.job_store import JobStore


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
        "specstyle.verification",
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


class _UnusedControlBuilder:
    def build(self, _source: object, _graph: object) -> object:
        raise AssertionError("control input must not be built while opening")


def test_private_factory_loads_and_owns_only_the_loaded_pipeline(
    tmp_path: Path,
) -> None:
    module = _request_type()[0]
    factory = getattr(module, "_open_production_generation_runtime", None)
    assert callable(factory)

    from tests.unit.generation.test_diffusers_loader import (
        _Diffusers,
        _Torch,
        _environment,
        _supply,
    )
    from tests.unit.spec.test_compiler import context

    supply, graph = _supply(tmp_path / "weights")
    store_root = tmp_path / "store"
    store_root.mkdir()
    runtime = factory(
        supply,
        graph,
        _environment(),
        context(),
        lambda _reference: b"",
        _UnusedControlBuilder(),
        JobStore(store_root),
        torch_module=_Torch(),
        diffusers_module=_Diffusers(),
    )
    pipeline = runtime._loaded.borrow_pipeline()

    runtime.close()
    runtime.close()

    assert pipeline.hooks == 1
    assert supply.borrow_component("base").model_id == "base"
    supply.close()


def test_private_factory_signature_remains_frozen() -> None:
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
        "job_store",
        "torch_module",
        "diffusers_module",
        "clock",
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in parameters[:7]
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters[7:]
    )
    assert parameters[-1].default is module._utc_now
