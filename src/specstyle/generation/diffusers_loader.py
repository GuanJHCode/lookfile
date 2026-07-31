"""Load production Diffusers pipeline from local cache only.

Never downloads. Requires ROCm/HIP when using real torch. On CPU CI the loader
fails closed unless a pipeline factory callable is injected for tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.local_weights import ResolvedWeight, resolve_weight
from specstyle.generation.pipeline_factory import PipelineGraph
from specstyle.generation.rocm_probe import RocmProbeResult, probe_rocm, require_rocm

PipelineBuilder = Callable[[PipelineGraph, tuple[ResolvedWeight, ...]], Any]


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
    """Resolve pins and environment before any heavy import."""
    if type(graph) is not PipelineGraph or not isinstance(cache_root, Path):
        raise DomainError("invalid load plan inputs")
    if type(relpaths) is not dict:
        raise DomainError("invalid load plan relpaths")
    if graph.profile not in ("production", "preview"):
        raise DomainError("invalid graph profile")
    rocm = probe_rocm(torch_module)
    if require_gpu:
        require_rocm(rocm)
    base = resolve_weight(cache_root, graph.base, _need(relpaths, graph.base.model_id))
    ip = resolve_weight(
        cache_root, graph.ip_adapter, _need(relpaths, graph.ip_adapter.model_id)
    )
    cn = resolve_weight(
        cache_root, graph.controlnet, _need(relpaths, graph.controlnet.model_id)
    )
    preview: ResolvedWeight | None = None
    if graph.profile == "preview":
        if graph.preview_adapter is None:
            raise DomainError("preview graph missing adapter")
        preview = resolve_weight(
            cache_root,
            graph.preview_adapter,
            _need(relpaths, graph.preview_adapter.model_id),
        )
    return LoadPlan(graph, base, ip, cn, preview, rocm)


def load_pipeline(
    plan: LoadPlan,
    *,
    builder: PipelineBuilder | None = None,
    local_files_only: bool = True,
) -> Any:
    """Materialize pipeline. Default builder uses Diffusers local_files_only."""
    if type(plan) is not LoadPlan:
        raise DomainError("invalid load plan")
    if not local_files_only:
        raise DomainError("remote model download forbidden")
    weights = (plan.base, plan.ip_adapter, plan.controlnet)
    if plan.preview is not None:
        weights = (*weights, plan.preview)
    if builder is not None:
        try:
            return builder(plan.graph, weights)
        except DomainError:
            raise
        except Exception as exc:
            raise InfrastructureError("pipeline builder failed") from exc
    return _default_diffusers_builder(plan, weights)


def _need(relpaths: dict[str, str], model_id: str) -> str:
    if model_id not in relpaths:
        raise DomainError("missing weights relpath")
    return relpaths[model_id]


def _default_diffusers_builder(
    plan: LoadPlan, weights: tuple[ResolvedWeight, ...]
) -> Any:
    """Attempt real Diffusers load; fail closed without deps or GPU."""
    if not plan.rocm.available:
        raise InfrastructureError("rocm required for default diffusers load")
    try:
        import torch  # type: ignore
        from diffusers import (  # type: ignore
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
        )
    except Exception as exc:
        raise InfrastructureError("diffusers_or_torch_unavailable") from exc
    base_path = str(plan.base.path)
    cn_path = str(plan.controlnet.path)
    try:
        controlnet = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=torch.float16, local_files_only=True
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        # IP-Adapter weights path is recorded; binding is host-specific.
        pipe.to("cuda")
        return pipe
    except DomainError:
        raise
    except Exception as exc:
        raise InfrastructureError("diffusers local load failed") from exc
