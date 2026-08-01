"""Disabled legacy loader retained only to fail closed at old call sites.

GEN-004R will introduce the sole production loader over VerifiedPipelineSupply.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from specstyle.errors import DomainError
from specstyle.generation.local_weights import ResolvedWeight
from specstyle.generation.pipeline_factory import PipelineGraph
from specstyle.generation.rocm_probe import RocmProbeResult

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
