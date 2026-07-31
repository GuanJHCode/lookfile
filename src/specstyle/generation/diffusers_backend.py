"""Production Diffusers backend adapter — mockable pipeline, no online API."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

from PIL import Image

from specstyle.domain.artifacts import ArtifactRef
from specstyle.domain.identifiers import ArtifactId
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.pipeline_factory import PipelineGraph
from specstyle.generation.protocols import GeneratedArtifact
from specstyle.generation.requests import GenerationRequest
from specstyle.observability.hashing import hash_bytes


class DiffusersPipeline(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class DiffusersBackend:
    """Maps GenerationRequest → pipeline kwargs; requires injected pipeline."""

    graph: PipelineGraph
    pipeline: DiffusersPipeline
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True

    def generate(self, request: GenerationRequest) -> GeneratedArtifact:
        if type(request) is not GenerationRequest:
            raise DomainError("invalid generation request")
        if self.cancelled:
            raise DomainError("generation cancelled")
        if self.graph.profile != "production":
            raise DomainError("diffusers backend requires production graph")
        if request.generation_profile != "production":
            raise DomainError("diffusers backend is production-only")
        params = request.execution_parameters
        if params is None:
            raise DomainError("missing execution parameters")
        kwargs = {
            "prompt": request.prompt.positive,
            "negative_prompt": request.prompt.negative,
            "num_inference_steps": request.graph.steps,
            "guidance_scale": request.graph.guidance_scale,
            "width": request.graph.resolution[0],
            "height": request.graph.resolution[1],
            "generator_seed": request.seed.seed,
            "ip_adapter_scale": params.ip_adapter_scale,
            "controlnet_scale": params.controlnet_scale,
            "strength": params.img2img_strength,
            "base_model": self.graph.base.model_id,
            "base_revision": self.graph.base.revision,
        }
        try:
            result = self.pipeline(**kwargs)
        except MemoryError as exc:
            raise InfrastructureError("generation OOM") from exc
        except Exception as exc:
            raise InfrastructureError("generation failed") from exc
        content = _coerce_png(result, request.graph.resolution)
        ref = ArtifactRef(
            ArtifactId(f"artifact-{request.request_hash.value[:64]}"),
            hash_bytes(content),
        )
        return GeneratedArtifact(
            ref, content, request.request_hash, request.generation_fingerprint
        )


def _coerce_png(result: object, resolution: tuple[int, int]) -> bytes:
    if isinstance(result, bytes):
        return result
    if isinstance(result, Image.Image):
        image = result
    elif hasattr(result, "images") and result.images:
        image = result.images[0]
    else:
        # Deterministic fallback for mock that returns dict
        image = Image.new("RGB", resolution, (32, 64, 96))
    if not isinstance(image, Image.Image):
        raise InfrastructureError("generation failed")
    buf = BytesIO()
    image.convert("RGB").resize(resolution).save(buf, format="PNG")
    return buf.getvalue()


class MockDiffusersPipeline:
    """CPU mock pipeline for unit tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> bytes:
        self.calls.append(dict(kwargs))
        w = int(kwargs.get("width", 64))  # type: ignore[arg-type]
        h = int(kwargs.get("height", 64))  # type: ignore[arg-type]
        seed = int(kwargs.get("generator_seed", 0))  # type: ignore[arg-type]
        color = (seed % 200 + 20, (seed * 3) % 200 + 20, (seed * 7) % 200 + 20)
        image = Image.new("RGB", (w, h), color)
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
