"""SEC-001A candidates, local weights, fail-closed Diffusers loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from specstyle.domain.identifiers import Sha256
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.candidates import (
    approved_production_registry,
    catalog_summary,
    default_candidates,
    with_license_status,
)
from specstyle.generation.diffusers_loader import load_pipeline, plan_local_load
from specstyle.generation.local_weights import resolve_weight
from specstyle.generation.model_registry import ModelDescriptor
from specstyle.generation.pipeline_factory import PipelineFactory
from specstyle.observability.hashing import hash_bytes, hash_file


def _sha(c: str = "a") -> Sha256:
    return Sha256(c * 64)


def test_default_candidates_all_unknown_and_no_production_registry() -> None:
    cats = default_candidates()
    assert cats
    assert all(c.license_status == "UNKNOWN" for c in cats)
    summary = catalog_summary(cats)
    assert len(summary) == len(cats)
    assert all(row["license_status"] == "UNKNOWN" for row in summary)
    with pytest.raises(DomainError, match="no APPROVED"):
        approved_production_registry(cats)


def test_human_approve_builds_registry_only_for_approved() -> None:
    cats = default_candidates()
    # Approve a full production triple with honest human flip.
    for mid in (
        "sdxl-base-1.0",
        "ip-adapter-plus-sdxl",
        "controlnet-canny-sdxl",
    ):
        cats = with_license_status(cats, mid, "APPROVED")
    reg = approved_production_registry(cats)
    assert reg.require_production("sdxl-base-1.0").role == "base"
    with pytest.raises(DomainError):
        # preview still UNKNOWN
        reg.get("lcm-lora-sdxl")


def test_floating_revision_forbidden_in_candidates() -> None:
    from specstyle.generation.candidates import CandidateModel, LicenseEvidence

    with pytest.raises(DomainError, match="floating"):
        CandidateModel(
            "x",
            "base",
            "sdxl",
            "main",
            _sha("1"),
            "UNKNOWN",
            LicenseEvidence(
                "MIT",
                "https://example.com/lic",
                "allowed",
                "allowed",
                "allowed",
                "n",
            ),
            "x/main",
            "test",
        )


def test_resolve_weight_file_hash_and_mismatch(tmp_path: Path) -> None:
    payload = b"weight-bytes-v1"
    rel = "models/base.bin"
    path = tmp_path.joinpath(*rel.split("/"))
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    digest = hash_file(tmp_path, rel)
    desc = ModelDescriptor(
        "base1", "base", "rev1", digest, "Apache-2.0", "APPROVED", "sdxl"
    )
    resolved = resolve_weight(tmp_path, desc, rel)
    assert resolved.kind == "file"
    assert resolved.content_sha256 == digest
    bad = ModelDescriptor(
        "base1", "base", "rev1", _sha("9"), "Apache-2.0", "APPROVED", "sdxl"
    )
    with pytest.raises(InfrastructureError, match="hash mismatch"):
        resolve_weight(tmp_path, bad, rel)


def test_resolve_weight_directory_marker(tmp_path: Path) -> None:
    digest = hash_bytes(b"dir-marker")
    rel = "models/base-dir"
    d = tmp_path.joinpath(*rel.split("/"))
    d.mkdir(parents=True)
    (d / "WEIGHTS.sha256").write_text(digest.value + "\n", encoding="ascii")
    desc = ModelDescriptor(
        "base1", "base", "rev1", digest, "Apache-2.0", "APPROVED", "sdxl"
    )
    resolved = resolve_weight(tmp_path, desc, rel)
    assert resolved.kind == "directory_marker"


def test_resolve_weight_symlink_refused(tmp_path: Path) -> None:
    real = tmp_path / "real.bin"
    real.write_bytes(b"abc")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    desc = ModelDescriptor(
        "base1", "base", "rev1", _sha("1"), "Apache-2.0", "APPROVED", "sdxl"
    )
    with pytest.raises(InfrastructureError, match="symlink"):
        resolve_weight(tmp_path, desc, "link.bin")


def test_plan_local_load_requires_gpu_by_default(tmp_path: Path) -> None:
    payload = b"w"
    paths = {}
    descs = []
    for mid, role, ch in (
        ("base1", "base", "1"),
        ("ip1", "ip_adapter", "2"),
        ("cn1", "controlnet", "3"),
    ):
        rel = f"{mid}.bin"
        p = tmp_path / rel
        p.write_bytes(payload + ch.encode())
        digest = hash_file(tmp_path, rel)
        descs.append(
            ModelDescriptor(mid, role, "rev1", digest, "Apache-2.0", "APPROVED", "sdxl")
        )
        paths[mid] = rel
    from specstyle.generation.model_registry import ModelRegistry

    reg = ModelRegistry(tuple(descs))
    graph = PipelineFactory(reg, tmp_path).build_production("base1", "ip1", "cn1")
    with pytest.raises(DomainError, match="rocm not available"):
        plan_local_load(graph, tmp_path, paths, require_gpu=True)


def test_plan_and_load_with_injected_builder_cpu(tmp_path: Path) -> None:
    paths = {}
    descs = []
    for mid, role, ch in (
        ("base1", "base", "1"),
        ("ip1", "ip_adapter", "2"),
        ("cn1", "controlnet", "3"),
    ):
        rel = f"{mid}.bin"
        (tmp_path / rel).write_bytes(b"w" + ch.encode())
        digest = hash_file(tmp_path, rel)
        descs.append(
            ModelDescriptor(mid, role, "rev1", digest, "Apache-2.0", "APPROVED", "sdxl")
        )
        paths[mid] = rel
    from specstyle.generation.model_registry import ModelRegistry

    reg = ModelRegistry(tuple(descs))
    graph = PipelineFactory(reg, tmp_path).build_production("base1", "ip1", "cn1")
    plan = plan_local_load(graph, tmp_path, paths, require_gpu=False)

    def builder(g, weights):
        assert g is graph
        assert len(weights) == 3
        return {"mock": True, "ids": [w.model_id for w in weights]}

    pipe = load_pipeline(plan, builder=builder)
    assert pipe["mock"] is True


def test_load_pipeline_forbids_remote_download(tmp_path: Path) -> None:
    from specstyle.generation.diffusers_loader import LoadPlan
    from specstyle.generation.local_weights import ResolvedWeight
    from specstyle.generation.pipeline_factory import PipelineGraph
    from specstyle.generation.rocm_probe import RocmProbeResult

    desc = ModelDescriptor(
        "base1", "base", "rev1", _sha("1"), "Apache-2.0", "APPROVED", "sdxl"
    )
    graph = PipelineGraph("production", desc, desc, desc, None, "cache")
    rw = ResolvedWeight("base1", tmp_path, _sha("1"), "file")
    plan = LoadPlan(
        graph,
        rw,
        rw,
        rw,
        None,
        RocmProbeResult(False, None, None, "x", "a" * 64),
    )
    with pytest.raises(DomainError, match="download forbidden"):
        load_pipeline(plan, local_files_only=False)
