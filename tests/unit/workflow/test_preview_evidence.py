from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from specstyle.errors import DomainError, InfrastructureError
from specstyle.observability.hashing import hash_bytes


def _artifact(tmp_path: Path):
    from specstyle.generation.preview_diffusers_loader import load_preview_pipeline
    from specstyle.generation.preview_execution import (
        bind_preview_execution,
        build_preview_artifact,
    )
    from tests.unit.generation.test_preview_diffusers_runtime import (
        _Peft,
        _PreviewDiffusers,
        _Torch,
        _png,
        _runtime,
    )
    from tests.unit.generation.test_diffusers_loader import _environment

    supply, adapter, graph, request = _runtime(tmp_path / "models")
    loaded = load_preview_pipeline(
        supply,
        adapter,
        graph,
        _environment(),
        torch_module=_Torch(),
        diffusers_module=_PreviewDiffusers(),
        peft_module=_Peft(),
    )
    try:
        binding = bind_preview_execution(loaded, request)
        return build_preview_artifact(_png((512, 512), "green"), binding)
    finally:
        loaded.close()
        adapter.close()
        supply.close()


def _roots(tmp_path: Path) -> tuple[int, int, Path, Path]:
    private = tmp_path / "private"
    display = tmp_path / "display"
    private.mkdir(mode=0o700)
    display.mkdir(mode=0o700)
    return (
        os.open(private, os.O_RDONLY | os.O_DIRECTORY),
        os.open(display, os.O_RDONLY | os.O_DIRECTORY),
        private,
        display,
    )


def test_preview_evidence_atomically_publishes_private_pair_then_display(
    tmp_path: Path,
) -> None:
    from specstyle.workflow.preview_evidence import publish_preview_evidence

    artifact = _artifact(tmp_path)
    private_fd, display_fd, private, display = _roots(tmp_path)
    try:
        published = publish_preview_evidence(
            private_fd, display_fd, "preview-run-1", artifact
        )
    finally:
        os.close(display_fd)
        os.close(private_fd)

    evidence = private / published.evidence_name
    assert sorted(item.name for item in evidence.iterdir()) == [
        "artifact.png",
        "record.json",
    ]
    record = json.loads((evidence / "record.json").read_text(encoding="utf-8"))
    display_bytes = (display / published.display_name).read_bytes()
    assert record["schema_version"] == "specstyle.preview.evidence.v1"
    assert record["run_id"] == "preview-run-1"
    assert record["artifact"]["content_sha256"] == hash_bytes(display_bytes).value
    assert record["artifact"]["execution_fingerprint"] == (
        artifact.execution_fingerprint.value
    )
    assert record["planes"] == {
        "verification": "NOT_RUN",
        "repair": "NOT_RUN",
        "export": "NOT_RUN",
    }
    assert "APPROVED" not in (evidence / "record.json").read_text()
    assert published.content_sha256 == hash_bytes(display_bytes)


def test_display_failure_never_returns_completed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.workflow.preview_evidence as module

    artifact = _artifact(tmp_path)
    private_fd, display_fd, private, display = _roots(tmp_path)
    monkeypatch.setattr(
        module,
        "_publish_display",
        lambda *_args: (_ for _ in ()).throw(
            InfrastructureError("preview display unavailable")
        ),
    )
    try:
        with pytest.raises(InfrastructureError, match="display"):
            module.publish_preview_evidence(
                private_fd, display_fd, "preview-run-display-fail", artifact
            )
    finally:
        os.close(display_fd)
        os.close(private_fd)
    assert (private / "preview-run-display-fail" / "record.json").is_file()
    assert list(display.iterdir()) == []


def test_preview_evidence_is_no_replace_and_display_orphans_are_removed(
    tmp_path: Path,
) -> None:
    from specstyle.workflow.preview_evidence import (
        publish_preview_evidence,
        reconcile_preview_display,
    )

    artifact = _artifact(tmp_path)
    private_fd, display_fd, _private, display = _roots(tmp_path)
    try:
        published = publish_preview_evidence(
            private_fd, display_fd, "preview-run-stable", artifact
        )
        with pytest.raises(DomainError, match="already exists"):
            publish_preview_evidence(
                private_fd, display_fd, "preview-run-stable", artifact
            )
        orphan = display / "orphan.png"
        orphan.write_bytes(artifact.content)
        orphan.chmod(0o600)
        removed = reconcile_preview_display(private_fd, display_fd)
    finally:
        os.close(display_fd)
        os.close(private_fd)
    assert removed == ("orphan.png",)
    assert not orphan.exists()
    assert (display / published.display_name).is_file()


def test_reconcile_unlinks_display_symlink_without_following_target(
    tmp_path: Path,
) -> None:
    from specstyle.workflow.preview_evidence import reconcile_preview_display

    private_fd, display_fd, _private, display = _roots(tmp_path)
    target = tmp_path / "outside.png"
    target.write_bytes(b"must remain")
    link = display / "orphan-link.png"
    link.symlink_to(target)
    try:
        removed = reconcile_preview_display(private_fd, display_fd)
    finally:
        os.close(display_fd)
        os.close(private_fd)
    assert removed == ("orphan-link.png",)
    assert not link.exists()
    assert target.read_bytes() == b"must remain"
