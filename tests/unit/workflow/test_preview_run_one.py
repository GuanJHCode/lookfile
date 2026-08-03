from __future__ import annotations

import copy
import os
from pathlib import Path
import pickle

import pytest

from specstyle.domain.identifiers import ArtifactId, Sha256
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.gpu_runtime_lane import acquire_gpu_runtime_lane
from specstyle.workflow.preview_evidence import PreviewEvidencePublication


def _publication(run_id: str) -> PreviewEvidencePublication:
    return PreviewEvidencePublication(
        run_id,
        f"{run_id}-{'a' * 16}.png",
        ArtifactId(f"preview-{'b' * 64}"),
        Sha256("a" * 64),
        Sha256("c" * 64),
    )


def _fds(tmp_path: Path):
    from specstyle.workflow.preview_run_one import PreviewRunOneFds

    roots = []
    for name in (
        "production-config",
        "production-evidence",
        "preview-config",
        "models",
        "preview-evidence",
        "display",
        "styles",
    ):
        path = tmp_path / name
        path.mkdir(mode=0o700)
        roots.append(os.open(path, os.O_RDONLY | os.O_DIRECTORY))
    files = []
    for name in ("source", "style", "spec", "metadata"):
        path = tmp_path / name
        path.write_bytes(b"input")
        path.chmod(0o600)
        files.append(os.open(path, os.O_RDONLY))
    return PreviewRunOneFds(*roots, *files), roots + files


def test_busy_is_immediate_and_does_not_consume_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.workflow.preview_run_one as module

    fds, descriptors = _fds(tmp_path)
    reservation = module.reserve_preview_run_one()
    monkeypatch.setattr(
        module,
        "_execute_preview",
        lambda *_args: pytest.fail("busy request must not execute"),
    )
    lease = acquire_gpu_runtime_lane()
    try:
        result = module.run_preview_one(fds, reservation)
    finally:
        lease.close()
    assert result.status is module.PreviewRunStatus.BUSY
    assert result.run_id == reservation.run_id
    assert result.publication is None
    assert result.verification == result.repair == result.export == "NOT_RUN"

    monkeypatch.setattr(
        module,
        "_execute_preview",
        lambda _owned, run_id, _variation: _publication(run_id),
    )
    completed = module.run_preview_one(fds, reservation)
    assert completed.status is module.PreviewRunStatus.COMPLETED
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="reservations"):
            operation(reservation)
    with pytest.raises(DomainError, match="already consumed"):
        module.run_preview_one(fds, reservation)
    for descriptor in descriptors:
        os.close(descriptor)


def test_completed_becomes_visible_only_after_publication_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.workflow.preview_run_one as module
    from specstyle.generation.gpu_runtime_lane import try_acquire_gpu_runtime_lane

    fds, descriptors = _fds(tmp_path)
    events: list[str] = []

    def execute(_owned: tuple[int, ...], run_id: str, _variation: int):
        assert try_acquire_gpu_runtime_lane() is None
        events.append("published")
        return _publication(run_id)

    monkeypatch.setattr(module, "_execute_preview", execute)
    result = module.run_preview_one(fds, module.reserve_preview_run_one(2))
    events.append("returned")
    assert result.status is module.PreviewRunStatus.COMPLETED
    assert result.reason_code == "OK"
    assert result.publication is not None
    assert events == ["published", "returned"]
    next_lease = try_acquire_gpu_runtime_lane()
    assert next_lease is not None
    next_lease.close()
    for descriptor in descriptors:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("status", "reason"),
    (("UNAVAILABLE", "PREVIEW_CONFIG_INVALID"), ("FAILED", "PERSIST_FAILED")),
)
def test_failures_are_explicit_never_approved_and_release_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    reason: str,
) -> None:
    import specstyle.workflow.preview_run_one as module
    from specstyle.generation.gpu_runtime_lane import try_acquire_gpu_runtime_lane

    fds, descriptors = _fds(tmp_path)

    def fail(*_args: object):
        raise module._PreviewRunFailure(module.PreviewRunStatus(status), reason)

    monkeypatch.setattr(module, "_execute_preview", fail)
    result = module.run_preview_one(fds, module.reserve_preview_run_one())
    assert result.status.value == status
    assert result.reason_code == reason
    assert result.publication is None
    assert "APPROVED" not in repr(result)
    next_lease = try_acquire_gpu_runtime_lane()
    assert next_lease is not None
    next_lease.close()
    for descriptor in descriptors:
        os.close(descriptor)


def test_persist_failure_closes_all_runtime_resources_before_lane_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.workflow.preview_run_one as module
    from specstyle.generation.gpu_runtime_lane import try_acquire_gpu_runtime_lane

    fds, descriptors = _fds(tmp_path)
    events: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            assert try_acquire_gpu_runtime_lane() is None
            events.append(f"close:{self.name}")

    preflight = module._PreviewPreflight(
        object(), object(), object(), Resource("supply"), Resource("adapter"), object()
    )
    job_input = Resource("input")
    job_input.style_assets = object()
    loaded = Resource("pipeline")
    monkeypatch.setattr(module, "_open_preflight", lambda _owned: preflight)
    monkeypatch.setattr(module, "_open_input", lambda *_args: job_input)
    monkeypatch.setattr(module, "_compile_request", lambda *_args: object())
    monkeypatch.setattr(module, "load_preview_pipeline", lambda *_args: loaded)
    monkeypatch.setattr(
        module,
        "PreviewDiffusersBackend",
        lambda *_args: type(
            "Backend", (), {"generate": lambda self, _request: object()}
        )(),
    )
    monkeypatch.setattr(
        module,
        "publish_preview_evidence",
        lambda *_args: (_ for _ in ()).throw(
            InfrastructureError("preview display unavailable")
        ),
    )

    result = module.run_preview_one(fds, module.reserve_preview_run_one())
    assert result.status is module.PreviewRunStatus.FAILED
    assert result.reason_code == "PERSIST_FAILED"
    assert events == ["close:pipeline", "close:input", "close:adapter", "close:supply"]
    next_lease = try_acquire_gpu_runtime_lane()
    assert next_lease is not None
    next_lease.close()
    for descriptor in descriptors:
        os.close(descriptor)


def test_cleanup_failure_closes_remaining_resources_and_releases_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.workflow.preview_run_one as module
    from specstyle.generation.gpu_runtime_lane import try_acquire_gpu_runtime_lane

    fds, descriptors = _fds(tmp_path)
    events: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            events.append(f"close:{self.name}")
            if self.fail:
                raise RuntimeError("cleanup failed")

    preflight = module._PreviewPreflight(
        object(),
        object(),
        object(),
        Resource("supply"),
        Resource("adapter"),
        object(),
    )
    job_input = Resource("input")
    job_input.style_assets = object()
    loaded = Resource("pipeline", fail=True)
    monkeypatch.setattr(module, "_open_preflight", lambda _owned: preflight)
    monkeypatch.setattr(module, "_open_input", lambda *_args: job_input)
    monkeypatch.setattr(module, "_compile_request", lambda *_args: object())
    monkeypatch.setattr(module, "load_preview_pipeline", lambda *_args: loaded)
    monkeypatch.setattr(
        module,
        "PreviewDiffusersBackend",
        lambda *_args: type(
            "Backend", (), {"generate": lambda self, _request: object()}
        )(),
    )
    monkeypatch.setattr(
        module,
        "publish_preview_evidence",
        lambda _private, _display, run_id, _artifact: _publication(run_id),
    )

    result = module.run_preview_one(fds, module.reserve_preview_run_one())
    assert result.status is module.PreviewRunStatus.FAILED
    assert result.reason_code == "CLEANUP_FAILED"
    assert events == ["close:pipeline", "close:input", "close:adapter", "close:supply"]
    next_lease = try_acquire_gpu_runtime_lane()
    assert next_lease is not None
    next_lease.close()
    for descriptor in descriptors:
        os.close(descriptor)


def test_invalid_preview_config_isolated_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.workflow.preview_run_one as module

    fds, descriptors = _fds(tmp_path)
    monkeypatch.setattr(
        module, "load_production_context_config", lambda *_args: object()
    )
    monkeypatch.setattr(
        module, "load_production_supply_config", lambda *_args: object()
    )
    monkeypatch.setattr(
        module,
        "load_preview_supply_config",
        lambda *_args: (_ for _ in ()).throw(DomainError("bad preview config")),
    )
    result = module.run_preview_one(fds, module.reserve_preview_run_one())
    assert result.status is module.PreviewRunStatus.UNAVAILABLE
    assert result.reason_code == "PREVIEW_CONFIG_INVALID"
    assert result.publication is None
    for descriptor in descriptors:
        os.close(descriptor)
