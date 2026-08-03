from __future__ import annotations

import copy
from dataclasses import fields
import os
import pickle
from types import SimpleNamespace

import pytest

from specstyle.domain.identifiers import JobId
from specstyle.errors import DomainError


class _HostileInt(int):
    pass


def test_fds_have_the_frozen_11_descriptor_boundary() -> None:
    from specstyle.workflow.run_one import ProductionRunOneFds

    assert tuple(field.name for field in fields(ProductionRunOneFds)) == (
        "config_root_fd",
        "evidence_root_fd",
        "model_root_fd",
        "state_root_fd",
        "artifact_root_fd",
        "style_asset_root_fd",
        "export_root_fd",
        "source_fd",
        "style_fd",
        "spec_fd",
        "metadata_fd",
    )
    descriptors = [os.open("/dev/null", os.O_RDONLY) for _ in range(11)]
    try:
        instance = ProductionRunOneFds(*descriptors)
        assert instance.config_root_fd == descriptors[0]
        with pytest.raises(Exception):
            ProductionRunOneFds(True, *descriptors[1:])
        with pytest.raises(Exception):
            ProductionRunOneFds(-1, *descriptors[1:])
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_run_aligns_pre_export_job_state_with_completed_export_state() -> None:
    """runtime.run() is pre-export; export_result is COMPLETED — public result must share state."""
    from dataclasses import replace

    from specstyle.domain.identifiers import AttemptId, JobId as RealJobId, Sha256
    from specstyle.workflow.job_models import Job, JobBudget, JobState, JobStatus
    from specstyle.workflow.production_service import ProductionJobResult

    job_id = RealJobId("run-one-test")
    budget = JobBudget(2)
    ts = "2026-08-03T00:00:00.000Z"
    pre_export = JobState(
        Job(
            job_id, Sha256("a" * 64), ("xhs_grid",), budget, JobStatus.APPROVED, ts, ts
        ),
        5,
        (AttemptId("run-one-test-a0-xhs_grid-0"),),
        (),
    )
    completed = JobState(
        Job(
            job_id,
            Sha256("a" * 64),
            ("xhs_grid",),
            budget,
            JobStatus.COMPLETED,
            ts,
            "2026-08-03T00:00:01.000Z",
        ),
        7,
        (AttemptId("run-one-test-a0-xhs_grid-0"),),
        ("bundle-run-one-test",),
    )
    job_result = object.__new__(ProductionJobResult)
    for name, value in (
        ("compiled", object()),
        ("graph", object()),
        ("verification_plan", object()),
        ("request", object()),
        ("artifact", object()),
        ("report", object()),
        ("history", object()),
        ("terminal", object()),
        ("job_state", pre_export),
    ):
        object.__setattr__(job_result, name, value)

    aligned = replace(job_result, job_state=completed)
    assert aligned.job_state is completed or aligned.job_state == completed
    assert aligned.job_state.job.status is JobStatus.COMPLETED
    assert aligned.job_state.bundle_names == ("bundle-run-one-test",)
    assert job_result.job_state.job.status is JobStatus.APPROVED


def test_reservation_is_fresh_and_cannot_be_serialized() -> None:
    from specstyle.workflow.run_one import reserve_production_run_one

    reservation = reserve_production_run_one()

    assert type(reservation.job_id) is JobId
    assert reservation.variation_index == 0
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            operation(reservation)


@pytest.mark.parametrize("variation_index", (True, _HostileInt(0), -1, 2**31))
def test_reservation_rejects_noncanonical_variation(variation_index: object) -> None:
    from specstyle.workflow.run_one import reserve_production_run_one

    with pytest.raises(DomainError, match="^invalid production run-one input$"):
        reserve_production_run_one(variation_index)  # type: ignore[arg-type]


def test_reservation_revalidates_variation_when_consumed() -> None:
    from specstyle.workflow.run_one import reserve_production_run_one

    reservation = reserve_production_run_one(7)
    reservation._variation_index = True

    with pytest.raises(DomainError, match="^invalid production run-one input$"):
        reservation._consume()


def test_open_duplicates_borrowed_fds_and_uses_the_shared_config_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import specstyle.workflow.run_one as module

    root_paths = []
    for index in range(7):
        path = tmp_path / f"root-{index}"
        path.mkdir()
        root_paths.append(path)
    roots = [os.open(path, os.O_RDONLY | os.O_DIRECTORY) for path in root_paths]
    file_paths = []
    for index in range(4):
        path = tmp_path / f"file-{index}.bin"
        path.write_bytes(b"x")
        file_paths.append(path)
    files = [os.open(path, os.O_RDONLY) for path in file_paths]
    fds = module.ProductionRunOneFds(*roots, *files)
    seen: dict[str, object] = {}
    closed: list[str] = []

    class _Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    input_value = SimpleNamespace(
        request=object(), style_assets=lambda _ref: object(), asset_credits=()
    )
    input_value.close = _Resource("input").close
    supply = _Resource("supply")
    store = _Resource("store")
    runtime = _Resource("runtime")

    def load_supply(config: int) -> SimpleNamespace:
        seen["supply_config"] = config
        return SimpleNamespace(graph=object(), manifests=(), approvals=())

    class _Context:
        canny = object()

    def load_context_config(config: int, evidence: int) -> _Context:
        seen["context"] = (config, evidence)
        return _Context()

    monkeypatch.setattr(
        module, "load_production_job_input_metadata", lambda fd: object()
    )
    monkeypatch.setattr(module, "load_production_context_config", load_context_config)
    monkeypatch.setattr(module, "load_production_supply_config", load_supply)
    monkeypatch.setattr(module, "verify_pipeline_supply", lambda *_args: supply)
    monkeypatch.setattr(module, "capture_environment", lambda: object())
    monkeypatch.setattr(
        module, "make_production_compiler_context_factory", lambda *_args: object()
    )
    monkeypatch.setattr(module.JobStore, "from_root_fd", lambda _fd: store)

    def open_job_input(*_args: object, **kwargs: object) -> object:
        seen["variation_index"] = kwargs.get("variation_index")
        return input_value

    monkeypatch.setattr(module, "open_production_job_input", open_job_input)
    monkeypatch.setattr(module, "open_production_runtime", lambda *_args: runtime)
    # Avoid importing real canny/cv2: stub the lazy import path inside _open_resources.
    import types
    import sys

    canny_stub = types.ModuleType("specstyle.generation.canny")
    canny_stub.CannyControlInputBuilder = lambda _config: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "specstyle.generation.canny", canny_stub)

    try:
        execution = module.open_production_run_one(
            fds, module.reserve_production_run_one(7)
        )
        assert type(execution.job_id) is JobId
        assert seen["context"][0] == seen["supply_config"]
        assert seen["variation_index"] == 7
        execution.close()
        assert closed == ["runtime", "input", "supply", "store"]
    finally:
        for descriptor in (*roots, *files):
            try:
                os.close(descriptor)
            except OSError:
                pass
