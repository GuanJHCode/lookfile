from __future__ import annotations

import json
import hashlib
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from specstyle.domain.artifacts import ArtifactRef
from specstyle.domain.enums import RuleLevel, RuleScope, RuleStatus, StaticApplicability
from specstyle.domain.identifiers import ArtifactId, JobId, RuleId, Sha256
from specstyle.exporting.bundle import ExportedFile
from specstyle.ui.app import UiServices
from specstyle.verification.rule_models import (
    GatePolicy,
    RuleDefinition,
    RuleResult,
    VerificationReport,
)
from tests.unit.spec.test_compiler import raw_spec


class _NamedString(str):
    pass


def _roots(tmp_path: Path):
    from specstyle.ui.production_run import ProductionUiRuntimePaths

    paths = {}
    for name in (
        "config_root",
        "evidence_root",
        "model_root",
        "state_root",
        "artifact_root",
        "style_asset_root",
        "export_root",
        "staging_root",
    ):
        value = tmp_path / name
        value.mkdir()
        paths[name] = value
    return ProductionUiRuntimePaths(**paths)


def _file(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    path.chmod(0o600)
    return str(path)


def _uploads(tmp_path: Path) -> tuple[str, str, str]:
    source = _file(tmp_path / "source.png", b"source-image")
    style = _file(tmp_path / "style.png", b"style-image")
    spec = _file(
        tmp_path / "style.json",
        json.dumps(raw_spec().model_dump(mode="json"), separators=(",", ":")).encode(),
    )
    return source, style, spec


def _qa_report() -> VerificationReport:
    aid = ArtifactId("artifact-1")
    policy = GatePolicy("reject", "reject", "reject")
    rule = RuleDefinition(
        RuleId("l2_style"),
        RuleLevel.L2,
        RuleScope.ITEM,
        True,
        StaticApplicability.APPLICABLE,
        policy,
    )
    return VerificationReport(
        (ArtifactRef(aid, Sha256("b" * 64)),),
        (rule,),
        (RuleResult(RuleId("l2_style"), RuleStatus.UNVERIFIABLE, (aid,), None),),
    )


def test_production_ui_service_runs_one_job_and_returns_export_evidence(
    tmp_path: Path,
) -> None:
    from specstyle.ui.production_run import bind_production_run_one_services

    roots = _roots(tmp_path)
    source_bytes = b"source-image"
    style_bytes = b"style-image"
    spec = raw_spec().model_copy(
        update={
            "assets": raw_spec().assets.model_copy(
                update={
                    "style_references": (
                        raw_spec()
                        .assets.style_references[0]
                        .model_copy(update={"asset_sha256": Sha256("c" * 64).value}),
                    )
                }
            )
        }
    )
    source_path = _NamedString(_file(tmp_path / "source.png", source_bytes))
    style_path = _NamedString(_file(tmp_path / "style.png", style_bytes))
    spec_path = _NamedString(
        _file(
            tmp_path / "style.json",
            json.dumps(spec.model_dump(mode="json"), separators=(",", ":")).encode(),
        )
    )
    approved = roots.export_root / "bundle-run-one-ui" / "approved" / "xhs_grid"
    rejected = roots.export_root / "bundle-run-one-ui" / "rejected"
    approved.mkdir(parents=True)
    rejected.mkdir(parents=True)
    (approved / "artifact-approved.png").write_bytes(b"png")
    (rejected / "artifact-rejected.png").write_bytes(b"png")
    calls: dict[str, object] = {}

    class _Reservation:
        job_id = JobId("run-one-ui")

    class _Execution:
        job_id = JobId("run-one-ui")

        def run(self):
            calls["ran"] = True
            return SimpleNamespace(
                job_result=SimpleNamespace(report=_qa_report()),
                export_result=SimpleNamespace(
                    bundle=SimpleNamespace(
                        bundle_name="bundle-run-one-ui",
                        bundle_sha256=Sha256("d" * 64),
                        files=(
                            ExportedFile(
                                "approved/xhs_grid/artifact-approved.png",
                                Sha256("e" * 64),
                                3,
                            ),
                            ExportedFile(
                                "rejected/artifact-rejected.png",
                                Sha256("f" * 64),
                                3,
                            ),
                            ExportedFile("manifest.json", Sha256("a" * 64), 2),
                        ),
                    ),
                    job_state=SimpleNamespace(
                        job=SimpleNamespace(status=SimpleNamespace(value="COMPLETED"))
                    ),
                ),
            )

        def close(self) -> None:
            calls["closed"] = True

    def opener(fds, reservation):
        calls["job_id"] = reservation.job_id.value
        descriptors = (
            fds.config_root_fd,
            fds.evidence_root_fd,
            fds.model_root_fd,
            fds.state_root_fd,
            fds.artifact_root_fd,
            fds.style_asset_root_fd,
            fds.export_root_fd,
        )
        identities = {(os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in descriptors}
        assert len(identities) == 7
        for fd in (fds.source_fd, fds.style_fd, fds.spec_fd, fds.metadata_fd):
            info = os.fstat(fd)
            assert stat.S_ISREG(info.st_mode)
            assert stat.S_IMODE(info.st_mode) == 0o600
        metadata = json.loads(os.pread(fds.metadata_fd, 65536, 0))
        calls["metadata"] = metadata
        return _Execution()

    service = bind_production_run_one_services(
        UiServices(lambda _text: pytest.fail("compile not used")),
        roots,
        reserve=lambda: _Reservation(),
        open_run_one=opener,
    )

    view = service.run_production_job(
        source_path,
        style_path,
        spec_path,
        "soft product lighting",
        "watermark",
        "https://example.test/source",
        "CC0-1.0",
        "Example",
        "obtained",
    )

    assert view.job_id == "run-one-ui"
    assert view.status == "COMPLETED"
    assert view.profile_label == "production"
    assert view.bundle_name == "bundle-run-one-ui"
    assert view.approved_images == (str(approved / "artifact-approved.png"),)
    assert view.rejected_images == (str(rejected / "artifact-rejected.png"),)
    assert "UNVERIFIABLE" in view.qa_table
    assert "\tpass" not in view.qa_table
    assert calls["job_id"] == "run-one-ui"
    assert calls["ran"] is True
    assert calls["closed"] is True
    assert calls["metadata"] == {
        "schema_version": "specstyle.production.job_input.v1",
        "source": {
            "asset_id": f"source-{hashlib.sha256(source_bytes).hexdigest()[:16]}",
            "credit": {
                "source_url": "https://example.test/source",
                "license": "CC0-1.0",
                "attribution": "Example",
                "consent": "obtained",
            },
        },
        "style": {"asset_id": f"style-{hashlib.sha256(style_bytes).hexdigest()[:16]}"},
        "prompt": {
            "template_pin": {
                "id": "ui-prompt-template",
                "revision": "v1",
                "sha256": (
                    "52e3054077274103b29878dc23626312ed3c4b27d4596579635d1af7bb90f84e"
                ),
            },
            "preset_id": "preset",
            "positive": "soft product lighting",
            "negative": "watermark",
        },
    }
    assert list(roots.staging_root.iterdir()) == []


def test_production_ui_service_rejects_missing_upload_without_opening(
    tmp_path: Path,
) -> None:
    from specstyle.ui.production_run import bind_production_run_one_services

    called = False

    def opener(_fds, _reservation):
        nonlocal called
        called = True
        raise AssertionError("should not open run_one")

    service = bind_production_run_one_services(
        UiServices(lambda _text: pytest.fail("compile not used")),
        _roots(tmp_path),
        reserve=lambda: SimpleNamespace(job_id=JobId("run-one-ui")),
        open_run_one=opener,
    )

    view = service.run_production_job(
        None,
        "style.png",
        "style.json",
        "positive",
        "",
        None,
        None,
        None,
        "not_applicable",
    )

    assert called is False
    assert view.status == "JOB_FAILED"
    assert view.message == "source upload required"


def test_production_ui_service_cleans_staging_after_open_failure(
    tmp_path: Path,
) -> None:
    from specstyle.errors import InfrastructureError
    from specstyle.ui.production_run import bind_production_run_one_services

    roots = _roots(tmp_path)
    source_path = _file(tmp_path / "source.png", b"source-image")
    style_path = _file(tmp_path / "style.png", b"style-image")
    spec_path = _file(
        tmp_path / "style.json",
        json.dumps(raw_spec().model_dump(mode="json"), separators=(",", ":")).encode(),
    )

    def opener(_fds, _reservation):
        raise InfrastructureError("open failed")

    service = bind_production_run_one_services(
        UiServices(lambda _text: pytest.fail("compile not used")),
        roots,
        reserve=lambda: SimpleNamespace(job_id=JobId("run-one-ui")),
        open_run_one=opener,
    )

    view = service.run_production_job(
        source_path,
        style_path,
        spec_path,
        "positive",
        "",
        None,
        None,
        None,
        "not_applicable",
    )

    assert view.status == "JOB_FAILED"
    assert view.message == "open failed"
    assert list(roots.staging_root.iterdir()) == []


def test_production_runtime_paths_reject_staging_alias(tmp_path: Path) -> None:
    from specstyle.errors import DomainError
    from specstyle.ui.production_run import ProductionUiRuntimePaths

    roots = _roots(tmp_path)
    values = {name: getattr(roots, name) for name in roots.__slots__}
    values["staging_root"] = roots.config_root

    with pytest.raises(DomainError, match="production runtime roots must be distinct"):
        ProductionUiRuntimePaths(**values)


def test_production_ui_service_allows_only_one_pipeline_lifecycle(
    tmp_path: Path,
) -> None:
    from specstyle.ui.production_run import bind_production_run_one_services

    roots = _roots(tmp_path)
    uploads = _uploads(tmp_path)
    first_running = threading.Event()
    release_first = threading.Event()
    reservations: list[str] = []

    class _Execution:
        def __init__(self, sequence: int) -> None:
            self.sequence = sequence

        def run(self):
            if self.sequence == 1:
                first_running.set()
                assert release_first.wait(timeout=2)
            return SimpleNamespace(
                job_result=SimpleNamespace(report=None),
                export_result=SimpleNamespace(
                    bundle=SimpleNamespace(
                        bundle_name=f"bundle-run-one-{self.sequence}",
                        bundle_sha256=Sha256("d" * 64),
                        files=(),
                    ),
                    job_state=SimpleNamespace(
                        job=SimpleNamespace(status=SimpleNamespace(value="COMPLETED"))
                    ),
                ),
            )

        def close(self) -> None:
            pass

    def reserve():
        value = f"run-one-{len(reservations) + 1}"
        reservations.append(value)
        return SimpleNamespace(job_id=JobId(value))

    def opener(_fds, _reservation):
        return _Execution(len(reservations))

    service = bind_production_run_one_services(
        UiServices(lambda _text: pytest.fail("compile not used")),
        roots,
        reserve=reserve,
        open_run_one=opener,
    )
    results = []

    def run_first() -> None:
        results.append(
            service.run_production_job(
                *uploads,
                "positive",
                "",
                None,
                None,
                None,
                "not_applicable",
            )
        )

    thread = threading.Thread(target=run_first)
    thread.start()
    assert first_running.wait(timeout=2)

    busy = service.run_production_job(
        *uploads,
        "positive",
        "",
        None,
        None,
        None,
        "not_applicable",
    )
    release_first.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert busy.status == "BUSY"
    assert busy.message == "production run busy"
    assert reservations == ["run-one-1"]
    assert len(results) == 1
    assert results[0].status == "COMPLETED"
