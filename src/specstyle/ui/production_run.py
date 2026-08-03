"""Production run-one UI binding.

This module adapts Gradio upload values into the file-descriptor boundary owned by
``workflow.run_one``. It does not construct generation, verification, or repair
backends directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from specstyle.errors import DomainError, InfrastructureError
from specstyle.ui.app import UiServices
from specstyle.ui.presenters import format_qa_table, present_qa_report
from specstyle.ui.view_models import ProductionRunUiView
from specstyle.workflow.run_one import (
    ProductionRunOneFds,
    open_production_run_one,
    reserve_production_run_one,
)
from specstyle.spec.loader import load_style_spec_text

_METADATA_VERSION = "specstyle.production.job_input.v1"
_PROMPT_TEMPLATE_ID = "ui-prompt-template"
_PROMPT_TEMPLATE_REVISION = "v1"
_PROMPT_TEMPLATE_SHA256 = (
    "52e3054077274103b29878dc23626312ed3c4b27d4596579635d1af7bb90f84e"
)


@dataclass(frozen=True, slots=True)
class ProductionUiRuntimePaths:
    config_root: Path
    evidence_root: Path
    model_root: Path
    state_root: Path
    artifact_root: Path
    style_asset_root: Path
    export_root: Path
    staging_root: Path

    def __post_init__(self) -> None:
        values = tuple(Path(getattr(self, field)) for field in self.__slots__)
        for field, value in zip(self.__slots__, values, strict=True):
            if not value.is_dir():
                raise DomainError(f"{field} unavailable")
            object.__setattr__(self, field, value)
        identities = tuple(
            (value.stat().st_dev, value.stat().st_ino) for value in values[:7]
        )
        if len(set(identities)) != 7:
            raise DomainError("production runtime roots must be distinct")


class _UiRunInputError(DomainError):
    pass


def bind_production_run_one_services(
    base: UiServices,
    paths: ProductionUiRuntimePaths,
    *,
    reserve: Callable[[], object] = reserve_production_run_one,
    open_run_one: Callable[
        [ProductionRunOneFds, object], object
    ] = open_production_run_one,
) -> UiServices:
    if type(base) is not UiServices or type(paths) is not ProductionUiRuntimePaths:
        raise DomainError("invalid production ui services")

    def run(
        source: object,
        style: object,
        spec: object,
        positive: str,
        negative: str,
        source_url: str | None,
        license_: str | None,
        attribution: str | None,
        consent: str,
    ) -> ProductionRunUiView:
        return _run(paths, reserve, open_run_one, source, style, spec, positive,
                    negative, source_url, license_, attribution, consent)

    return UiServices(
        base.compile_spec,
        get_job_status=base.get_job_status,
        cancel_job=base.cancel_job,
        get_qa_table=base.get_qa_table,
        get_repair_timeline=base.get_repair_timeline,
        get_export_summary=base.get_export_summary,
        run_replay=base.run_replay,
        run_production_job=run,
    )


def _run(
    paths: ProductionUiRuntimePaths,
    reserve: Callable[[], object],
    open_run_one: Callable[[ProductionRunOneFds, object], object],
    source: object,
    style: object,
    spec: object,
    positive: str,
    negative: str,
    source_url: str | None,
    license_: str | None,
    attribution: str | None,
    consent: str,
) -> ProductionRunUiView:
    job_id = ""
    staged: _StagedInputs | None = None
    try:
        staged = _stage_inputs(paths, source, style, spec, positive, negative,
                               source_url, license_, attribution, consent)
        reservation = reserve()
        job_id = getattr(getattr(reservation, "job_id", None), "value", "")
        return _execute(paths, staged, reservation, open_run_one)
    except _UiRunInputError as exc:
        return _failure(job_id, str(exc))
    except (DomainError, InfrastructureError) as exc:
        return _failure(job_id, str(exc))
    except Exception:
        return _failure(job_id, "internal error")
    finally:
        if staged is not None:
            _cleanup_staging(staged.directory)


@dataclass(frozen=True, slots=True)
class _StagedInputs:
    directory: Path
    source: Path
    style: Path
    spec: Path
    metadata: Path


def _stage_inputs(
    paths: ProductionUiRuntimePaths,
    source: object,
    style: object,
    spec: object,
    positive: str,
    negative: str,
    source_url: str | None,
    license_: str | None,
    attribution: str | None,
    consent: str,
) -> _StagedInputs:
    source_path = _upload_path(source, "source")
    style_path = _upload_path(style, "style")
    spec_path = _upload_path(spec, "spec")
    staged = Path(tempfile.mkdtemp(prefix="ui-run-", dir=paths.staging_root))
    try:
        staged.chmod(0o700)
        source_dst = _copy_upload(source_path, staged / "source.bin")
        style_dst = _copy_upload(style_path, staged / "style.bin")
        spec_dst = _copy_upload(spec_path, staged / "spec.json")
        metadata = _metadata(source_dst, style_dst, spec_dst, positive, negative,
                             source_url, license_, attribution, consent)
        metadata_dst = _write_private(staged / "metadata.json", metadata)
        return _StagedInputs(staged, source_dst, style_dst, spec_dst, metadata_dst)
    except BaseException:
        _cleanup_staging(staged)
        raise


def _upload_path(value: object, label: str) -> Path:
    raw = value
    if raw is None:
        raise _UiRunInputError(f"{label} upload required")
    if not isinstance(raw, str):
        raw = getattr(raw, "name", None)
    if not isinstance(raw, str) or not raw:
        raise _UiRunInputError(f"{label} upload required")
    path = Path(str(raw))
    if not path.is_file():
        raise _UiRunInputError(f"{label} upload required")
    return path


def _cleanup_staging(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _copy_upload(source: Path, target: Path) -> Path:
    with source.open("rb") as input_file:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as output:
                fd = -1
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            raise
    target.chmod(0o600)
    return target


def _metadata(
    source: Path,
    style: Path,
    spec: Path,
    positive: str,
    negative: str,
    source_url: str | None,
    license_: str | None,
    attribution: str | None,
    consent: str,
) -> bytes:
    raw = load_style_spec_text(spec.read_text(encoding="utf-8"))
    data = {
        "schema_version": _METADATA_VERSION,
        "source": {
            "asset_id": _asset_id("source", source),
            "credit": {
                "source_url": source_url or None,
                "license": license_ or None,
                "attribution": attribution or None,
                "consent": consent,
            },
        },
        "style": {"asset_id": _asset_id("style", style)},
        "prompt": {
            "template_pin": {
                "id": _PROMPT_TEMPLATE_ID,
                "revision": _PROMPT_TEMPLATE_REVISION,
                "sha256": _PROMPT_TEMPLATE_SHA256,
            },
            "preset_id": raw.style.preset_id,
            "positive": positive,
            "negative": negative,
        },
    }
    return json.dumps(data, separators=(",", ":"), sort_keys=False).encode("utf-8")


def _asset_id(prefix: str, path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _write_private(path: Path, content: bytes) -> Path:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            fd = -1
            output.write(content)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    path.chmod(0o600)
    return path


def _execute(
    paths: ProductionUiRuntimePaths,
    staged: _StagedInputs,
    reservation: object,
    open_run_one: Callable[[ProductionRunOneFds, object], object],
) -> ProductionRunUiView:
    job_id = getattr(getattr(reservation, "job_id", None), "value", "")
    with _OpenFds(paths, staged) as fds:
        execution = open_run_one(fds, reservation)
    try:
        result = execution.run()
    finally:
        execution.close()
    return _view(paths.export_root, result, job_id)


class _OpenFds:
    def __init__(self, paths: ProductionUiRuntimePaths, staged: _StagedInputs) -> None:
        self._paths = paths
        self._staged = staged
        self._fds: list[int] = []

    def __enter__(self) -> ProductionRunOneFds:
        try:
            for path in _root_paths(self._paths):
                self._fds.append(os.open(path, os.O_RDONLY | os.O_DIRECTORY))
            for path in (
                self._staged.source,
                self._staged.style,
                self._staged.spec,
                self._staged.metadata,
            ):
                self._assert_private_file(path)
                self._fds.append(os.open(path, os.O_RDONLY))
            return ProductionRunOneFds(*self._fds)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_args: object) -> None:
        while self._fds:
            os.close(self._fds.pop())

    @staticmethod
    def _assert_private_file(path: Path) -> None:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise DomainError("invalid staged production input")


def _root_paths(paths: ProductionUiRuntimePaths) -> tuple[Path, ...]:
    return (
        paths.config_root,
        paths.evidence_root,
        paths.model_root,
        paths.state_root,
        paths.artifact_root,
        paths.style_asset_root,
        paths.export_root,
    )


def _view(export_root: Path, result: object, job_id: str) -> ProductionRunUiView:
    export = getattr(result, "export_result")
    bundle = getattr(export, "bundle")
    status = _status(export)
    qa_table = _qa_table(getattr(getattr(result, "job_result", None), "report", None))
    approved = _image_paths(export_root, bundle, "approved/")
    rejected = _image_paths(export_root, bundle, "rejected/")
    return ProductionRunUiView(
        _job_id(export) or job_id,
        status,
        "production run completed",
        "production",
        bundle.bundle_name,
        getattr(getattr(bundle, "bundle_sha256", None), "value", None),
        approved,
        rejected,
        qa_table,
    )


def _status(export: object) -> str:
    job = getattr(getattr(export, "job_state", None), "job", None)
    status = getattr(job, "status", None)
    return getattr(status, "value", str(status or "COMPLETED"))


def _job_id(export: object) -> str:
    job = getattr(getattr(export, "job_state", None), "job", None)
    job_id = getattr(job, "job_id", None)
    return getattr(job_id, "value", "")


def _qa_table(report: object | None) -> str:
    if report is None:
        return "no qa"
    return format_qa_table(present_qa_report(report))  # type: ignore[arg-type]


def _image_paths(export_root: Path, bundle: object, prefix: str) -> tuple[str, ...]:
    result: list[str] = []
    for file in getattr(bundle, "files", ()):
        relative = getattr(file, "relative_path", "")
        if relative.startswith(prefix) and relative.endswith(".png"):
            result.append(str(export_root / bundle.bundle_name / relative))
    return tuple(sorted(result))


def _failure(job_id: str, message: str) -> ProductionRunUiView:
    return ProductionRunUiView(
        job_id,
        "JOB_FAILED",
        message,
        "production",
        "",
        None,
        (),
        (),
        "no qa",
    )
