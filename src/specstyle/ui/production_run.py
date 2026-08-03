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
import threading

from specstyle.errors import DomainError, InfrastructureError
from specstyle.ui.app import UiServices
from specstyle.ui.presenters import format_qa_table, present_qa_report
from specstyle.ui.view_models import (
    ProductionBatchItemUiView,
    ProductionBatchUiView,
    ProductionRunUiView,
)
from specstyle.workflow.run_one import (
    ProductionRunOneCleanupError,
    ProductionRunOneFds,
    open_production_run_one,
    reserve_production_run_one,
)
from specstyle.workflow.production_job_input import (
    validate_production_job_spec_text,
)

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
            (value.stat().st_dev, value.stat().st_ino) for value in values
        )
        if len(set(identities)) != len(values):
            raise DomainError("production runtime roots must be distinct")


class _UiRunInputError(DomainError):
    pass


def bind_production_run_one_services(
    base: UiServices,
    paths: ProductionUiRuntimePaths,
    *,
    reserve: Callable[..., object] = reserve_production_run_one,
    open_run_one: Callable[
        [ProductionRunOneFds, object], object
    ] = open_production_run_one,
) -> UiServices:
    if type(base) is not UiServices or type(paths) is not ProductionUiRuntimePaths:
        raise DomainError("invalid production ui services")
    active_run = threading.Lock()

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
        if not active_run.acquire(blocking=False):
            return _busy()
        try:
            return _run(
                paths,
                reserve,
                open_run_one,
                source,
                style,
                spec,
                positive,
                negative,
                source_url,
                license_,
                attribution,
                consent,
            )
        finally:
            active_run.release()

    def run_batch(
        source: object,
        style: object,
        spec: object,
        positive: str,
        negative: str,
        source_url: str | None,
        license_: str | None,
        attribution: str | None,
        consent: str,
        count: int,
    ) -> ProductionBatchUiView:
        if not active_run.acquire(blocking=False):
            return _batch_busy()
        try:
            return _run_batch(
                paths,
                reserve,
                open_run_one,
                source,
                style,
                spec,
                positive,
                negative,
                source_url,
                license_,
                attribution,
                consent,
                count,
            )
        finally:
            active_run.release()

    return UiServices(
        base.compile_spec,
        get_job_status=base.get_job_status,
        cancel_job=base.cancel_job,
        get_qa_table=base.get_qa_table,
        get_repair_timeline=base.get_repair_timeline,
        get_export_summary=base.get_export_summary,
        run_replay=base.run_replay,
        run_production_job=run,
        run_production_batch=run_batch,
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
        staged = _stage_inputs(
            paths,
            source,
            style,
            spec,
            positive,
            negative,
            source_url,
            license_,
            attribution,
            consent,
        )
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


def _run_batch(
    paths: ProductionUiRuntimePaths,
    reserve: Callable[..., object],
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
    count: int,
) -> ProductionBatchUiView:
    if type(count) is not int or not 2 <= count <= 4:
        return _batch_failure("batch count must be an exact int from 2 to 4")
    staged: _StagedInputs | None = None
    try:
        staged = _stage_inputs(
            paths, source, style, spec, positive, negative,
            source_url, license_, attribution, consent,
        )
        stride = staged.max_rounds + 1
        items = tuple(
            _reserve_batch_item(
                paths, staged, reserve, open_run_one, index, index * stride
            )
            for index in range(count)
        )
        return _batch_view(items)
    except _UiRunInputError as exc:
        return _batch_failure(str(exc))
    except (DomainError, InfrastructureError) as exc:
        return _batch_failure(str(exc))
    except Exception:
        return _batch_failure("internal error")
    finally:
        if staged is not None:
            _cleanup_staging(staged.directory)


def _reserve_batch_item(
    paths: ProductionUiRuntimePaths,
    staged: _StagedInputs,
    reserve: Callable[..., object],
    open_run_one: Callable[[ProductionRunOneFds, object], object],
    item_index: int,
    requested_variation: int,
) -> ProductionBatchItemUiView:
    try:
        reservation = reserve(requested_variation)
        job_id = getattr(getattr(reservation, "job_id", None), "value", "")
    except (DomainError, InfrastructureError) as exc:
        return _failed_batch_item(item_index, requested_variation, "", str(exc))
    except Exception:
        return _failed_batch_item(item_index, requested_variation, "", "internal error")
    return _execute_batch_item(
        paths, staged, reservation, open_run_one,
        item_index, requested_variation, job_id,
    )


def _execute_batch_item(
    paths: ProductionUiRuntimePaths,
    staged: _StagedInputs,
    reservation: object,
    open_run_one: Callable[[ProductionRunOneFds, object], object],
    item_index: int,
    requested_variation: int,
    job_id: str,
) -> ProductionBatchItemUiView:
    try:
        with _OpenFds(paths, staged) as fds:
            execution = open_run_one(fds, reservation)
    except (DomainError, InfrastructureError) as exc:
        return _failed_batch_item(item_index, requested_variation, job_id, str(exc))
    except Exception:
        return _failed_batch_item(item_index, requested_variation, job_id, "internal error")
    try:
        result = execution.run()
    except BaseException as primary:
        _ignored, cleanup_error = _close_batch_execution(execution)
        if not isinstance(primary, Exception):
            if cleanup_error is not None:
                primary.add_note("production batch item cleanup failed")
            raise
        message = str(primary) if isinstance(primary, (DomainError, InfrastructureError)) else "internal error"
        return _failed_batch_item(
            item_index, requested_variation, job_id, message, cleanup_error
        )
    cleanup_result, cleanup_error = _close_batch_execution(execution)
    if cleanup_error is not None and cleanup_result is None:
        return _failed_batch_item(
            item_index, requested_variation, job_id, cleanup_error, cleanup_error
        )
    result = cleanup_result if cleanup_result is not None else result
    try:
        return _successful_batch_item(
            paths.export_root, result, item_index, requested_variation,
            staged.max_rounds, job_id, cleanup_error,
        )
    except (DomainError, InfrastructureError) as exc:
        return _failed_batch_item(
            item_index, requested_variation, job_id, str(exc), cleanup_error
        )
    except Exception:
        return _failed_batch_item(
            item_index, requested_variation, job_id, "internal error", cleanup_error
        )


def _close_batch_execution(execution: object) -> tuple[object | None, str | None]:
    try:
        execution.close()
    except ProductionRunOneCleanupError as exc:
        return exc.result, str(exc)
    except Exception:
        return None, "internal cleanup error"
    return None, None


@dataclass(frozen=True, slots=True)
class _StagedInputs:
    directory: Path
    source: Path
    style: Path
    spec: Path
    metadata: Path
    max_rounds: int


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
        metadata, max_rounds = _metadata(
            source_dst,
            style_dst,
            spec_dst,
            positive,
            negative,
            source_url,
            license_,
            attribution,
            consent,
        )
        metadata_dst = _write_private(staged / "metadata.json", metadata)
        return _StagedInputs(
            staged, source_dst, style_dst, spec_dst, metadata_dst, max_rounds
        )
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
) -> tuple[bytes, int]:
    summary = validate_production_job_spec_text(spec.read_text(encoding="utf-8"))
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
            "preset_id": summary.preset_id,
            "positive": positive,
            "negative": negative,
        },
    }
    encoded = json.dumps(data, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return encoded, summary.max_rounds


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


def _successful_batch_item(
    export_root: Path,
    result: object,
    item_index: int,
    requested_variation: int,
    max_rounds: int,
    job_id: str,
    cleanup_error: str | None,
) -> ProductionBatchItemUiView:
    job_result = getattr(result, "job_result", None)
    history = getattr(job_result, "history", None)
    initial_attempt = getattr(history, "initial_attempt", None)
    initial_request = getattr(initial_attempt, "request", None)
    final_request = getattr(job_result, "request", None)
    initial_variation, initial_seed = _seed_evidence(initial_request)
    final_variation, final_seed = _seed_evidence(final_request)
    if (
        initial_variation != requested_variation
        or not requested_variation <= final_variation <= requested_variation + max_rounds
    ):
        raise InfrastructureError("production batch evidence unavailable")
    return ProductionBatchItemUiView(
        item_index,
        requested_variation,
        initial_seed,
        final_variation,
        final_seed,
        _view(export_root, result, job_id),
        cleanup_error,
    )


def _seed_evidence(request: object) -> tuple[int, int]:
    variation = getattr(request, "variation_index", None)
    snapshot = getattr(request, "seed", None)
    snapshot_variation = getattr(snapshot, "variation_index", None)
    seed = getattr(snapshot, "seed", None)
    if (
        type(variation) is not int
        or not 0 <= variation < 2**31
        or type(snapshot_variation) is not int
        or snapshot_variation != variation
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise InfrastructureError("production batch evidence unavailable")
    return variation, seed


def _failed_batch_item(
    item_index: int,
    requested_variation: int,
    job_id: str,
    message: str,
    cleanup_error: str | None = None,
) -> ProductionBatchItemUiView:
    return ProductionBatchItemUiView(
        item_index,
        requested_variation,
        None,
        None,
        None,
        _failure(job_id, message),
        cleanup_error,
    )


def _batch_view(
    items: tuple[ProductionBatchItemUiView, ...],
) -> ProductionBatchUiView:
    completed = sum(item.run.status == "COMPLETED" for item in items)
    cleanup_failed = any(item.cleanup_error is not None for item in items)
    if completed == 0:
        status = "JOB_FAILED"
    elif completed == len(items) and not cleanup_failed:
        status = "COMPLETED"
    else:
        status = "PARTIAL"
    final_seeds = tuple(
        item.final_seed
        for item in items
        if item.run.status == "COMPLETED" and item.final_seed is not None
    )
    collision = len(final_seeds) != len(set(final_seeds))
    diversity = status == "COMPLETED" and not collision
    return ProductionBatchUiView(
        status,
        f"{completed}/{len(items)} exports completed",
        "production",
        items,
        collision,
        diversity,
        tuple(path for item in items for path in item.run.approved_images),
        tuple(path for item in items for path in item.run.rejected_images),
        _batch_tsv(items, status, collision, diversity),
    )


def _batch_tsv(
    items: tuple[ProductionBatchItemUiView, ...],
    status: str,
    collision: bool,
    diversity: bool,
) -> str:
    if collision:
        evidence = "NOT_DIVERSITY_EVIDENCE_FINAL_SEED_COLLISION"
    elif diversity:
        evidence = "VALID_DIVERSITY_EVIDENCE"
    else:
        evidence = "NOT_DIVERSITY_EVIDENCE_INCOMPLETE_BATCH"
    rows = [
        "item_index\trequested_variation\tinitial_seed\tfinal_variation\t"
        "final_seed\tjob_id\tjob_status\tmessage\tbundle\tbundle_sha256\t"
        "cleanup_error\tqa\tbatch_status\tevidence"
    ]
    for item in items:
        values = (
            item.item_index,
            item.requested_variation_index,
            item.initial_seed,
            item.final_variation_index,
            item.final_seed,
            item.run.job_id,
            item.run.status,
            item.run.message,
            item.run.bundle_name,
            item.run.bundle_sha256,
            item.cleanup_error,
            item.run.qa_table,
            status,
            evidence,
        )
        rows.append("\t".join(_tsv_value(value) for value in values))
    return "\n".join(rows)


def _tsv_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


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


def _busy() -> ProductionRunUiView:
    return ProductionRunUiView(
        "",
        "BUSY",
        "production run busy",
        "production",
        "",
        None,
        (),
        (),
        "no qa",
    )


def _batch_failure(message: str) -> ProductionBatchUiView:
    return ProductionBatchUiView(
        "JOB_FAILED",
        message,
        "production",
        (),
        False,
        False,
        (),
        (),
        _batch_tsv((), "JOB_FAILED", False, False),
    )


def _batch_busy() -> ProductionBatchUiView:
    return ProductionBatchUiView(
        "BUSY",
        "production run busy",
        "production",
        (),
        False,
        False,
        (),
        (),
        _batch_tsv((), "BUSY", False, False),
    )
