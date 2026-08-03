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
from specstyle.domain.identifiers import JobId
from specstyle.ui.app import UiServices
from specstyle.ui.production_ui_projection import (
    batch_busy as _batch_busy,
    batch_failure as _batch_failure,
    batch_projection as _batch_projection,
    batch_view as _batch_view,
    busy as _busy,
    cancelled as _cancelled,
    cancelled_batch_item as _cancelled_batch_item,
    failed_batch_item as _failed_batch_item,
    failure as _failure,
    failure_projection as _failure_projection,
    projection_with_message as _projection_with_message,
    successful_batch_item as _successful_batch_item,
    status_projection as _status_projection,
    terminal_projection as _terminal_projection,
)
from specstyle.ui.production_ui_state import (
    ProductionTerminalProjection,
    ProductionUiState,
)
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
from specstyle.workflow.job_store import JobStore
from specstyle.workflow.production_job_input import (
    validate_production_job_spec_text,
)


def _read_persisted_job_status(state_root: Path, job_id: str) -> str | None:
    store = JobStore(state_root)
    try:
        try:
            state = store.load(JobId(job_id))
        except DomainError as exc:
            if str(exc) == "job not found":
                return None
            raise
        return state.job.status.value
    finally:
        store.close()


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


def _safe_message(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")[:160]


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
    ui_state = ProductionUiState(
        lambda job_id: _read_persisted_job_status(paths.state_root, job_id)
    )

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
        token = ui_state.try_begin("single", 1)
        if token is None:
            return _busy()
        try:
            projection = _run(
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
                ui_state,
                token,
            )
            ui_state.finish(token, projection)
            return projection.run_view
        except BaseException:
            ui_state.abandon(token)
            raise

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
        token = ui_state.try_begin("batch", count if type(count) is int else 1)
        if token is None:
            return _batch_busy()
        try:
            view = _run_batch(
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
                ui_state,
                token,
            )
            ui_state.finish(
                token,
                _batch_projection(view, ui_state.item_projections(token)),
            )
            return view
        except BaseException:
            ui_state.abandon(token)
            raise

    return UiServices(
        base.compile_spec,
        get_job_status=ui_state.get_job_status,
        cancel_job=ui_state.cancel_job,
        get_qa_table=ui_state.get_qa_table,
        get_repair_timeline=ui_state.get_repair_timeline,
        get_export_summary=ui_state.get_export_summary,
        run_replay=ui_state.run_replay,
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
    ui_state: ProductionUiState,
    token: int,
) -> ProductionTerminalProjection:
    job_id = ""
    staged: _StagedInputs | None = None
    try:
        ui_state.set_phase(token, "STAGING")
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
        ui_state.set_phase(token, "RESERVED", job_id=job_id)
        if ui_state.is_cancel_requested(token):
            return _failure_projection(_cancelled(job_id))
        ui_state.set_phase(token, "OPENING", job_id=job_id)
        return _execute(paths, staged, reservation, open_run_one, ui_state, token)
    except _UiRunInputError as exc:
        return _failure_projection(_failure(job_id, _safe_message(exc)))
    except (DomainError, InfrastructureError) as exc:
        return _failure_projection(_failure(job_id, _safe_message(exc)))
    except Exception:
        return _failure_projection(_failure(job_id, "internal error"))
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
    ui_state: ProductionUiState,
    token: int,
) -> ProductionBatchUiView:
    if type(count) is not int or not 2 <= count <= 4:
        return _batch_failure("batch count must be an exact int from 2 to 4")
    staged: _StagedInputs | None = None
    try:
        ui_state.set_phase(token, "STAGING")
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
        stride = staged.max_rounds + 1
        items: list[ProductionBatchItemUiView] = []
        for index in range(count):
            if ui_state.is_cancel_requested(token):
                break
            ui_state.set_phase(token, "STAGING", job_id="", current_index=index)
            outcome = _reserve_batch_item(
                paths,
                staged,
                reserve,
                open_run_one,
                index,
                index * stride,
                ui_state,
                token,
            )
            items.append(outcome.item)
            ui_state.add_item_projection(token, outcome.projection)
            if ui_state.is_cancel_requested(token):
                break
        return _batch_view(tuple(items))
    except _UiRunInputError as exc:
        return _batch_failure(_safe_message(exc))
    except (DomainError, InfrastructureError) as exc:
        return _batch_failure(_safe_message(exc))
    except Exception:
        return _batch_failure("internal error")
    finally:
        if staged is not None:
            _cleanup_staging(staged.directory)


@dataclass(frozen=True, slots=True)
class _BatchItemOutcome:
    item: ProductionBatchItemUiView
    projection: ProductionTerminalProjection


def _item_outcome(item: ProductionBatchItemUiView) -> _BatchItemOutcome:
    return _BatchItemOutcome(item, _failure_projection(item.run))


def _reserve_batch_item(
    paths: ProductionUiRuntimePaths,
    staged: _StagedInputs,
    reserve: Callable[..., object],
    open_run_one: Callable[[ProductionRunOneFds, object], object],
    item_index: int,
    requested_variation: int,
    ui_state: ProductionUiState,
    token: int,
) -> _BatchItemOutcome:
    try:
        reservation = reserve(requested_variation)
        job_id = getattr(getattr(reservation, "job_id", None), "value", "")
        ui_state.set_phase(token, "RESERVED", job_id=job_id, current_index=item_index)
        if ui_state.is_cancel_requested(token):
            return _item_outcome(
                _cancelled_batch_item(item_index, requested_variation, job_id)
            )
    except (DomainError, InfrastructureError) as exc:
        return _item_outcome(
            _failed_batch_item(item_index, requested_variation, "", _safe_message(exc))
        )
    except Exception:
        return _item_outcome(
            _failed_batch_item(item_index, requested_variation, "", "internal error")
        )
    return _execute_batch_item(
        paths,
        staged,
        reservation,
        open_run_one,
        item_index,
        requested_variation,
        job_id,
        ui_state,
        token,
    )


def _execute_batch_item(
    paths: ProductionUiRuntimePaths,
    staged: _StagedInputs,
    reservation: object,
    open_run_one: Callable[[ProductionRunOneFds, object], object],
    item_index: int,
    requested_variation: int,
    job_id: str,
    ui_state: ProductionUiState,
    token: int,
) -> _BatchItemOutcome:
    try:
        ui_state.set_phase(token, "OPENING", job_id=job_id, current_index=item_index)
        with _OpenFds(paths, staged) as fds:
            execution = open_run_one(fds, reservation)
    except (DomainError, InfrastructureError) as exc:
        return _item_outcome(
            _failed_batch_item(
                item_index, requested_variation, job_id, _safe_message(exc)
            )
        )
    except Exception:
        return _item_outcome(
            _failed_batch_item(
                item_index, requested_variation, job_id, "internal error"
            )
        )
    pending_cancel = ui_state.register_execution(token, execution)
    if pending_cancel:
        ui_state.cancel_registered_execution(token)
    try:
        result = execution.run()
    except BaseException as primary:
        _ignored, cleanup_error = _close_single_execution(ui_state, token, execution)
        if not isinstance(primary, Exception):
            if cleanup_error is not None:
                primary.add_note("production batch item cleanup failed")
            raise
        if isinstance(primary, DomainError) and ui_state.is_cancel_requested(token):
            return _item_outcome(
                _cancelled_batch_item(
                    item_index, requested_variation, job_id, cleanup_error
                )
            )
        message = (
            _safe_message(primary)
            if isinstance(primary, (DomainError, InfrastructureError))
            else "internal error"
        )
        return _item_outcome(
            _failed_batch_item(
                item_index, requested_variation, job_id, message, cleanup_error
            )
        )
    ui_state.set_phase(token, "RESULT_READY", job_id=job_id, current_index=item_index)
    cleanup_result, cleanup_error = _close_single_execution(ui_state, token, execution)
    if cleanup_error is not None and cleanup_result is None:
        return _item_outcome(
            _failed_batch_item(
                item_index,
                requested_variation,
                job_id,
                cleanup_error,
                cleanup_error,
            )
        )
    result = cleanup_result if cleanup_result is not None else result
    try:
        projection = _terminal_projection(paths.export_root, result, job_id)
        item = _successful_batch_item(
            paths.export_root,
            result,
            item_index,
            requested_variation,
            staged.max_rounds,
            job_id,
            cleanup_error,
        )
        if cleanup_error is not None:
            projection = _projection_with_message(
                projection, "production run completed; cleanup failed"
            )
        return _BatchItemOutcome(item, projection)
    except (DomainError, InfrastructureError) as exc:
        return _item_outcome(
            _failed_batch_item(
                item_index,
                requested_variation,
                job_id,
                _safe_message(exc),
                cleanup_error,
            )
        )
    except Exception:
        return _item_outcome(
            _failed_batch_item(
                item_index,
                requested_variation,
                job_id,
                "internal error",
                cleanup_error,
            )
        )


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
    ui_state: ProductionUiState,
    token: int,
) -> ProductionTerminalProjection:
    job_id = getattr(getattr(reservation, "job_id", None), "value", "")
    with _OpenFds(paths, staged) as fds:
        execution = open_run_one(fds, reservation)
    pending_cancel = ui_state.register_execution(token, execution)
    if pending_cancel:
        ui_state.cancel_registered_execution(token)
    try:
        result = execution.run()
    except BaseException as primary:
        _cleanup_result, cleanup_error = _close_single_execution(
            ui_state, token, execution
        )
        if not isinstance(primary, Exception):
            if cleanup_error is not None:
                primary.add_note("production run cleanup failed")
            raise
        if isinstance(primary, DomainError) and ui_state.is_cancel_requested(token):
            projection = _failure_projection(_cancelled(job_id))
            if cleanup_error is not None:
                projection = _projection_with_message(
                    projection, "production run cancelled; cleanup failed"
                )
            return projection
        raise
    ui_state.set_phase(token, "RESULT_READY", job_id=job_id)
    cleanup_result, cleanup_error = _close_single_execution(ui_state, token, execution)
    if cleanup_error is not None and cleanup_result is None:
        try:
            persisted = _read_persisted_job_status(paths.state_root, job_id)
        except Exception:
            persisted = None
        if persisted is not None:
            return _status_projection(job_id, persisted, cleanup_error)
        raise InfrastructureError(cleanup_error)
    result = cleanup_result if cleanup_result is not None else result
    projection = _terminal_projection(paths.export_root, result, job_id)
    if cleanup_error is not None:
        projection = _projection_with_message(
            projection, "production run completed; cleanup failed"
        )
    return projection


def _close_single_execution(
    ui_state: ProductionUiState, token: int, execution: object
) -> tuple[object | None, str | None]:
    try:
        ui_state.close_execution(token, execution)
    except ProductionRunOneCleanupError as exc:
        return exc.result, str(exc)
    except Exception:
        return None, "internal cleanup error"
    return None, None


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
