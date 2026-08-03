"""Atomic private Preview evidence and derived display-only PNG copies."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import fcntl
import json
import os
import re
import stat
import sys
import uuid

from PIL import Image, UnidentifiedImageError

from specstyle.domain.identifiers import ArtifactId, Sha256
from specstyle.errors import DomainError, InfrastructureError
from specstyle.exporting.qa_report import canonical_json_bytes
from specstyle.generation.preview_execution import (
    PreviewGeneratedArtifact,
    _validate_execution_binding,
)
from specstyle.observability.hashing import hash_bytes
from specstyle.workflow._job_store_fs import (
    CorruptStore,
    DestinationExists,
    StoreIO,
    close_owned,
    directory_names,
    fsync_directory,
    open_directory,
    read_file,
    rename_noreplace,
    validate_directory,
    validate_file,
    write_slot,
)
from specstyle.workflow.production_job_input import _strict_json

__all__ = (
    "PreviewEvidencePublication",
    "publish_preview_evidence",
    "reconcile_preview_display",
)

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", re.ASCII)
_DISPLAY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}-[0-9a-f]{16}\.png")
_PNG_BYTES = 32 * 1024 * 1024
_RECORD_BYTES = 64 * 1024


def _domain(message: str = "invalid preview evidence") -> DomainError:
    return DomainError(message)


def _infra(message: str = "preview evidence unavailable") -> InfrastructureError:
    return InfrastructureError(message)


@dataclass(frozen=True, slots=True)
class PreviewEvidencePublication:
    evidence_name: str
    display_name: str
    artifact_id: ArtifactId
    content_sha256: Sha256
    execution_fingerprint: Sha256

    def __post_init__(self) -> None:
        if (
            type(self.evidence_name) is not str
            or _NAME.fullmatch(self.evidence_name) is None
            or type(self.display_name) is not str
            or _DISPLAY.fullmatch(self.display_name) is None
            or type(self.artifact_id) is not ArtifactId
            or type(self.content_sha256) is not Sha256
            or type(self.execution_fingerprint) is not Sha256
        ):
            raise _domain()


def _duplicate_root(root_fd: object) -> tuple[int, int, tuple[int, int]]:
    if type(root_fd) is not int or root_fd < 0:
        raise _domain()
    duplicate = -1
    try:
        duplicate = fcntl.fcntl(root_fd, fcntl.F_DUPFD_CLOEXEC, 0)
        info = os.fstat(duplicate)
        validate_directory(info, info.st_dev)
        return duplicate, info.st_dev, (info.st_dev, info.st_ino)
    except (OSError, CorruptStore, StoreIO):
        if duplicate >= 0:
            close_owned(duplicate, sys.exception())
        raise _infra() from None


def _png_size(content: bytes) -> tuple[int, int]:
    image: Image.Image | None = None
    try:
        image = Image.open(BytesIO(content))
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or getattr(image, "n_frames", 1) != 1
            or image.info
        ):
            raise _domain()
        image.load()
        if image.info:
            raise _domain()
        return image.size
    except DomainError:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise _domain() from None
    finally:
        if image is not None:
            image.close()


def _artifact_material(
    artifact: object,
) -> tuple[PreviewGeneratedArtifact, dict[str, object]]:
    if type(artifact) is not PreviewGeneratedArtifact:
        raise _domain()
    _validate_execution_binding(artifact.binding)
    try:
        material = json.loads(artifact.binding.material_json)
        graph = material["compiled_request"]["graph"]
        resolution = tuple(graph["resolution"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise _domain() from None
    if (
        len(resolution) != 2
        or any(type(item) is not int for item in resolution)
        or _png_size(artifact.content) != resolution
        or hash_bytes(artifact.content) != artifact.content_sha256
    ):
        raise _domain()
    return artifact, material


def _record(
    run_id: str,
    display_name: str,
    artifact: PreviewGeneratedArtifact,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": "specstyle.preview.evidence.v1",
            "run_id": run_id,
            "profile": "preview",
            "artifact": {
                "artifact_id": artifact.artifact_id.value,
                "file": "artifact.png",
                "display_file": display_name,
                "content_sha256": artifact.content_sha256.value,
                "compiled_request_fingerprint": (
                    artifact.binding.compiled_request_fingerprint.value
                ),
                "execution_fingerprint": artifact.execution_fingerprint.value,
            },
            "planes": {
                "verification": "NOT_RUN",
                "repair": "NOT_RUN",
                "export": "NOT_RUN",
            },
        }
    )


def _remove_staging(root_fd: int, name: str) -> None:
    try:
        opened = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
    except OSError:
        return
    try:
        for child in ("record.json", "artifact.png"):
            try:
                os.unlink(child, dir_fd=opened)
            except FileNotFoundError:
                pass
    finally:
        os.close(opened)
    try:
        os.rmdir(name, dir_fd=root_fd)
    except OSError:
        pass


def _publish_private(
    root_fd: int,
    root_dev: int,
    run_id: str,
    artifact: PreviewGeneratedArtifact,
    record: bytes,
) -> None:
    staging = f".preview-{uuid.uuid4().hex}"
    directory_fd = -1
    try:
        os.mkdir(staging, 0o700, dir_fd=root_fd)
        opened = open_directory(root_fd, staging, root_dev)
        if opened is None:
            raise CorruptStore
        directory_fd = opened[0]
        write_slot(directory_fd, "artifact.png", artifact.content, _PNG_BYTES, root_dev)
        write_slot(directory_fd, "record.json", record, _RECORD_BYTES, root_dev)
        fsync_directory(directory_fd)
        close_owned(directory_fd)
        directory_fd = -1
        rename_noreplace(root_fd, staging, run_id)
        fsync_directory(root_fd)
    except DestinationExists:
        _remove_staging(root_fd, staging)
        raise _domain("preview evidence already exists") from None
    except (OSError, CorruptStore, StoreIO):
        _remove_staging(root_fd, staging)
        raise _infra() from None
    finally:
        if directory_fd >= 0:
            close_owned(directory_fd, sys.exception())


def _publish_display(
    root_fd: int, root_dev: int, display_name: str, artifact: PreviewGeneratedArtifact
) -> None:
    staging = f".display-{uuid.uuid4().hex}"
    try:
        write_slot(root_fd, staging, artifact.content, _PNG_BYTES, root_dev)
        rename_noreplace(root_fd, staging, display_name)
        fsync_directory(root_fd)
        record = read_file(
            root_fd,
            display_name,
            len(artifact.content),
            len(artifact.content),
            root_dev,
        )
        if record is None or hash_bytes(record.data) != artifact.content_sha256:
            raise CorruptStore
    except (DestinationExists, OSError, CorruptStore, StoreIO):
        try:
            os.unlink(staging, dir_fd=root_fd)
        except OSError:
            pass
        raise _infra("preview display unavailable") from None


def publish_preview_evidence(
    evidence_root_fd: int,
    display_root_fd: int,
    run_id: str,
    artifact: PreviewGeneratedArtifact,
    /,
) -> PreviewEvidencePublication:
    if type(run_id) is not str or _NAME.fullmatch(run_id) is None:
        raise _domain()
    artifact, _material = _artifact_material(artifact)
    display_name = f"{run_id}-{artifact.content_sha256.value[:16]}.png"
    private_fd = display_fd = -1
    try:
        private_fd, private_dev, private_identity = _duplicate_root(evidence_root_fd)
        display_fd, display_dev, display_identity = _duplicate_root(display_root_fd)
        if private_identity == display_identity:
            raise _domain()
        record = _record(run_id, display_name, artifact)
        _publish_private(private_fd, private_dev, run_id, artifact, record)
        _publish_display(display_fd, display_dev, display_name, artifact)
        return PreviewEvidencePublication(
            run_id,
            display_name,
            artifact.artifact_id,
            artifact.content_sha256,
            artifact.execution_fingerprint,
        )
    finally:
        close_owned(display_fd, sys.exception())
        close_owned(private_fd, sys.exception())


def _valid_record(root_fd: int, root_dev: int, name: str) -> tuple[str, Sha256] | None:
    if _NAME.fullmatch(name) is None:
        return None
    opened = open_directory(root_fd, name, root_dev, missing_ok=True)
    if opened is None:
        return None
    directory_fd = opened[0]
    try:
        if set(directory_names(directory_fd)) != {"artifact.png", "record.json"}:
            return None
        record_file = read_file(directory_fd, "record.json", 1, _RECORD_BYTES, root_dev)
        artifact_file = read_file(directory_fd, "artifact.png", 1, _PNG_BYTES, root_dev)
        if record_file is None or artifact_file is None:
            return None
        raw = _strict_json(record_file.data)
        if (
            raw.get("schema_version") != "specstyle.preview.evidence.v1"
            or raw.get("run_id") != name
            or raw.get("profile") != "preview"
            or raw.get("planes")
            != {"verification": "NOT_RUN", "repair": "NOT_RUN", "export": "NOT_RUN"}
        ):
            return None
        artifact = raw.get("artifact")
        if type(artifact) is not dict:
            return None
        display_name = artifact.get("display_file")
        digest = Sha256(artifact.get("content_sha256"))
        if (
            _DISPLAY.fullmatch(display_name) is None
            or artifact.get("file") != "artifact.png"
            or hash_bytes(artifact_file.data) != digest
        ):
            return None
        return display_name, digest
    except (DomainError, CorruptStore, StoreIO, OSError, TypeError, ValueError):
        return None
    finally:
        close_owned(directory_fd, sys.exception())


def _remove_display_file(root_fd: int, root_dev: int, name: str) -> bool:
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISLNK(info.st_mode):
            validate_file(info, root_dev, 0, _PNG_BYTES)
        os.unlink(name, dir_fd=root_fd)
        return True
    except FileNotFoundError:
        return False
    except (OSError, CorruptStore):
        raise _infra("preview display unavailable") from None


def reconcile_preview_display(
    evidence_root_fd: int, display_root_fd: int, /
) -> tuple[str, ...]:
    private_fd = display_fd = -1
    removed: list[str] = []
    try:
        private_fd, private_dev, private_identity = _duplicate_root(evidence_root_fd)
        display_fd, display_dev, display_identity = _duplicate_root(display_root_fd)
        if private_identity == display_identity:
            raise _domain()
        valid = {
            item[0]: item[1]
            for name in directory_names(private_fd)
            if (item := _valid_record(private_fd, private_dev, name)) is not None
        }
        for name in directory_names(display_fd):
            expected = valid.get(name)
            keep = False
            if expected is not None and _DISPLAY.fullmatch(name) is not None:
                try:
                    item = read_file(display_fd, name, 1, _PNG_BYTES, display_dev)
                    keep = item is not None and hash_bytes(item.data) == expected
                except (CorruptStore, StoreIO):
                    keep = False
            if not keep and _remove_display_file(display_fd, display_dev, name):
                removed.append(name)
        if removed:
            fsync_directory(display_fd)
        return tuple(sorted(removed))
    except (CorruptStore, StoreIO, OSError):
        raise _infra() from None
    finally:
        close_owned(display_fd, sys.exception())
        close_owned(private_fd, sys.exception())
