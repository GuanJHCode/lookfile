#!/usr/bin/env python3
"""Explicitly enable audited LCM compiler capability in an AMD runtime context."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Iterator, Sequence

_OLD_PIPELINES = ["sdxl_turbo", "sdxl_base"]
_LCM_PIPELINES = ["sdxl_turbo", "lcm", "sdxl_base"]
_ROLES = ("base", "ip_adapter", "controlnet")
_MAX_CONTEXT_BYTES = 1024 * 1024
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
if hasattr(os, "O_CLOEXEC"):
    _DIRECTORY_FLAGS |= os.O_CLOEXEC
    _FILE_READ_FLAGS |= os.O_CLOEXEC


class ContextMigrationError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _result(status: str, reason: str, **fields: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "reason_code": reason,
        **fields,
    }


def _validate_directory(fd: int, label: str, *, private: bool = False) -> None:
    try:
        info = os.fstat(fd)
    except OSError as exc:
        raise ContextMigrationError(f"{label.upper()}_UNAVAILABLE") from exc
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid not in {0, os.geteuid()}
        or mode & 0o022
        or (private and (info.st_uid != os.geteuid() or mode != 0o700))
    ):
        raise ContextMigrationError(f"{label.upper()}_UNTRUSTED")


@contextmanager
def _open_root(path: Path, label: str, *, private: bool = False) -> Iterator[int]:
    try:
        fd = os.open(path, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise ContextMigrationError(f"{label.upper()}_UNAVAILABLE") from exc
    try:
        _validate_directory(fd, label, private=private)
        yield fd
    finally:
        os.close(fd)


def _validate_context_file(info: os.stat_result) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid not in {0, os.geteuid()}
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
        or not 0 < info.st_size <= _MAX_CONTEXT_BYTES
    ):
        raise ContextMigrationError("CONTEXT_FILE_UNTRUSTED")


@contextmanager
def _open_context_file(config_fd: int) -> Iterator[int]:
    try:
        fd = os.open("context.json", _FILE_READ_FLAGS, dir_fd=config_fd)
    except OSError as exc:
        raise ContextMigrationError("CONTEXT_FILE_UNAVAILABLE") from exc
    try:
        _validate_context_file(os.fstat(fd))
        yield fd
    finally:
        os.close(fd)


def _read_fd(fd: int) -> bytes:
    try:
        info = os.fstat(fd)
        _validate_context_file(info)
        os.lseek(fd, 0, os.SEEK_SET)
        value = os.read(fd, info.st_size + 1)
    except OSError as exc:
        raise ContextMigrationError("CONTEXT_READ_FAILED") from exc
    if len(value) != info.st_size or len(value) > _MAX_CONTEXT_BYTES:
        raise ContextMigrationError("CONTEXT_READ_FAILED")
    return value


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _load_context(config_fd: int, evidence_fd: int):
    from specstyle.production.context_config import load_production_context_config

    try:
        return load_production_context_config(config_fd, evidence_fd)
    except Exception as exc:
        raise ContextMigrationError("CONTEXT_VALIDATION_FAILED") from exc


def _parse_document(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextMigrationError("CONTEXT_JSON_INVALID") from exc
    if type(value) is not dict:
        raise ContextMigrationError("CONTEXT_JSON_INVALID")
    return value


def _model_support_state(document: dict[str, Any]) -> str:
    support = document.get("model_support")
    if type(support) is not list or len(support) != len(_ROLES):
        raise ContextMigrationError("MODEL_SUPPORT_REFUSED")
    roles: list[object] = []
    pipelines: list[object] = []
    for item in support:
        if type(item) is not dict or set(item) != {"role", "supported_pipelines"}:
            raise ContextMigrationError("MODEL_SUPPORT_REFUSED")
        roles.append(item["role"])
        pipelines.append(item["supported_pipelines"])
    if tuple(roles) != _ROLES:
        raise ContextMigrationError("MODEL_SUPPORT_REFUSED")
    if all(value == _OLD_PIPELINES for value in pipelines):
        return "OLD"
    if all(value == _LCM_PIPELINES for value in pipelines):
        return "LCM"
    raise ContextMigrationError("MODEL_SUPPORT_REFUSED")


def _target_document(document: dict[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(document)
    for item in target["model_support"]:
        item["supported_pipelines"] = list(_LCM_PIPELINES)
    comparison = copy.deepcopy(target)
    for item in comparison["model_support"]:
        item["supported_pipelines"] = list(_OLD_PIPELINES)
    if comparison != document:
        raise ContextMigrationError("CONTEXT_DIFF_REFUSED")
    return target


@contextmanager
def _migration_lock(audit_fd: int) -> Iterator[int]:
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open("preview-lcm-context.lock", flags, 0o600, dir_fd=audit_fd)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
        ):
            raise ContextMigrationError("MIGRATION_LOCK_UNTRUSTED")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        if "fd" in locals():
            os.close(fd)
        raise ContextMigrationError("MIGRATION_BUSY") from exc
    except OSError as exc:
        if "fd" in locals():
            os.close(fd)
        raise ContextMigrationError("MIGRATION_LOCK_UNAVAILABLE") from exc
    try:
        yield fd
    finally:
        os.close(fd)


def _write_all(fd: int, value: bytes) -> None:
    offset = 0
    try:
        while offset < len(value):
            written = os.write(fd, value[offset:])
            if written <= 0:
                raise OSError
            offset += written
        os.fsync(fd)
    except OSError as exc:
        raise ContextMigrationError("MIGRATION_WRITE_FAILED") from exc


def _write_once(root_fd: int, name: str, value: bytes) -> None:
    _cleanup_audit_temps(root_fd, name)
    if _existing_audit_matches(root_fd, name, value):
        return
    temp_name = f".{name}.tmp-{secrets.token_hex(12)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(temp_name, flags, 0o600, dir_fd=root_fd)
    except OSError as exc:
        raise ContextMigrationError("AUDIT_WRITE_FAILED") from exc
    try:
        _write_all(fd, value)
    except Exception:
        os.close(fd)
        _remove_audit_file(root_fd, temp_name)
        raise
    except BaseException:
        os.close(fd)
        raise
    else:
        os.close(fd)
    try:
        os.link(
            temp_name,
            name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        _remove_audit_file(root_fd, temp_name)
        if _existing_audit_matches(root_fd, name, value):
            return
        raise ContextMigrationError("AUDIT_CONFLICT")
    except OSError as exc:
        _remove_audit_file(root_fd, temp_name)
        raise ContextMigrationError("AUDIT_WRITE_FAILED") from exc
    try:
        os.unlink(temp_name, dir_fd=root_fd)
        os.fsync(root_fd)
    except OSError as exc:
        _remove_audit_files(root_fd, (temp_name, name))
        raise ContextMigrationError("AUDIT_SYNC_FAILED") from exc


def _existing_audit_matches(root_fd: int, name: str, value: bytes) -> bool:
    try:
        fd = os.open(name, _FILE_READ_FLAGS, dir_fd=root_fd)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ContextMigrationError("AUDIT_CONFLICT") from exc
    try:
        if _read_fd(fd) != value:
            raise ContextMigrationError("AUDIT_CONFLICT")
        return True
    finally:
        os.close(fd)


def _cleanup_audit_temps(root_fd: int, name: str) -> None:
    prefix = f".{name}.tmp-"
    try:
        candidates = [
            entry for entry in os.listdir(root_fd) if entry.startswith(prefix)
        ]
    except OSError as exc:
        raise ContextMigrationError("AUDIT_RECOVERY_FAILED") from exc
    for candidate in candidates:
        try:
            fd = os.open(candidate, _FILE_READ_FLAGS, dir_fd=root_fd)
            info = os.fstat(fd)
            os.close(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink not in {1, 2}
            ):
                raise ContextMigrationError("AUDIT_RECOVERY_FAILED")
            os.unlink(candidate, dir_fd=root_fd)
        except ContextMigrationError:
            raise
        except OSError as exc:
            raise ContextMigrationError("AUDIT_RECOVERY_FAILED") from exc
    if candidates:
        try:
            os.fsync(root_fd)
        except OSError as exc:
            raise ContextMigrationError("AUDIT_RECOVERY_FAILED") from exc


def _remove_audit_file(root_fd: int, name: str) -> None:
    _remove_audit_files(root_fd, (name,))


def _remove_audit_files(root_fd: int, names: Sequence[str]) -> None:
    try:
        for name in names:
            try:
                os.unlink(name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
        os.fsync(root_fd)
    except OSError as exc:
        raise ContextMigrationError("AUDIT_CLEANUP_FAILED") from exc


def _audit_payload(status: str, before: str, after: str) -> bytes:
    return _canonical_json(
        {
            "schema_version": "specstyle.preview-lcm-context-migration.v1",
            "status": status,
            "before_sha256": before,
            "after_sha256": after,
        }
    )


def _prepare_audit(
    audit_fd: int, raw: bytes, before: str, after: str
) -> tuple[str, str]:
    stem = f"migration-{before}-{after}"
    _write_once(audit_fd, f"context-before-{before}.json", raw)
    _write_once(
        audit_fd, f"{stem}.prepared.json", _audit_payload("PREPARED", before, after)
    )
    return stem, f"{stem}.committed.json"


def _read_audit_file(audit_fd: int, name: str) -> bytes:
    try:
        fd = os.open(name, _FILE_READ_FLAGS, dir_fd=audit_fd)
    except OSError as exc:
        raise ContextMigrationError("AUDIT_RECOVERY_FAILED") from exc
    try:
        return _read_fd(fd)
    except Exception as exc:
        raise ContextMigrationError("AUDIT_RECOVERY_FAILED") from exc
    finally:
        os.close(fd)


def _complete_commit_audit(audit_fd: int, before: str, after: str) -> None:
    stem = f"migration-{before}-{after}"
    backup = _read_audit_file(audit_fd, f"context-before-{before}.json")
    prepared = _read_audit_file(audit_fd, f"{stem}.prepared.json")
    try:
        document = _parse_document(backup)
        valid_backup = (
            _sha256(backup) == before
            and _model_support_state(document) == "OLD"
            and _sha256(_canonical_json(_target_document(document))) == after
            and prepared == _audit_payload("PREPARED", before, after)
        )
    except ContextMigrationError as exc:
        raise ContextMigrationError("AUDIT_RECOVERY_FAILED") from exc
    if not valid_backup:
        raise ContextMigrationError("AUDIT_RECOVERY_FAILED")
    _write_once(
        audit_fd,
        f"{stem}.committed.json",
        _audit_payload("COMMITTED", before, after),
    )


def _create_stage(config_fd: int, raw: bytes) -> tuple[str, int]:
    name = f".preview-lcm-context-{secrets.token_hex(12)}"
    created = False
    stage_fd: int | None = None
    try:
        os.mkdir(name, 0o700, dir_fd=config_fd)
        created = True
        stage_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=config_fd)
        _validate_directory(stage_fd, "staged_context", private=True)
        file_fd = os.open(
            "context.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=stage_fd,
        )
        try:
            _write_all(file_fd, raw)
        finally:
            os.close(file_fd)
        os.fsync(stage_fd)
        return name, stage_fd
    except BaseException as exc:
        if stage_fd is not None:
            _cleanup_stage(config_fd, name, stage_fd)
        elif created:
            try:
                os.rmdir(name, dir_fd=config_fd)
                os.fsync(config_fd)
            except OSError as cleanup_exc:
                raise ContextMigrationError("STAGING_CLEANUP_FAILED") from cleanup_exc
        if isinstance(exc, ContextMigrationError):
            raise
        if isinstance(exc, OSError):
            raise ContextMigrationError("STAGING_FAILED") from exc
        raise


def _cleanup_stage(config_fd: int, name: str, stage_fd: int) -> None:
    failure = False
    try:
        try:
            os.unlink("context.json", dir_fd=stage_fd)
        except FileNotFoundError:
            pass
        os.close(stage_fd)
        os.rmdir(name, dir_fd=config_fd)
        os.fsync(config_fd)
    except OSError:
        failure = True
    if failure:
        raise ContextMigrationError("STAGING_CLEANUP_FAILED")


def _validate_staged_context(stage_fd: int, evidence_fd: int) -> None:
    from specstyle.production.context_config import require_model_pipeline_support

    context = _load_context(stage_fd, evidence_fd)
    try:
        require_model_pipeline_support(context, "lcm", _ROLES)
    except Exception as exc:
        raise ContextMigrationError("STAGED_LCM_CAPABILITY_MISSING") from exc


def _validate_online_context(config_fd: int, evidence_fd: int) -> None:
    _validate_staged_context(config_fd, evidence_fd)


def _online_unchanged(
    config_fd: int, expected_identity: tuple[int, ...], expected_raw: bytes
) -> None:
    with _open_context_file(config_fd) as current:
        if (
            _identity(os.fstat(current)) != expected_identity
            or _read_fd(current) != expected_raw
        ):
            raise ContextMigrationError("CONTEXT_CHANGED_DURING_MIGRATION")


def _publish_stage(config_fd: int, stage_fd: int) -> None:
    try:
        os.replace(
            "context.json",
            "context.json",
            src_dir_fd=stage_fd,
            dst_dir_fd=config_fd,
        )
    except OSError as exc:
        raise ContextMigrationError("CONTEXT_PUBLISH_FAILED") from exc


def _sync_config_directory(config_fd: int) -> None:
    try:
        os.fsync(config_fd)
    except OSError as exc:
        raise ContextMigrationError("CONTEXT_SYNC_FAILED") from exc


def _rollback_context(config_fd: int, evidence_fd: int, raw: bytes) -> None:
    name, stage_fd = _create_stage(config_fd, raw)
    try:
        _load_context(stage_fd, evidence_fd)
        _publish_stage(config_fd, stage_fd)
        _sync_config_directory(config_fd)
        _load_context(config_fd, evidence_fd)
    except Exception as exc:
        raise ContextMigrationError("ROLLBACK_FAILED") from exc
    finally:
        _cleanup_stage(config_fd, name, stage_fd)


def _apply_migration(
    config_fd: int,
    evidence_fd: int,
    audit_fd: int,
    raw: bytes,
    original_identity: tuple[int, ...],
    before: str,
    target_raw: bytes,
    after: str,
) -> dict[str, object]:
    stem, committed_name = _prepare_audit(audit_fd, raw, before, after)
    name, stage_fd = _create_stage(config_fd, target_raw)
    published = False
    try:
        _validate_staged_context(stage_fd, evidence_fd)
        _online_unchanged(config_fd, original_identity, raw)
        _publish_stage(config_fd, stage_fd)
        published = True
        _sync_config_directory(config_fd)
        try:
            _validate_online_context(config_fd, evidence_fd)
            _write_once(
                audit_fd,
                committed_name,
                _audit_payload("COMMITTED", before, after),
            )
        except Exception:
            _rollback_context(config_fd, evidence_fd, raw)
            return _result(
                "ROLLED_BACK", "POST_PUBLISH_VALIDATION_FAILED", before_sha256=before
            )
        return _result("APPLIED", "OK", before_sha256=before, after_sha256=after)
    except Exception:
        if published:
            _rollback_context(config_fd, evidence_fd, raw)
            return _result("ROLLED_BACK", "PUBLISH_FAILED", before_sha256=before)
        raise
    finally:
        _cleanup_stage(config_fd, name, stage_fd)


def _migrate_locked(
    config_fd: int,
    evidence_fd: int,
    audit_fd: int,
    expected_before: str,
    apply: bool,
) -> dict[str, object]:
    _load_context(config_fd, evidence_fd)
    with _open_context_file(config_fd) as original:
        raw = _read_fd(original)
        original_identity = _identity(os.fstat(original))
        before = _sha256(raw)
        document = _parse_document(raw)
        state = _model_support_state(document)
        if state == "LCM":
            _validate_online_context(config_fd, evidence_fd)
            if before != expected_before:
                if not apply:
                    raise ContextMigrationError("EXPECTED_DIGEST_MISMATCH")
                _complete_commit_audit(audit_fd, expected_before, before)
                return _result(
                    "ALREADY_ENABLED",
                    "RECOVERED_COMMIT",
                    before_sha256=expected_before,
                    after_sha256=before,
                )
            return _result(
                "ALREADY_ENABLED", "OK", before_sha256=before, after_sha256=before
            )
        target_raw = _canonical_json(_target_document(document))
        after = _sha256(target_raw)
        if before != expected_before:
            raise ContextMigrationError("EXPECTED_DIGEST_MISMATCH")
        if not apply:
            return _result(
                "REFUSED",
                "DRY_RUN",
                before_sha256=before,
                after_sha256=after,
            )
        return _apply_migration(
            config_fd,
            evidence_fd,
            audit_fd,
            raw,
            original_identity,
            before,
            target_raw,
            after,
        )


def _valid_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def migrate_preview_lcm_context(
    *,
    config_root: Path,
    context_evidence_root: Path,
    audit_root: Path,
    expected_before_sha256: str,
    apply: bool,
) -> dict[str, object]:
    if not _valid_digest(expected_before_sha256) or type(apply) is not bool:
        return _result("REFUSED", "INVALID_ARGUMENT")
    try:
        with _open_root(Path(config_root), "config") as config_fd:
            with _open_root(Path(context_evidence_root), "evidence") as evidence_fd:
                if not apply:
                    return _migrate_locked(
                        config_fd, evidence_fd, -1, expected_before_sha256, False
                    )
                with _open_root(Path(audit_root), "audit", private=True) as audit_fd:
                    with _migration_lock(audit_fd):
                        return _migrate_locked(
                            config_fd,
                            evidence_fd,
                            audit_fd,
                            expected_before_sha256,
                            True,
                        )
    except ContextMigrationError as exc:
        return _result("REFUSED", exc.reason_code)
    except Exception:
        return _result("REFUSED", "INTERNAL_FAILURE")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enable audited Preview LCM context")
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--context-evidence-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--expected-before-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    result = migrate_preview_lcm_context(
        config_root=arguments.config_root,
        context_evidence_root=arguments.context_evidence_root,
        audit_root=arguments.audit_root,
        expected_before_sha256=arguments.expected_before_sha256,
        apply=arguments.apply,
    )
    sys.stdout.write(_canonical_json(result).decode("utf-8") + "\n")
    return 0 if result["status"] in {"APPLIED", "ALREADY_ENABLED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
