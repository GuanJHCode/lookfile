"""EXP-001B secure atomic bundle publish (§13.11).

实现 trusted root fd、同盘 staging、``O_NOFOLLOW/O_EXCL``、short-write loop、
fsync、写后 readback、Linux ``renameat2(RENAME_NOREPLACE)`` 与 macOS
``renameatx_np(RENAME_EXCL)`` 原子发布、安全 cleanup 与 ``ExportBundle``。
不使用 Path convenience、``resolve``、``shutil`` 或普通 rename fallback；
不重跑 verifier 或修改 gate。
"""

from __future__ import annotations

import ctypes
import errno
import os
import re
import stat
import sys
from dataclasses import dataclass
from io import BytesIO

from specstyle.domain.identifiers import Sha256
from specstyle.errors import DomainError, InfrastructureError
from specstyle.exporting.manifest import ExportRequest, _prepare_export
from specstyle.observability.hashing import hash_bytes

_BUNDLE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", re.ASCII)
_ALWAYS_ON_DIRS = (
    ("approved",),
    ("approved", "xhs_grid"),
    ("approved", "talking_head_cover"),
    ("approved", "background_sequence"),
    ("rejected",),
    ("manual_review",),
)
_RENAME_EXCL = 0x00000004
_RENAME_NOREPLACE = 1


@dataclass(frozen=True, slots=True)
class ExportedFile:
    relative_path: str
    sha256: Sha256
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ExportBundle:
    bundle_name: str
    root_device: int
    root_inode: int
    manifest_sha256: Sha256
    payload_sha256: Sha256
    bundle_sha256: Sha256
    files: tuple[ExportedFile, ...]


def _invalid_target() -> None:
    raise DomainError("invalid export target") from None


def _target_exists() -> None:
    raise DomainError("export target exists") from None


def _hash_mismatch() -> None:
    raise DomainError("export hash mismatch") from None


def _write_failed(cause: BaseException) -> None:
    raise InfrastructureError("export write failed") from cause


def _readback_failed(cause: BaseException) -> None:
    raise InfrastructureError("export readback failed") from cause


def _sync_failed(cause: BaseException) -> None:
    raise InfrastructureError("export sync failed") from cause


def _publication_unavailable(cause: BaseException | None) -> None:
    raise InfrastructureError("secure atomic publication unavailable") from cause


def _publish_failed(cause: BaseException) -> None:
    raise InfrastructureError("export publish failed") from cause


def _validate_bundle_name(bundle_name: object) -> str:
    if type(bundle_name) is not str or _BUNDLE_NAME_RE.fullmatch(bundle_name) is None:
        _invalid_target()
    return bundle_name


def _dup_root_fd(target_root_fd: object) -> int:
    if type(target_root_fd) is bool or type(target_root_fd) is not int:
        _invalid_target()
    try:
        dup = os.dup(target_root_fd)
    except OSError:
        _invalid_target()
    os.set_inheritable(dup, False)
    try:
        st = os.fstat(dup)
    except OSError:
        os.close(dup)
        _invalid_target()
    if not stat.S_ISDIR(st.st_mode):
        os.close(dup)
        _invalid_target()
    return dup


def _libc():
    try:
        if sys.platform == "darwin":
            return ctypes.CDLL("/usr/lib/libc.dylib", use_errno=True)
        return ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError as cause:
        _publication_unavailable(cause)


def _native_rename(from_fd: int, from_name: str, to_fd: int, to_name: str) -> None:
    libc = _libc()
    from_bytes = from_name.encode("utf-8")
    to_bytes = to_name.encode("utf-8")
    if sys.platform == "darwin":
        fn = getattr(libc, "renameatx_np", None)
        if fn is None:
            _publication_unavailable(None)
        fn.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        fn.restype = ctypes.c_int
        ret = fn(from_fd, from_bytes, to_fd, to_bytes, _RENAME_EXCL)
    else:
        fn = getattr(libc, "renameat2", None)
        if fn is None:
            _publication_unavailable(None)
        fn.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        fn.restype = ctypes.c_int
        ret = fn(from_fd, from_bytes, to_fd, to_bytes, _RENAME_NOREPLACE)
    if ret != 0:
        err = ctypes.get_errno()
        if err in (errno.EEXIST, errno.ENOTEMPTY):
            _target_exists()
        caps = {
            errno.ENOSYS,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if err in caps:
            _publication_unavailable(OSError(err, os.strerror(err)))
        raise InfrastructureError("export publish failed") from OSError(
            err, os.strerror(err)
        )


def _create_staging(root_dup: int, root_st: os.stat_result) -> tuple[int, str]:
    while True:
        name = os.urandom(16).hex()
        try:
            os.mkdir(name, 0o700, dir_fd=root_dup)
        except FileExistsError:
            continue
        except OSError as cause:
            _write_failed(cause)
        fd = _open_dir(root_dup, name)
        try:
            st = os.fstat(fd)
        except OSError as cause:
            os.close(fd)
            _readback_failed(cause)
        if st.st_dev != root_st.st_dev:
            os.close(fd)
            _publication_unavailable(None)
        return fd, name


def _open_dir(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as cause:
        _readback_failed(cause)


def _mkdir_rel(staging_fd: int, parts: tuple[str, ...]) -> None:
    parent = staging_fd
    opened: list[int] = []
    try:
        for component in parts:
            try:
                os.mkdir(component, 0o700, dir_fd=parent)
            except FileExistsError:
                pass
            except OSError as cause:
                _write_failed(cause)
            child = _open_dir(parent, component)
            opened.append(child)
            parent = child
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _dir_parents(rel_path: str) -> list[tuple[str, ...]]:
    parts = tuple(rel_path.split("/"))
    return [parts[: i + 1] for i in range(len(parts) - 1)]


def _write_file(staging_fd: int, rel_path: str, content: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(rel_path, flags, 0o600, dir_fd=staging_fd)
    except OSError as cause:
        _write_failed(cause)
    try:
        written = 0
        total = len(content)
        while written < total:
            try:
                chunk = os.write(fd, content[written:])
            except OSError as cause:
                _write_failed(cause)
            if type(chunk) is not int or chunk <= 0:
                _write_failed(InfrastructureError("short write returned zero"))
            written += chunk
        try:
            os.fsync(fd)
        except OSError as cause:
            _sync_failed(cause)
    finally:
        os.close(fd)


def _fsync_dir(staging_fd: int, parts: tuple[str, ...]) -> None:
    parent = staging_fd
    opened: list[int] = []
    try:
        for component in parts:
            child = _open_dir(parent, component)
            opened.append(child)
            parent = child
        for fd in reversed(opened):
            try:
                os.fsync(fd)
            except OSError as cause:
                _sync_failed(cause)
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _readback_file(staging_fd: int, rel_path: str, expected: bytes) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(rel_path, flags, dir_fd=staging_fd)
    except OSError as cause:
        _readback_failed(cause)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            _hash_mismatch()
        data = b""
        while len(data) < st.st_size:
            try:
                block = os.read(fd, 1 << 20)
            except OSError as cause:
                _readback_failed(cause)
            if not block:
                break
            data += block
        if len(data) != st.st_size:
            _readback_failed(InfrastructureError("short read"))
        if data != expected or hash_bytes(data) != hash_bytes(expected):
            _hash_mismatch()
    finally:
        os.close(fd)


def _readback_png(content: bytes, resolution: tuple[int, int]) -> None:
    from PIL import Image

    with Image.open(BytesIO(content)) as image:
        if image.mode != "RGB":
            _hash_mismatch()
        if getattr(image, "n_frames", 1) != 1:
            _hash_mismatch()
        if image.info != {}:
            _hash_mismatch()
        if (image.width, image.height) != resolution:
            _hash_mismatch()


def export_bundle(
    request: ExportRequest, target_root_fd: int, bundle_name: str, /
) -> ExportBundle:
    bundle_name = _validate_bundle_name(bundle_name)
    root_dup = _dup_root_fd(target_root_fd)
    try:
        root_st = os.fstat(root_dup)
        prepared = _prepare_export(request)
        staging_fd, staging_name = _create_staging(root_dup, root_st)
        expected_dirs: set[str] = set()
        expected_files: set[str] = set()
        committed = False
        try:
            for parts in _ALWAYS_ON_DIRS:
                _mkdir_rel(staging_fd, parts)
                expected_dirs.add("/".join(parts))
            for f in (*prepared.payload_files, prepared.manifest_file):
                for parents in _dir_parents(f.relative_path):
                    _mkdir_rel(staging_fd, parents)
                    expected_dirs.add("/".join(parents))
                expected_files.add(f.relative_path)
                _write_file(staging_fd, f.relative_path, f.content)
            for f in (*prepared.payload_files, prepared.manifest_file):
                parents = tuple(f.relative_path.split("/")[:-1])
                if parents:
                    _fsync_dir(staging_fd, parents)
            for parts in _ALWAYS_ON_DIRS:
                _fsync_dir(staging_fd, parts)
            _fsync_dir(staging_fd, ())
            _readback(staging_fd, prepared, expected_dirs | expected_files)
            _native_rename(root_dup, staging_name, root_dup, bundle_name)
            committed = True
            try:
                os.fsync(root_dup)
            except OSError as cause:
                raise InfrastructureError(
                    "export published but directory fsync failed"
                ) from cause
        except BaseException:
            if not committed:
                _safe_cleanup(
                    staging_fd,
                    root_dup,
                    staging_name,
                    frozenset(expected_files),
                    frozenset(expected_dirs),
                )
            raise
        finally:
            os.close(staging_fd)
        files = tuple(
            sorted(
                (
                    ExportedFile(f.relative_path, f.sha256, f.size_bytes)
                    for f in (*prepared.payload_files, prepared.manifest_file)
                ),
                key=lambda x: x.relative_path,
            )
        )
        return ExportBundle(
            bundle_name,
            root_st.st_dev,
            root_st.st_ino,
            prepared.manifest_file.sha256,
            prepared.payload_sha256,
            prepared.bundle_sha256,
            files,
        )
    finally:
        os.close(root_dup)


def _readback(staging_fd, prepared, expected: set[str]) -> None:
    from specstyle.exporting import qa_report as _qa

    manifest = prepared.manifest_file
    _readback_file(staging_fd, manifest.relative_path, manifest.content)
    _qa.assert_canonical_round_trip(manifest.content)
    res_by_path = _resolution_by_path(manifest.content)
    for f in prepared.payload_files:
        _readback_file(staging_fd, f.relative_path, f.content)
        if f.relative_path.endswith(".json") or f.relative_path.endswith(".yaml"):
            _qa.assert_canonical_round_trip(f.content)
        if f.relative_path.endswith(".png"):
            resolution = res_by_path.get(f.relative_path)
            if resolution is None:
                _hash_mismatch()
            _readback_png(f.content, resolution)
    _verify_inventory(staging_fd, expected)


def _resolution_by_path(manifest_bytes: bytes) -> dict[str, tuple[int, int]]:
    import json

    manifest = json.loads(manifest_bytes)
    result: dict[str, tuple[int, int]] = {}
    for cohort in manifest["cohorts"]:
        for item in cohort["items"]:
            path = item["final_artifact"]["relative_path"]
            result[path] = tuple(item["initial_attempt"]["graph"]["resolution"])
    return result


def _verify_inventory(staging_fd: int, expected: set[str]) -> None:
    seen: set[str] = set()
    _walk(staging_fd, "", seen)
    if seen != expected:
        _hash_mismatch()


def _walk(fd: int, prefix: str, seen: set[str]) -> None:
    try:
        names = os.listdir(fd)
    except OSError as cause:
        _readback_failed(cause)
    for name in names:
        rel = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
        try:
            st = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except OSError as cause:
            _readback_failed(cause)
        seen.add(rel)
        if stat.S_ISDIR(st.st_mode):
            child = _open_dir(fd, name)
            try:
                _walk(child, rel, seen)
            finally:
                os.close(child)


def _open_chain(root: int, parts: tuple[str, ...]) -> tuple[list[int], int]:
    parent = root
    opened: list[int] = []
    for component in parts:
        try:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent,
            )
        except OSError:
            for fd in reversed(opened):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        opened.append(child)
        parent = child
    return opened, parent


def _close_fds(opened: list[int]) -> None:
    for fd in reversed(opened):
        try:
            os.close(fd)
        except OSError:
            pass


def _safe_cleanup(
    staging_fd: int,
    root_dup: int,
    staging_name: str,
    expected_files: frozenset[str],
    expected_dirs: frozenset[str],
) -> None:
    """pre-commit cleanup：只遍历 expected inventory，发现额外 entry/类型变化/symlink/
    OSError 时停止并保留整个隐藏 staging；绝不清理 final。"""
    seen: set[str] = set()
    try:
        _walk(staging_fd, "", seen)
    except Exception:
        return
    if seen != (expected_files | expected_dirs):
        return
    for rel in sorted(expected_files, key=lambda p: p.count("/"), reverse=True):
        parts = tuple(rel.split("/"))
        opened, parent = _open_chain(staging_fd, parts[:-1])
        try:
            st = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(st.st_mode):
                return
            os.unlink(parts[-1], dir_fd=parent)
        except OSError:
            return
        finally:
            _close_fds(opened)
    for parts in sorted(
        {tuple(d.split("/")) for d in expected_dirs},
        key=lambda p: (-len(p), p),
    ):
        opened, parent = _open_chain(staging_fd, parts[:-1])
        try:
            st = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(st.st_mode):
                return
            os.rmdir(parts[-1], dir_fd=parent)
        except OSError:
            return
        finally:
            _close_fds(opened)
    try:
        os.rmdir(staging_name, dir_fd=root_dup)
    except OSError:
        pass
