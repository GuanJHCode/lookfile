"""Local weight resolution — no network, no auto-download, fail closed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from specstyle.domain.identifiers import Sha256
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.model_registry import ModelDescriptor
from specstyle.observability.hashing import hash_file


@dataclass(frozen=True, slots=True)
class ResolvedWeight:
    model_id: str
    path: Path
    content_sha256: Sha256
    kind: str  # file | directory_marker


def _validate_relpath(relpath: str) -> str:
    if type(relpath) is not str or not relpath or relpath.startswith("/"):
        raise DomainError("invalid weights relpath")
    parts = relpath.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise DomainError("weights path escape")
    return "/".join(parts)


def resolve_weight(
    cache_root: Path,
    descriptor: ModelDescriptor,
    relpath: str,
    *,
    marker_name: str = "WEIGHTS.sha256",
) -> ResolvedWeight:
    """Resolve a model under ``cache_root/relpath``.

    Accepts either a single regular file whose SHA matches ``expected_sha256``,
    or a directory containing ``marker_name`` whose first 64 hex chars match.
    Never follows the final path component as a symlink (O_NOFOLLOW when open).
    """
    if not isinstance(cache_root, Path) or type(descriptor) is not ModelDescriptor:
        raise DomainError("invalid weight resolve inputs")
    if type(marker_name) is not str or not marker_name or "/" in marker_name:
        raise DomainError("invalid marker name")
    safe = _validate_relpath(relpath)
    root = cache_root
    if not root.is_dir():
        raise InfrastructureError("model cache root missing")
    target = root.joinpath(*safe.split("/"))
    try:
        st = os.lstat(target)
    except OSError as exc:
        raise InfrastructureError("model weights missing") from exc
    if stat_is_symlink(st):
        raise InfrastructureError("model weights symlink refused")
    if stat_is_file(st):
        digest = hash_file(root, safe)
        if digest != descriptor.expected_sha256:
            raise InfrastructureError("model weight hash mismatch")
        return ResolvedWeight(descriptor.model_id, target, digest, "file")
    if stat_is_dir(st):
        marker_rel = f"{safe}/{marker_name}"
        marker = root.joinpath(*marker_rel.split("/"))
        try:
            mst = os.lstat(marker)
        except OSError as exc:
            raise InfrastructureError("model weight marker missing") from exc
        if stat_is_symlink(mst) or not stat_is_file(mst):
            raise InfrastructureError("model weight marker invalid")
        text = _read_marker(marker)
        digest = Sha256(text[:64].lower())
        if digest != descriptor.expected_sha256:
            raise InfrastructureError("model weight hash mismatch")
        return ResolvedWeight(descriptor.model_id, target, digest, "directory_marker")
    raise InfrastructureError("model weights unsupported type")


def resolve_graph_weights(
    cache_root: Path,
    descriptors: tuple[ModelDescriptor, ...],
    relpaths: dict[str, str],
) -> tuple[ResolvedWeight, ...]:
    if type(descriptors) is not tuple or type(relpaths) is not dict:
        raise DomainError("invalid graph weight inputs")
    out: list[ResolvedWeight] = []
    for desc in descriptors:
        if type(desc) is not ModelDescriptor:
            raise DomainError("invalid descriptor")
        if desc.model_id not in relpaths:
            raise DomainError("missing weights relpath")
        out.append(resolve_weight(cache_root, desc, relpaths[desc.model_id]))
    return tuple(out)


def _read_marker(path: Path) -> str:
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
    except OSError as exc:
        raise InfrastructureError("model weight marker open failed") from exc
    try:
        raw = os.read(fd, 128)
    finally:
        os.close(fd)
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise InfrastructureError("model weight marker corrupt") from exc
    if len(text) < 64:
        raise InfrastructureError("model weight marker corrupt")
    return text


def stat_is_symlink(st: os.stat_result) -> bool:
    import stat as statmod

    return statmod.S_ISLNK(st.st_mode)


def stat_is_file(st: os.stat_result) -> bool:
    import stat as statmod

    return statmod.S_ISREG(st.st_mode)


def stat_is_dir(st: os.stat_result) -> bool:
    import stat as statmod

    return statmod.S_ISDIR(st.st_mode)
