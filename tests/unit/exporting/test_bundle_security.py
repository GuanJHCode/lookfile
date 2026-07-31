"""EXP-001B bundle security tests (§13.11, §13.12)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from specstyle.errors import DomainError, InfrastructureError
from specstyle.exporting.bundle import export_bundle
from tests.unit.exporting.test_bundle import _root_fd
from tests.unit.exporting.test_manifest import _export_request


def test_two_writers_same_name_exactly_one_wins(tmp_path: Path) -> None:
    root_fd = _root_fd(tmp_path)
    try:
        bundle = export_bundle(_export_request(), root_fd, "shared")
    finally:
        os.close(root_fd)
    assert bundle.bundle_name == "shared"
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(DomainError, match="export target exists"):
            export_bundle(_export_request(), root_fd, "shared")
    finally:
        os.close(root_fd)
    assert (tmp_path / "shared").is_dir()


def test_native_no_replace_leaves_no_partial_final(tmp_path: Path) -> None:
    (tmp_path / "collide").mkdir()
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(DomainError, match="export target exists"):
            export_bundle(_export_request(), root_fd, "collide")
    finally:
        os.close(root_fd)
    # 原目录未被覆盖（仍为空目录，无 manifest）
    assert not (tmp_path / "collide" / "manifest.json").exists()


def test_short_write_raises_and_leaves_no_final(tmp_path: Path, monkeypatch) -> None:
    real_write = os.write
    state = {"n": 0}

    def short_write(fd: int, data: bytes) -> int:
        state["n"] += 1
        if state["n"] == 1 and data:
            return 0
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", short_write)
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(InfrastructureError, match="export write failed"):
            export_bundle(_export_request(), root_fd, "short")
    finally:
        os.close(root_fd)
    assert not (tmp_path / "short").exists()


def test_fsync_failure_raises_sync_failed(tmp_path: Path, monkeypatch) -> None:
    def boom(fd: int) -> None:
        raise OSError("sync denied")

    monkeypatch.setattr(os, "fsync", boom)
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(InfrastructureError, match="export sync failed"):
            export_bundle(_export_request(), root_fd, "syncfail")
    finally:
        os.close(root_fd)
    assert not (tmp_path / "syncfail").exists()


def test_readback_failure_raises_readback_failed(tmp_path: Path, monkeypatch) -> None:
    def boom(fd: int, n: int) -> bytes:
        raise OSError("read denied")

    # 只在 readback 阶段（写完后）触发；写阶段 os.read 不被调用。
    monkeypatch.setattr(os, "read", boom)
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(InfrastructureError, match="export readback failed"):
            export_bundle(_export_request(), root_fd, "rback")
    finally:
        os.close(root_fd)
    assert not (tmp_path / "rback").exists()


def test_root_fd_not_directory_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("x")
    fd = os.open(os.fspath(file_path), os.O_RDONLY)
    try:
        with pytest.raises(DomainError, match="invalid export target"):
            export_bundle(_export_request(), fd, "x")
    finally:
        os.close(fd)


def test_bundle_name_rejects_path_escape(tmp_path: Path) -> None:
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(DomainError, match="invalid export target"):
            export_bundle(_export_request(), root_fd, "../escape")
    finally:
        os.close(root_fd)
    assert not (tmp_path.parent / "escape").exists()


def test_post_rename_fsync_failure_keeps_final(tmp_path: Path, monkeypatch) -> None:
    import specstyle.exporting.bundle as bundle_mod

    real_rename = bundle_mod._native_rename
    real_fsync = os.fsync
    state = {"renamed": False}

    def rename_wrapper(from_fd, from_name, to_fd, to_name):
        result = real_rename(from_fd, from_name, to_fd, to_name)
        state["renamed"] = True
        return result

    def fsync_boom(fd: int) -> None:
        if state["renamed"]:
            raise OSError("post-rename fsync denied")
        return real_fsync(fd)

    monkeypatch.setattr(bundle_mod, "_native_rename", rename_wrapper)
    monkeypatch.setattr(os, "fsync", fsync_boom)
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(
            InfrastructureError, match="export published but directory fsync failed"
        ):
            export_bundle(_export_request(), root_fd, "postren")
    finally:
        os.close(root_fd)
    # final must NOT be deleted (§13.11)
    final = tmp_path / "postren"
    assert final.is_dir()
    assert (final / "manifest.json").exists()


def test_native_unavailable_fails_closed(tmp_path: Path, monkeypatch) -> None:
    import specstyle.exporting.bundle as bundle_mod

    def boom(*a: object, **k: object) -> None:
        raise InfrastructureError("secure atomic publication unavailable")

    monkeypatch.setattr(bundle_mod, "_native_rename", boom)
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(
            InfrastructureError, match="secure atomic publication unavailable"
        ):
            export_bundle(_export_request(), root_fd, "native")
    finally:
        os.close(root_fd)
    assert not (tmp_path / "native").exists()
    assert not list(tmp_path.iterdir())


def test_readback_mismatch_does_not_publish(tmp_path: Path, monkeypatch) -> None:
    import specstyle.exporting.bundle as bundle_mod

    def boom(*a: object, **k: object) -> None:
        raise DomainError("export hash mismatch")

    monkeypatch.setattr(bundle_mod, "_readback_file", boom)
    root_fd = _root_fd(tmp_path)
    try:
        with pytest.raises(DomainError, match="export hash mismatch"):
            export_bundle(_export_request(), root_fd, "mismatch")
    finally:
        os.close(root_fd)
    assert not (tmp_path / "mismatch").exists()
    assert not list(tmp_path.iterdir())
