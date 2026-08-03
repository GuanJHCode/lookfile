"""Offline CLI for formal evaluation evidence; it never runs generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Sequence

from specstyle.errors import DomainError
from specstyle.evaluation.evidence import (
    import_blind_labels,
    validate_machine_ledger,
)
from specstyle.evaluation.protocol import prepare_protocol, seal_protocol
from specstyle.evaluation.stats import finalize_evaluation
from specstyle.production.context_config import load_production_context_config

_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _add_evidence_inputs(parser: argparse.ArgumentParser, *, labels: bool) -> None:
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--sealed-protocol", required=True, type=Path)
    parser.add_argument("--machine-ledger", required=True, type=Path)
    if labels:
        parser.add_argument("--blind-labels", required=True, type=Path)
        parser.add_argument("--label-approval-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-protocol")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    seal = commands.add_parser("seal-protocol")
    seal.add_argument("--protocol", required=True, type=Path)
    seal.add_argument("--config-root", required=True, type=Path)
    seal.add_argument("--context-evidence-root", required=True, type=Path)
    seal.add_argument("--sealed-at", required=True)
    seal.add_argument("--repo-sha", required=True)
    seal.add_argument("--out", required=True, type=Path)
    _add_evidence_inputs(commands.add_parser("validate-machine-ledger"), labels=False)
    _add_evidence_inputs(commands.add_parser("import-blind-labels"), labels=True)
    _add_evidence_inputs(commands.add_parser("finalize"), labels=True)
    return parser


def _read(path: Path, name: str) -> bytes:
    try:
        descriptor = os.open(path, _READ_FLAGS)
    except OSError as exc:
        raise DomainError(f"unable to read {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 0 < metadata.st_size <= _MAX_EVIDENCE_BYTES
        ):
            raise DomainError(f"invalid {name} file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise DomainError(f"invalid {name} file")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise DomainError(f"invalid {name} file")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short evidence write")
        view = view[written:]


def _write_new(path: Path, content: bytes) -> None:
    try:
        parent_fd = os.open(path.parent, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise DomainError("invalid evaluation output directory") from exc
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(path.name, _WRITE_FLAGS, 0o600, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise DomainError("evaluation output already exists") from exc
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(parent_fd)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(path.name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    finally:
        os.close(parent_fd)


def _load_production_context(config_root: Path, evidence_root: Path) -> object:
    descriptors: list[int] = []
    try:
        descriptors = [
            os.open(config_root, _DIRECTORY_FLAGS),
            os.open(evidence_root, _DIRECTORY_FLAGS),
        ]
        return load_production_context_config(*descriptors)
    except OSError as exc:
        raise DomainError("invalid Production context roots") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _evaluate(arguments: argparse.Namespace) -> bytes:
    protocol = _read(arguments.protocol, "evaluation protocol")
    sealed = _read(arguments.sealed_protocol, "sealed protocol")
    ledger = _read(arguments.machine_ledger, "machine ledger")
    if arguments.command == "validate-machine-ledger":
        return validate_machine_ledger(protocol, sealed, ledger)
    labels = _read(arguments.blind_labels, "blind labels")
    approval = _read(arguments.label_approval_receipt, "label approval receipt")
    if arguments.command == "import-blind-labels":
        return import_blind_labels(protocol, sealed, ledger, labels, approval)
    return finalize_evaluation(protocol, sealed, ledger, labels, approval)


def _content(arguments: argparse.Namespace) -> bytes:
    if arguments.command == "prepare-protocol":
        return prepare_protocol(_read(arguments.input, "protocol draft"))
    if arguments.command == "seal-protocol":
        context = _load_production_context(
            arguments.config_root, arguments.context_evidence_root
        )
        return seal_protocol(
            _read(arguments.protocol, "evaluation protocol"),
            context,
            sealed_at=arguments.sealed_at,
            repo_sha=arguments.repo_sha,
        )
    return _evaluate(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, seal, join, or summarize offline evaluation evidence."""
    arguments = _argument_parser().parse_args(argv)
    content = _content(arguments)
    _write_new(arguments.out, content)
    status = json.loads(content).get("status")
    return 2 if status in {"BLIND_LABELS_INCOMPLETE", "INCOMPLETE", "TEST_ONLY"} else 0


def run() -> None:
    try:
        raise SystemExit(main())
    except DomainError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    run()
