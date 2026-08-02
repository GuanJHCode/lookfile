#!/usr/bin/env python3
"""Install Lookfile AMD app dependencies without touching the image PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

FORBIDDEN = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch",
        "torch-tensorrt",
        "pytorch-triton",
    }
)
SCHEMA_KEYS = frozenset(
    (
        "schema_version",
        "status",
        "reason_code",
        "repo_sha",
        "lock_sha",
        "planned",
        "installed",
        "skipped",
        "torch_before",
        "torch_after",
        "torch_unchanged",
        "pip_command",
    )
)


def _load_probe():
    path = Path(__file__).with_name("probe_environment.py")
    spec = importlib.util.spec_from_file_location("_amd_probe_environment", path)
    if spec is None or spec.loader is None:
        raise ImportError("probe_environment unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PROBE = _load_probe()
DEPENDENCIES: dict[str, str] = dict(_PROBE.DEPENDENCIES)
canonical_json = _PROBE.canonical_json
torch_fingerprints = _PROBE.torch_fingerprints


def normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def locked_requirements(dependencies: Mapping[str, str] = DEPENDENCIES) -> list[str]:
    planned: list[str] = []
    for name, version in dependencies.items():
        if type(name) is not str or type(version) is not str:
            raise ValueError("invalid dependency pin")
        if normalize_name(name) in FORBIDDEN:
            raise ValueError(f"forbidden package in plan: {name}")
        if not _PROBE.valid_version(version):
            raise ValueError(f"invalid version pin: {name}=={version}")
        planned.append(f"{name}=={version}")
    return planned


def lock_sha(dependencies: Mapping[str, str] = DEPENDENCIES) -> str:
    payload = canonical_json(
        {name: dependencies[name] for name in sorted(dependencies)}
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def current_versions(names: Sequence[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in names:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def packages_to_install(
    dependencies: Mapping[str, str] = DEPENDENCIES,
) -> tuple[list[str], list[str]]:
    install: list[str] = []
    skipped: list[str] = []
    for name, version in dependencies.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            install.append(f"{name}=={version}")
            continue
        if actual == version:
            skipped.append(f"{name}=={version}")
        else:
            install.append(f"{name}=={version}")
    return install, skipped


def snapshot_torch(
    import_torch: Callable[[], Any] = lambda: __import__("torch"),
    fingerprints: Callable[[Any], dict[str, str]] = torch_fingerprints,
) -> dict[str, str] | None:
    try:
        torch = import_torch()
    except Exception:
        return None
    try:
        value = fingerprints(torch)
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    return {str(key): str(item) for key, item in value.items()}


def assert_no_forbidden(specs: Sequence[str]) -> None:
    for spec in specs:
        name = normalize_name(spec.split("==", 1)[0])
        if name in FORBIDDEN:
            raise ValueError(f"refusing to install forbidden package: {spec}")


def build_pip_command(
    specs: Sequence[str], *, python: str = sys.executable
) -> list[str]:
    assert_no_forbidden(specs)
    # --no-deps keeps ROCm torch from being pulled as a transitive requirement.
    return [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-deps",
        *specs,
    ]


def run_pip(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    if not command or command[1:3] != ["-m", "pip"] or "install" not in command:
        raise ValueError("invalid pip command")
    assert_no_forbidden([part for part in command if "==" in part])
    completed = runner(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PIP_NO_INPUT": "1"},
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or "pip install failed")


def empty_result(repo_sha: str | None, lock: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "PASS",
        "reason_code": "OK",
        "repo_sha": repo_sha,
        "lock_sha": lock,
        "planned": [],
        "installed": [],
        "skipped": [],
        "torch_before": None,
        "torch_after": None,
        "torch_unchanged": False,
        "pip_command": [],
    }


def fail(result: dict[str, Any], reason: str) -> tuple[dict[str, Any], int]:
    result = dict(result)
    result["status"] = "FAIL"
    result["reason_code"] = reason
    return result, 2


def install_dependencies(
    *,
    repo_sha: str | None = None,
    dry_run: bool = False,
    require_torch: bool = True,
    dependencies: Mapping[str, str] = DEPENDENCIES,
    import_torch: Callable[[], Any] = lambda: __import__("torch"),
    fingerprints: Callable[[Any], dict[str, str]] = torch_fingerprints,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    python: str = sys.executable,
) -> tuple[dict[str, Any], int]:
    lock = lock_sha(dependencies)
    result = empty_result(repo_sha, lock)
    try:
        planned = locked_requirements(dependencies)
    except ValueError:
        return fail(result, "INVALID_LOCK")
    result["planned"] = planned

    before = snapshot_torch(import_torch, fingerprints)
    result["torch_before"] = before
    if require_torch and before is None:
        return fail(result, "TORCH_MISSING")

    install, skipped = packages_to_install(dependencies)
    result["skipped"] = skipped
    if not install:
        result["installed"] = []
        result["torch_after"] = before
        result["torch_unchanged"] = before is not None
        result["pip_command"] = []
        return result, 0

    try:
        command = build_pip_command(install, python=python)
    except ValueError:
        return fail(result, "FORBIDDEN_PACKAGE")
    result["pip_command"] = command
    result["installed"] = install

    if dry_run:
        result["torch_after"] = before
        result["torch_unchanged"] = before is not None
        result["reason_code"] = "DRY_RUN"
        return result, 0

    try:
        run_pip(command, runner=runner)
    except RuntimeError:
        return fail(result, "PIP_FAILED")

    after = snapshot_torch(import_torch, fingerprints)
    result["torch_after"] = after
    if require_torch and after is None:
        return fail(result, "TORCH_MISSING_AFTER")
    if before is not None and after != before:
        return fail(result, "TORCH_DRIFT")
    result["torch_unchanged"] = before is not None and after == before

    versions = current_versions(list(dependencies))
    for name, expected in dependencies.items():
        if versions.get(name) != expected:
            return fail(result, "VERSION_MISMATCH")
    return result, 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Lookfile dependencies without modifying ROCm torch"
    )
    parser.add_argument("--repo-sha", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-torch",
        action="store_true",
        help="For offline unit tests only; AMD install must see torch",
    )
    parser.add_argument("--json-out", default=None, help="Optional evidence path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, code = install_dependencies(
        repo_sha=args.repo_sha,
        dry_run=args.dry_run,
        require_torch=not args.allow_missing_torch,
    )
    payload = canonical_json(result) + "\n"
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="ascii")
        os.chmod(path, 0o600)
    else:
        sys.stdout.write(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
